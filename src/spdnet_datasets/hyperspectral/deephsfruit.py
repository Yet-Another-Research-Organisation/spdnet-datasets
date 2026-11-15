"""DeepHS-Fruit (Varga et al.): hyperspectral cubes of five fruits, ripeness labels.

.. warning::
   Minimal adaptation only (QUESTIONS U4). The covariances are the historical
   precomputed ``cov_VIS/*.pt`` files (VIS camera, 224 bands), read
   **read-only**; their construction (pixel mask, estimator, scaling) is not
   documented and has not been re-verified. No raw data or channel means are
   served.

Files ``cov_VIS/<fruit>_VIS_<day>_<record>_<side>.pt``: ``record`` is the id of
the record in ``annotations/merged_annotations.json`` (not in the v2 files,
which number their records differently). A record gives the ENVI header
``<Fruit>/VIS/<day>/<fruit>_<day>_<k>_<side>.hdr``; the **physical fruit** is
``<Fruit>|<fruit>_<day>_<k>`` (both faces). Ripeness labels and the official
v2 split come from ``annotations/{train_only_labeled,val,test}_v2.json``,
matched by header file (``ripe`` is merged into ``perfect``).

Labels: ``classification_mode="fruit"`` (5 fruits, every covariance) or
``"ripeness"`` (3 states, the annotated VIS cubes only).

Splits: ``"by_fruit"`` = ``"recommended"`` (StratifiedGroupKFold(5) by
physical fruit). The official v2 split cannot be used: the VIS cubes of
``val_v2`` (72) and ``test_v2`` (140) are absent from ``merged_annotations``
and have no ``cov_VIS`` file, so ripeness mode holds the 384 annotated VIS
cubes of ``train_only_labeled_v2`` only (it would also leak: 46 of the 140
test cubes have the other face of their fruit in train).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core.config import dataset_path
from ..core.fetch import Bunch, verify_local
from ..manager import DatasetManager
from ._common import HSIDataset, group_kfold_split

DOMAIN, NAME = "HYPERSPECTRAL", "DeepHS-Fruit"
FRUIT_CLASSES = ["Avocado", "Kaki", "Kiwi", "Mango", "Papaya"]
RIPENESS = {"unripe": "unripe", "perfect": "perfect", "ripe": "perfect",
            "overripe": "overripe"}
V2_FILES = {"train": "train_only_labeled_v2.json", "val": "val_v2.json", "test": "test_v2.json"}
WARNING = ("DeepHS-Fruit: historical precomputed covariances (cov_VIS/*.pt) of "
           "undocumented construction, adapted to the interface without re-verification.")


class _LazyCovariances:
    """Covariances read on demand from ``.pt`` files (float64)."""

    def __init__(self, paths):
        self.paths = list(paths)

    def _one(self, k):
        c = torch.load(self.paths[int(k)], weights_only=False)
        return np.asarray(c.double().numpy() if torch.is_tensor(c) else c, dtype=np.float64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        if np.ndim(k) == 0:
            return self._one(k)
        return np.stack([self._one(i) for i in np.asarray(k)])


def _physical_fruit(header_file):
    stem = Path(header_file).stem
    return f"{Path(header_file).parts[0]}|{stem.rsplit('_', 1)[0]}"


def fetch_deephsfruit(data_dir=None, verify="size", *, classification_mode="fruit",
                      n_splits=5, fold=0, seed=0):
    """Index of the precomputed DeepHS-Fruit VIS covariances (no computation).

    Returns
    -------
    Bunch with ``covariances`` (lazy, indexable), ``labels``, ``groups``
    (physical fruit), ``meta`` and ``splits``.
    """
    if classification_mode not in ("fruit", "ripeness"):
        raise ValueError("classification_mode must be 'fruit' or 'ripeness'")
    root = dataset_path(DOMAIN, NAME, data_dir)
    verify_local(root, verify=verify,
                 required=("cov_VIS", "annotations/merged_annotations.json"))
    merged = json.loads((root / "annotations/merged_annotations.json").read_text())
    records = {r["id"]: r for r in merged["records"]}
    ripeness, official = {}, {}
    for part, name in V2_FILES.items():
        ann = json.loads((root / "annotations" / name).read_text())
        by_id = {r["id"]: r for r in ann["records"]}
        for a in ann["annotations"]:
            rec = by_id[a["record_id"]]
            if a.get("ripeness_state") in RIPENESS:
                ripeness[rec["files"]["header_file"]] = RIPENESS[a["ripeness_state"]]
                official[rec["files"]["header_file"]] = part
    rows = []
    for path in sorted((root / "cov_VIS").glob("*.pt")):
        rec = records.get(int(path.stem.split("_")[-2]))
        if rec is None or rec["camera_type"] != "VIS":
            continue
        header = rec["files"]["header_file"]
        rows.append({"file": path.name, "record": rec["id"], "fruit": rec["fruit"],
                     "day": rec["day"], "side": rec["side"], "header_file": header,
                     "physical_fruit": _physical_fruit(header),
                     "ripeness": ripeness.get(header), "official": official.get(header)})
    meta = pd.DataFrame(rows)
    if classification_mode == "ripeness":
        meta = meta[meta["ripeness"].notna()].reset_index(drop=True)
        labels = meta["ripeness"].to_numpy()
    else:
        labels = meta["fruit"].to_numpy()
    groups = meta["physical_fruit"].to_numpy()
    by_fruit = group_kfold_split(labels, groups, n_splits=n_splits, fold=fold, seed=seed)
    splits = {"by_fruit": by_fruit, "recommended": by_fruit}
    official = {k: np.flatnonzero(meta["official"].to_numpy() == k) for k in V2_FILES}
    if classification_mode == "ripeness" and len(official["val"]) and len(official["test"]):
        splits["official"] = official  # absent locally: no cov_VIS file for val/test cubes
    return Bunch(covariances=_LazyCovariances(root / "cov_VIS" / f for f in meta["file"]),
                 means=None, labels=labels, groups=groups, meta=meta, splits=splits,
                 info={"n_covariances": len(meta)})


@DatasetManager.register_dataset("deephsfruit")
class DeepHSFruitDataset(HSIDataset):
    """DeepHS-Fruit VIS covariances (precomputed, 224 x 224), split by physical fruit.

    Parameters
    ----------
    classification_mode : {"fruit", "ripeness"}
    modality : {"VIS"}
        Only the VIS covariances exist.
    n_splits, fold
        See :func:`fetch_deephsfruit`.
    split, subset
        See :class:`~spdnet_datasets.hyperspectral._common.HSIDataset`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options (``return_mean`` and
        ``return_raw`` are not available).
    """

    OUTPUTS = ("cov",)
    FRUIT_CLASSES = FRUIT_CLASSES
    RIPENESS_CLASSES = ["unripe", "perfect", "overripe"]

    def __init__(self, data_dir=None, *, classification_mode="fruit", modality="VIS",
                 n_splits=5, fold=0, split="recommended", subset=None, verify="size",
                 **kwargs):
        if modality != "VIS":
            raise ValueError(f"Only VIS modality is supported, got {modality}")
        warnings.warn(WARNING, UserWarning, stacklevel=2)
        self.modality = modality
        self.classification_mode = classification_mode
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(data_dir=str(root), **kwargs)
        bunch = fetch_deephsfruit(root, verify, classification_mode=classification_mode,
                                  n_splits=n_splits, fold=fold)
        self._setup(bunch, split, subset)


__all__ = ["DeepHSFruitDataset", "fetch_deephsfruit"]
