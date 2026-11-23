"""DeepHS-Fruit v2 (Varga et al.): hyperspectral cubes of five fruits, ripeness labels.

Raw ENVI cubes (BIP, float32 reflectance) of avocados, kakis, kiwis, mangos and
papayas, one cube per face of a fruit and per camera (``VIS``: 224 bands,
``NIR``: 252, ``VIS_COR``: 249), already cut out on a zero background. The
records come from the v2 annotation files (``annotations/*_v2.json``, update of
2024-01-09); ``merged_annotations.json`` is not used (it merges v1 and v2 and
gives one file several fruits).

Construction:

1. Fruit mask: Otsu threshold on the mean of bands 150-199 (VIS, about
   800-930 nm) or of all bands (other cameras), over the whole frame; it also
   removes the dark rim of the fruit.
2. Reflectance clipped to [0, 1] (aberrant pixels outside this range).
3. Bands averaged by groups of ``band_bin`` (default 1: p = 224 in VIS).
   One covariance per cube, observations = the pixels of the mask.

Labels: ``classification_mode="fruit"`` (5 fruits, every record of the camera in
``train_all_v2.json``: 3 193 VIS cubes) or ``"ripeness"`` (``unripe``,
``perfect``, ``overripe``; ``ripe`` is merged into ``perfect``; the annotated
cubes of ``train_only_labeled_v2``, ``val_v2`` and ``test_v2``: 596 in VIS).

``groups`` = the physical fruit ``<Fruit>|<fruit>_<day>_<k>`` (its two faces).
Within the v2 annotations this pair maps to one fruit (one weight, one
firmness, two faces); the number ``k`` is reused by other fruits on other days,
and a fruit recorded on several days cannot be traced.

Splits:

- ``"by_fruit"`` = ``"recommended"``: StratifiedGroupKFold(``n_splits``) grouped
  by physical fruit (``fold`` = test fold, the next one = validation);
- ``"official"`` (ripeness mode): the train / val / test files of the authors. It
  leaks: in VIS, 46 of the 72 validation cubes and 46 of the 140 test cubes
  have the other face of their fruit in the training part.

Download
--------
Manual (226 GB, licence not stated): the sources are listed at
https://github.com/cogsys-tuebingen/deephs_fruit (HTTP index
https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/ and its
``.torrent``); the server did not answer on 2026-11-15 and 2026-11-23. Without a
local copy, ``fetch_deephsfruit()`` opens the repository page and raises
:class:`~spdnet_datasets.core.access.ManualDownloadRequired`; ``archive=`` accepts the
downloaded folder (or a zip of it) with the layout of the authors
(``annotations/``, ``<Fruit>/<camera>/<day>/*.hdr|.bin``)::

    fetch_deephsfruit(archive="~/copies/DeepHS-Fruit")
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import dataset_path
from ..core.fetch import Bunch, run_pipeline
from ..manager import DatasetManager
from ..utils.sources import existing_dir, source_dir
from ._common import (
    HSIDataset,
    covariance_stage,
    envi_cube,
    group_kfold_split,
    load_covariance_dir,
    map_splits,
    write_json,
)

DOMAIN, NAME = "HYPERSPECTRAL", "DeepHS-Fruit"
PAGE = "https://github.com/cogsys-tuebingen/deephs_fruit"
FILES = {}  # manual source (server of the authors, no checksum registry)
FRUIT_CLASSES = ["Avocado", "Kaki", "Kiwi", "Mango", "Papaya"]
CAMERAS = {"VIS": 224, "NIR": 252, "VIS_COR": 249}
RIPENESS = {"unripe": "unripe", "perfect": "perfect", "ripe": "perfect",
            "overripe": "overripe"}
LABELLED = {"train": "train_only_labeled_v2.json", "val": "val_v2.json", "test": "test_v2.json"}
ALL_RECORDS = "train_all_v2.json"


def physical_fruit(header_file):
    """``<Fruit>|<fruit>_<day>_<k>``: the two faces of one fruit share it."""
    return f"{Path(header_file).parts[0]}|{Path(header_file).stem.rsplit('_', 1)[0]}"


def otsu_threshold(values, bins=256):
    """Otsu (1979): upper edge of the last bin of the low class."""
    hist, edges = np.histogram(values, bins)
    centres = (edges[:-1] + edges[1:]) / 2
    weight, moment = np.cumsum(hist), np.cumsum(hist * centres)
    between = (moment[-1] * weight / weight[-1] - moment) ** 2 / np.maximum(
        weight * (weight[-1] - weight), 1)
    return float(edges[np.argmax(between) + 1])


def fruit_mask(cube, camera="VIS"):
    """Boolean (lines, samples) mask of the fruit (see the module docstring)."""
    bands = slice(150, 200) if camera == "VIS" else slice(None)
    brightness = np.asarray(cube[..., bands], dtype=np.float64).mean(-1)
    return brightness > otsu_threshold(brightness)


def fruit_pixels(root, header_file, camera="VIS", band_bin=1):
    """Clipped reflectances (n, p) of the fruit of one cube."""
    cube = envi_cube(Path(root) / header_file)
    X = np.clip(np.asarray(cube[fruit_mask(cube, camera)], dtype=np.float32), 0.0, 1.0)
    if band_bin > 1:
        p = X.shape[1] // band_bin
        X = X[:, :p * band_bin].reshape(len(X), p, band_bin).mean(-1)
    return X


def records(root, camera="VIS", classification_mode="fruit"):
    """One row per cube of ``camera`` (header, fruit, day, side, physical fruit,
    official part and ripeness when annotated)."""
    ann = Path(root) / "annotations"
    rows = {}
    if classification_mode == "fruit":
        for rec in json.loads((ann / ALL_RECORDS).read_text())["records"]:
            if rec["camera_type"] == camera:
                rows[rec["files"]["header_file"]] = {}
    for part, name in LABELLED.items():
        data = json.loads((ann / name).read_text())
        by_id = {r["id"]: r for r in data["records"]}
        for a in data["annotations"]:
            rec = by_id[a["record_id"]]
            header = rec["files"]["header_file"]
            if rec["camera_type"] != camera or a.get("ripeness_state") not in RIPENESS:
                continue
            if classification_mode == "ripeness" or header in rows:
                rows[header] = {"official": part, "ripeness": RIPENESS[a["ripeness_state"]],
                                "firmness": a.get("firmness"),
                                "storage_days": a.get("storage_days")}
    out = [{"header_file": h, "fruit": Path(h).parts[0], "day": Path(h).parts[2],
            "side": Path(h).stem.rsplit("_", 1)[1], "physical_fruit": physical_fruit(h), **r}
           for h, r in sorted(rows.items())]
    return pd.DataFrame(out).reindex(
        columns=["header_file", "fruit", "day", "side", "physical_fruit", "official",
                 "ripeness", "firmness", "storage_days"])


def fetch_deephsfruit(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                      verify="size", *, classification_mode="fruit", camera="VIS",
                      band_bin=1, n_splits=5, fold=0, seed=0, archive=None):
    """Fruit covariances of DeepHS-Fruit v2, computed from the raw cubes.

    Parameters
    ----------
    classification_mode : {"fruit", "ripeness"}
    camera : {"VIS", "NIR", "VIS_COR"}
    band_bin : int
        Bands averaged by groups of ``band_bin`` (1: all bands).
    n_splits, fold, seed
        Folds of the ``by_fruit`` split.
    archive : path, optional
        Downloaded folder (or zip) with the layout of the authors.

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means``, ``labels``,
    ``groups`` (physical fruit), ``meta``, ``info``, ``splits`` and ``root``.
    """
    if classification_mode not in ("fruit", "ripeness"):
        raise ValueError("classification_mode must be 'fruit' or 'ripeness'")
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {sorted(CAMERAS)}")
    if band_bin < 1:
        raise ValueError("band_bin must be >= 1")
    root = source_dir(
        NAME, dataset_path(DOMAIN, NAME, data_dir),
        required=("annotations/" + ALL_RECORDS, "annotations/" + LABELLED["test"]),
        page_url=PAGE, expected="the v2 folder of the authors (annotations/ and the fruit "
                                "folders <Fruit>/<camera>/<day>/, 226 GB)",
        cache_dir=cache_dir, archive=archive, verify=verify)
    params = {"classification_mode": classification_mode, "camera": camera,
              "band_bin": band_bin, "source": str(Path(root).resolve()), "version": 2}

    def process(_, out_dir):
        meta = records(root, camera, classification_mode)
        meta["label"] = meta["fruit"] if classification_mode == "fruit" else meta["ripeness"]
        meta.to_csv(Path(out_dir) / "meta.csv", index=False)
        write_json(Path(out_dir) / "info.json", {
            "n_cubes": len(meta), "camera": camera,
            "p": CAMERAS[camera] // band_bin, "band_bin": band_bin})

    def pixels(_, meta):
        for header in meta["header_file"]:
            yield fruit_pixels(root, header, camera, band_bin)

    res = run_pipeline(NAME, params=params, process_fn=process,
                       covariance_fn=lambda pr, o: covariance_stage(pr, o, pixels, estimator),
                       covariance_params={"estimator": estimator},
                       cache_dir=cache_dir, keep=keep, force=force)
    out = load_covariance_dir(res.covariance_dir)
    meta, meta_all = out.meta, out.meta_all
    by_fruit = group_kfold_split(meta_all["label"].to_numpy(),
                                 meta_all["physical_fruit"].to_numpy(),
                                 n_splits=n_splits, fold=fold, seed=seed)
    splits = {"by_fruit": by_fruit, "recommended": by_fruit}
    if classification_mode == "ripeness":
        splits["official"] = {k: np.flatnonzero(meta_all["official"].to_numpy() == k)
                              for k in LABELLED}
    return Bunch(
        covariances=out.covariances, means=out.means, n_obs=out.n_obs,
        labels=meta["label"].to_numpy(), groups=meta["physical_fruit"].to_numpy(),
        meta=meta, info=out.info,
        splits=map_splits(splits, meta["proc_index"], len(meta_all)),
        processed_path=res.processed_dir, covariance_path=res.covariance_dir, root=root,
    )


@DatasetManager.register_dataset("deephsfruit")
class DeepHSFruitDataset(HSIDataset):
    """DeepHS-Fruit v2 fruit covariances (224 x 224 in VIS), split by physical fruit.

    Parameters
    ----------
    classification_mode : {"fruit", "ripeness"}
    camera : {"VIS", "NIR", "VIS_COR"}
        ``modality`` is accepted as its former name.
    band_bin, n_splits, fold
        See :func:`fetch_deephsfruit`.
    split : str
        ``"recommended"`` (= ``"by_fruit"``) or, in ripeness mode, ``"official"``.
    subset
        See :class:`~spdnet_datasets.hyperspectral._common.HSIDataset`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    FRUIT_CLASSES = FRUIT_CLASSES
    RIPENESS_CLASSES = ["unripe", "perfect", "overripe"]

    def __init__(self, data_dir=None, *, classification_mode="fruit", camera="VIS",
                 band_bin=1, n_splits=5, fold=0, split="recommended", subset=None,
                 cache_dir=None, keep=None, force=False, verify="size", archive=None,
                 modality=None, **kwargs):
        camera = modality or camera
        self.camera = self.modality = camera
        self.classification_mode = classification_mode
        self.band_bin = band_bin
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(data_dir=str(existing_dir(root, cache_dir)), **kwargs)
        bunch = fetch_deephsfruit(root, cache_dir, keep, self.estimator, force, verify,
                                  classification_mode=classification_mode, camera=camera,
                                  band_bin=band_bin, n_splits=n_splits, fold=fold,
                                  seed=self.seed, archive=archive)
        self.data_dir = Path(bunch.root)
        self._setup(bunch, split, subset)

    def _load_raw(self, k):
        """Clipped reflectances (n_pixels, p) of the fruit of sample ``k``."""
        return fruit_pixels(self.bunch.root, self.meta_all["header_file"].iloc[k],
                            self.camera, self.band_bin)


__all__ = ["DeepHSFruitDataset", "fetch_deephsfruit", "fruit_mask", "fruit_pixels", "records"]
