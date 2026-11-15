"""Wheat disease (ICPR 2026 "Beyond Visible Spectrum: AI for Agriculture", task 1).

Local copy ``HYPERSPECTRAL/Wheat_Disease_ICPR26``: ``{train,val}/HS/<stem>.tif``
hyperspectral patches 32 x 32 with 125 or 126 bands (uint16), plus MS and RGB
patches (not used here). Train: 600 labelled patches (``Health``, ``Rust``,
``Other``: 200 each, class in the file name); val: 300 patches without labels
(competition submission), available through ``fetch_kaggle_wheat(part="val")``.

Construction (same band window as the former precomputation):

1. The 126th band, when present, is dropped (125 bands), then the first
   ``skip_first`` (8) and last ``skip_last`` (10) noisy bands: p = 107.
2. Observations: the non-zero pixels of the patch (n <= 1024).
   **Entirely zero patches** (train: Health 9, Other 14) are excluded, as are
   patches with fewer than 2p non-zero pixels; both are counted. The former
   precomputation kept them as zero (non SPD) matrices.
3. Centred SCM (or another ``estimator``).

Split: no plot, flight or position is available, so the patches are split
with a stratified K-fold (``"stratified"`` = ``"recommended"``; test = fold
``fold``, val = fold ``fold + 1``). Neighbouring patches of one plot may fall
on both sides: the leakage cannot be measured (atlas fiche).

Download (manual): Kaggle competition of the ICPR 2026 challenge
(https://han-research.gitlab.io/Agvision/; Kaggle account and acceptance of the
competition rules required). Download ``train/`` and ``val/`` and place it under
``data_dir`` (default ``<data root>/HYPERSPECTRAL/Wheat_Disease_ICPR26``).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import dataset_path
from ..core.estimators import check_sample_size
from ..core.fetch import Bunch, check_local, run_pipeline
from ..manager import DatasetManager
from ._common import (
    HSIDataset,
    covariance_stage,
    group_kfold_split,
    load_covariance_dir,
    map_splits,
    valid_pixels,
    write_json,
)

DOMAIN, NAME = "HYPERSPECTRAL", "Wheat_Disease_ICPR26"
FILES = {}  # Kaggle competition data (account required): manual download
CLASSES = ("Health", "Other", "Rust")


def read_patch(path, skip_first=8, skip_last=10):
    """HS patch (32, 32, 125 - skip_first - skip_last) float32."""
    import tifffile

    hs = tifffile.imread(path).astype(np.float32)[..., :125]
    return hs[..., skip_first:125 - skip_last]


def fetch_kaggle_wheat(data_dir=None, cache_dir=None, keep=None, estimator="scm",
                       force=False, verify="size", *, part="train", skip_first=8,
                       skip_last=10, n_splits=5, fold=0, seed=0):
    """Patch covariances of the wheat disease dataset.

    Parameters
    ----------
    part : {"train", "val"}
        ``"val"`` is the unlabelled competition set (labels ``None``, no split).

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means``, ``labels``,
    ``groups`` (one per patch), ``meta`` (``stem``, ``n_valid``), ``info``
    (excluded zero / short patches) and ``splits``.
    """
    if part not in ("train", "val"):
        raise ValueError("part must be 'train' or 'val'")
    root = dataset_path(DOMAIN, NAME, data_dir)
    check_local(root, verify=verify, required=(f"{part}/HS",))
    p = 125 - skip_first - skip_last
    check_sample_size(32 * 32, p)
    params = {"part": part, "skip_first": skip_first, "skip_last": skip_last,
              "estimator": estimator, "version": 2}

    def process(_, out_dir):
        rows, zero, short = [], 0, 0
        for path in sorted((root / part / "HS").glob("*.tif")):
            X = valid_pixels(read_patch(path, skip_first, skip_last).reshape(-1, p))
            if len(X) == 0:
                zero += 1
                continue
            if len(X) < 2 * p:
                short += 1
                continue
            label = path.stem.split("_")[0] if part == "train" else None
            rows.append({"stem": path.stem, "label": label, "n_valid": len(X)})
        pd.DataFrame(rows).to_csv(Path(out_dir) / "meta.csv", index=False)
        write_json(Path(out_dir) / "info.json", {
            "part": part, "n_patches": len(rows) + zero + short, "excluded_all_zero": zero,
            "excluded_n_lt_2p": short, "p": p})

    def pixels(proc_dir, meta):
        for stem in meta["stem"]:
            hs = read_patch(root / part / "HS" / f"{stem}.tif", skip_first, skip_last)
            yield valid_pixels(hs.reshape(-1, p))

    res = run_pipeline(NAME, params=params, process_fn=process,
                       covariance_fn=lambda pr, o: covariance_stage(pr, o, pixels, estimator),
                       cache_dir=cache_dir, keep=keep, force=force)
    out = load_covariance_dir(res.covariance_dir)
    meta = out.meta
    groups = np.arange(len(meta))
    if part == "train":
        labels = meta["label"].to_numpy()
        full = out.meta_all
        sp = group_kfold_split(full["label"].to_numpy(), np.arange(len(full)),
                               n_splits=n_splits, fold=fold, seed=seed)
        splits = map_splits({"stratified": sp, "recommended": sp}, meta["proc_index"],
                            len(full))
    else:
        labels, splits = np.full(len(meta), None, dtype=object), {}
    return Bunch(covariances=out.covariances, means=out.means, n_obs=out.n_obs,
                 labels=labels, groups=groups, meta=meta, info=out.info, splits=splits,
                 root=root, part=part, skip=(skip_first, skip_last),
                 covariance_path=res.covariance_dir)


@DatasetManager.register_dataset("kaggle_wheat")
class KaggleWheatDataset(HSIDataset):
    """Wheat disease HS patches (labelled train set), 107 x 107 covariances.

    Parameters
    ----------
    skip_first, skip_last, n_splits, fold
        See :func:`fetch_kaggle_wheat`.
    split, subset
        See :class:`~spdnet_datasets.hyperspectral._common.HSIDataset`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    def __init__(self, data_dir=None, *, skip_first=8, skip_last=10, n_splits=5, fold=0,
                 split="recommended", subset=None, cache_dir=None, keep=None, force=False,
                 verify="size", **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(data_dir=str(root), **kwargs)
        bunch = fetch_kaggle_wheat(root, cache_dir, keep, self.estimator, force, verify,
                                   skip_first=skip_first, skip_last=skip_last,
                                   n_splits=n_splits, fold=fold)
        self._setup(bunch, split, subset)

    def _load_raw(self, k):
        stem = self.meta_all["stem"].iloc[k]
        return read_patch(self.bunch.root / "train" / "HS" / f"{stem}.tif", *self.bunch.skip)


__all__ = ["KaggleWheatDataset", "fetch_kaggle_wheat", "read_patch"]
