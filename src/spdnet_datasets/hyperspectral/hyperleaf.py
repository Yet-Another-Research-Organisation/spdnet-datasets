"""HyperLeaf2024: hyperspectral images of barley leaves (Kaggle competition).

``images/<ImageId>.tiff``: 48 x 352 pixels x 204 bands (uint16, stored
(B, H, W), background = 0); ``train.csv``: ``ImageId``, ``GrainWeight``,
``Gsw``, ``PhiPS2``, ``Fertilizer`` and one-hot cultivars (``Heerup``,
``Kvium``, ``Rembrandt``, ``Sheriff``). Only the 1 590 training images are
labelled (the 820 test images are the competition submission): every split
is cut from the training set.

Covariance: the non-zero (leaf) pixels of the image, one band in
``band_step`` (default 1: p = 204; leaves have 1 070-7 044 pixels, so
n >= 2p). Images with fewer than 2p leaf pixels are discarded and counted.

Labels: ``target="cultivar"`` (4 classes, default), ``"fertilizer"`` (3
levels) or ``"combined"`` (12); ``task="regression"`` returns the continuous
targets ``regression_targets`` (default ``("GrainWeight", "Gsw", "PhiPS2")``).

Split (atlas): StratifiedGroupKFold(5) by reconstructed plot, group =
``GrainWeight | cultivar | Fertilizer`` (24 groups; all the images of a plot
share its grain weight). A random split is much more optimistic (fiche:
TS-LR cultivar 0.915 random vs 0.669 grouped).

``mode="image"`` keeps the former image mode: ``x`` is the image restricted to
``bands`` (default approximate R, G, B), standardised per channel, float32
(C, 48, 352), for CNN backbones; no covariance is computed.

Download (manual): Kaggle competition
https://www.kaggle.com/competitions/hyperleaf2024 (Kaggle account and acceptance of
the rules required). Download ``images/``, ``train.csv`` and ``test.csv`` and place
it under ``data_dir`` (default ``<data root>/HYPERSPECTRAL/HyperLeaf2024``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core.config import dataset_path
from ..core.fetch import Bunch, run_pipeline, verify_local
from ..manager import DatasetManager
from ._common import (
    HSIDataset,
    band_index,
    covariance_stage,
    group_kfold_split,
    load_covariance_dir,
    map_splits,
    valid_pixels,
    write_json,
)

DOMAIN, NAME = "HYPERSPECTRAL", "HyperLeaf2024"
FILES = {}  # Kaggle competition data (account required): manual download
CULTIVARS = ("Heerup", "Kvium", "Rembrandt", "Sheriff")
TARGETS = ("cultivar", "fertilizer", "combined")
REGRESSION_TARGETS = ("GrainWeight", "Gsw", "PhiPS2")
N_BANDS = 204


def read_leaf(path, bands=None):
    """Image (H, W, B) float32 (``bands`` read only, in the given order)."""
    import tifffile

    a = tifffile.imread(path, key=None if bands is None else list(bands)).astype(np.float32)
    return np.moveaxis(a.reshape(-1, *a.shape[-2:]), 0, -1)


def train_table(root):
    """``train.csv`` with ``cultivar``, ``fertilizer``, ``combined`` and ``plot`` columns."""
    return _label_columns(pd.read_csv(Path(root) / "train.csv", dtype={"ImageId": str}))


def _label_columns(df):
    df = df.copy()
    df["ImageId"] = df["ImageId"].map(lambda v: f"{int(v):05d}")
    df["cultivar"] = np.asarray(CULTIVARS)[df[list(CULTIVARS)].to_numpy().argmax(1)]
    df["fertilizer"] = df["Fertilizer"].map(lambda v: f"{float(v):.1f}")
    df["combined"] = df["fertilizer"] + "_" + df["cultivar"]
    if "GrainWeight" in df:
        df["plot"] = df["GrainWeight"].astype(str) + "|" + df["cultivar"] + "|" + df["fertilizer"]
    else:
        df["plot"] = df["ImageId"]
    return df


def _splits(df, n_splits, fold, seed):
    sp = group_kfold_split(df["cultivar"].to_numpy(), df["plot"].to_numpy(),
                           n_splits=n_splits, fold=fold, seed=seed)
    return {"by_plot": sp, "recommended": sp}


def fetch_hyperleaf(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                    verify="size", *, band_step=1, n_splits=5, fold=0, seed=0):
    """Leaf covariances of the HyperLeaf2024 training images.

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means``, ``meta`` (the
    ``train.csv`` columns plus ``cultivar``, ``fertilizer``, ``combined``,
    ``plot``, ``n_valid``), ``groups`` (plot), ``info`` and ``splits``.
    """
    root = dataset_path(DOMAIN, NAME, data_dir)
    verify_local(root, verify=verify, required=("train.csv", "images"))
    bands = band_index(N_BANDS, band_step)
    params = {"band_step": band_step, "estimator": estimator, "version": 2}

    def process(_, out_dir):
        # leaves with fewer than 2p pixels are dropped (and counted) at estimation
        df = train_table(root)
        df.to_csv(Path(out_dir) / "meta.csv", index=False)
        write_json(Path(out_dir) / "info.json", {"n_images": len(df), "p": len(bands)})

    def pixels(proc_dir, meta):
        for image_id in meta["ImageId"]:
            leaf = read_leaf(root / "images" / f"{int(image_id):05d}.tiff", bands)
            yield valid_pixels(leaf.reshape(-1, len(bands)))

    res = run_pipeline(NAME, params=params, process_fn=process,
                       covariance_fn=lambda p, o: covariance_stage(p, o, pixels, estimator),
                       cache_dir=cache_dir, keep=keep, force=force)
    out = load_covariance_dir(res.covariance_dir)
    meta = out.meta
    meta = _label_columns(meta)
    meta["n_valid"] = out.n_obs
    return Bunch(covariances=out.covariances, means=out.means, n_obs=out.n_obs, meta=meta,
                 groups=meta["plot"].to_numpy(), info=out.info, bands=bands, root=root,
                 labels=meta["cultivar"].to_numpy(),
                 targets=meta[list(REGRESSION_TARGETS)].to_numpy(np.float64),
                 target_names=list(REGRESSION_TARGETS),
                 splits=map_splits(_splits(_label_columns(out.meta_all), n_splits, fold, seed),
                                   meta["proc_index"], len(out.meta_all)),
                 covariance_path=res.covariance_dir)


@DatasetManager.register_dataset("hyperleaf")
class HyperLeafDataset(HSIDataset):
    """HyperLeaf2024 leaves: 204 x 204 covariances (or band images), grouped split.

    Parameters
    ----------
    target : {"cultivar", "fertilizer", "combined"}
        Classification label (``task="fine"``).
    regression_targets : tuple of str
        Columns returned with ``task="regression"``.
    mode : {"cov", "image"}
        ``"image"``: ``x`` is the standardised band image (C, 48, 352).
    bands : list of int, optional
        ``mode="image"`` only: bands to read (default ``RGB_BANDS``).
    band_step, n_splits, fold
        See :func:`fetch_hyperleaf`.
    split, subset
        See :class:`~spdnet_datasets.hyperspectral._common.HSIDataset`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    CULTIVAR_NAMES = list(CULTIVARS)
    N_BANDS = N_BANDS
    # Approximate R, G, B bands (~640, 550, 460 nm), assuming 204 regularly
    # spaced bands over 397-1004 nm (Specim IQ); the TIFFs carry no wavelengths.
    RGB_BANDS = (81, 51, 21)

    def __init__(self, data_dir=None, *, target="cultivar",
                 regression_targets=REGRESSION_TARGETS, mode="cov", bands=None, band_step=1,
                 n_splits=5, fold=0, split="recommended", subset=None, cache_dir=None,
                 keep=None, force=False, verify="size", **kwargs):
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        if mode not in ("cov", "image"):
            raise ValueError(f"mode must be 'cov' or 'image', got {mode!r}")
        self.mode = mode
        self.bands = list(bands if bands is not None else self.RGB_BANDS) \
            if mode == "image" else None
        if self.bands is not None and not all(0 <= b < N_BANDS for b in self.bands):
            raise ValueError(f"bands must be in [0, {N_BANDS}), got {self.bands}")
        self.target_size = (N_BANDS, N_BANDS) if mode == "cov" else (len(self.bands), 48, 352)
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(data_dir=str(root), **kwargs)
        if mode == "image":
            if self.return_mean or self.return_raw:
                raise ValueError("mode='image' returns the image only")
            meta = train_table(root)
            bunch = Bunch(covariances=None, means=None, meta=meta, root=root,
                          groups=meta["plot"].to_numpy(), info={},
                          splits=_splits(meta, n_splits, fold, 0))
        else:
            bunch = fetch_hyperleaf(root, cache_dir, keep, self.estimator, force, verify,
                                    band_step=band_step, n_splits=n_splits, fold=fold)
        bunch.labels = bunch.meta[target].to_numpy()
        bunch.targets = bunch.meta[list(regression_targets)].to_numpy(np.float64) \
            if set(regression_targets) <= set(bunch.meta.columns) else None
        bunch.target_names = list(regression_targets)
        self._setup(bunch, split, subset)

    def _load_raw(self, k):
        image_id = self.meta_all["ImageId"].iloc[k]
        return read_leaf(self.bunch.root / "images" / f"{image_id}.tiff", self.bunch.bands)

    def __getitem__(self, i):
        if self.mode == "cov":
            return super().__getitem__(i)
        image_id = self.meta_all["ImageId"].iloc[int(self._index[i])]
        a = np.moveaxis(read_leaf(self.bunch.root / "images" / f"{image_id}.tiff", self.bands),
                        -1, 0)
        a = (a - a.mean(axis=(1, 2), keepdims=True)) / (a.std(axis=(1, 2), keepdims=True) + 1e-6)
        x = torch.from_numpy(a)
        if self.transform is not None:
            x = self.transform(x)
        y = self.labels[i]
        if self.task == "regression":
            return x, torch.tensor(np.array(y, dtype=np.float64))
        return x, int(y)


__all__ = ["HyperLeafDataset", "fetch_hyperleaf", "read_leaf", "train_table"]
