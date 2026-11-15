"""Hyperspectral placenta dataset (Puustinen et al., Data in Brief, 2023).

101 images 1024 x 1024 of four human placentas (surgical microscope, Senop
HSC-2), flat-field corrected reflectance, manual annotations. The local
derived version ``imagettes/`` holds 77 203 non-overlapping 24 x 24 windows
x 37 bands (515-697 nm), purity >= 0.9, from the 81 images P007-P087
(``placenta_windows_metadata.json``: ``sample_id`` = source image,
``dye_type`` red_blue / ICG, ``class``, ``original_class``, ``position``).
Five classes: Artery, Stroma, Suture, Umbilical, Vein.

Covariance: the 576 pixels of the window, one band in ``band_step``
(default 1: p = 37, n / 2p = 7.8, so no band subsampling is needed; QUESTIONS
A2). Windows with fewer than 2p valid pixels would be discarded and counted.

Splits:

- ``"by_image"`` = ``"recommended"``: StratifiedGroupKFold(5) by
  ``sample_id`` (test = fold ``fold``, val = fold ``fold + 1``). The 81 images
  come from only four placentas and no placenta identifier exists: a
  placenta is almost surely on both sides (leakage not measurable);
- ``"dye_transfer"``: train red_blue images, test ICG images (the ICG images
  were taken after the red/blue injection on the same placentas).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import dataset_path
from ..core.estimators import check_sample_size
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

DOMAIN, NAME = "HYPERSPECTRAL", "Hyperspectral_Placenta_Dataset"
FILES = {}  # Data in Brief 2023 (Zenodo / Mendeley): manual download
WINDOWS = "imagettes/placenta_windows_data.npy"
METADATA = "imagettes/placenta_windows_metadata.json"
N_BANDS = 37


def fetch_placenta(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                   verify="size", *, band_step=1, n_splits=5, fold=0, seed=0):
    """Window covariances of the placenta dataset.

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means``, ``labels``,
    ``groups`` (``sample_id``), ``meta``, ``info`` and ``splits``
    (``"by_image"`` = ``"recommended"``, ``"dye_transfer"``).
    """
    root = dataset_path(DOMAIN, NAME, data_dir)
    verify_local(root, verify=verify, required=(WINDOWS, METADATA))
    windows = np.load(root / WINDOWS, mmap_mode="r")
    if windows.shape[-1] != N_BANDS:
        raise ValueError(f"Expected {N_BANDS} bands, got {windows.shape}")
    bands = band_index(N_BANDS, band_step)
    check_sample_size(windows.shape[1] * windows.shape[2], len(bands))
    params = {"band_step": band_step, "estimator": estimator, "version": 2}

    def process(_, out_dir):
        records = json.loads((root / METADATA).read_text())
        if len(records) != len(windows):
            raise ValueError("metadata and windows have different lengths")
        meta = pd.DataFrame({
            "window_index": np.arange(len(records)),
            "sample_id": [r["sample_id"] for r in records],
            "dye_type": [r["dye_type"] for r in records],
            "label": [r["class"] for r in records],
            "original_class": [r["original_class"] for r in records],
            "row": [r["position"][0] for r in records],
            "col": [r["position"][1] for r in records],
        })
        meta.to_csv(Path(out_dir) / "meta.csv", index=False)
        write_json(Path(out_dir) / "info.json", {"n_windows": len(meta), "p": len(bands)})

    def pixels(proc_dir, meta):
        for i in meta["window_index"]:
            w = np.asarray(windows[int(i)][..., bands])
            yield valid_pixels(w.reshape(-1, len(bands)))

    res = run_pipeline(NAME, params=params, process_fn=process,
                       covariance_fn=lambda p, o: covariance_stage(p, o, pixels, estimator),
                       cache_dir=cache_dir, keep=keep, force=force)
    out = load_covariance_dir(res.covariance_dir)
    meta = out.meta
    labels = meta["label"].to_numpy()
    groups = meta["sample_id"].to_numpy()
    full = out.meta_all
    by_image = map_splits(
        {"s": group_kfold_split(full["label"].to_numpy(), full["sample_id"].to_numpy(),
                                n_splits=n_splits, fold=fold, seed=seed)},
        meta["proc_index"], len(full))["s"]
    dye = meta["dye_type"].to_numpy()
    return Bunch(
        covariances=out.covariances, means=out.means, n_obs=out.n_obs, labels=labels,
        groups=groups, meta=meta, info=out.info, bands=bands, windows_path=root / WINDOWS,
        splits={"by_image": by_image, "recommended": by_image,
                "dye_transfer": {"train": np.flatnonzero(dye == "red_blue"),
                                 "val": np.zeros(0, int),
                                 "test": np.flatnonzero(dye == "ICG")}},
        covariance_path=res.covariance_dir,
    )


@DatasetManager.register_dataset("placenta")
class PlacentaDataset(HSIDataset):
    """Placenta 24 x 24 windows, 37 x 37 covariances, split grouped by source image.

    Parameters
    ----------
    band_step, n_splits, fold
        See :func:`fetch_placenta`.
    split : str
        ``"recommended"`` (by image) or ``"dye_transfer"``.
    subset
        See :class:`~spdnet_datasets.hyperspectral._common.HSIDataset`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    def __init__(self, data_dir=None, *, band_step=1, n_splits=5, fold=0,
                 split="recommended", subset=None, cache_dir=None, keep=None, force=False,
                 verify="size", **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(data_dir=str(root), **kwargs)
        bunch = fetch_placenta(root, cache_dir, keep, self.estimator, force, verify,
                               band_step=band_step, n_splits=n_splits, fold=fold)
        self._setup(bunch, split, subset)
        self._windows = np.load(bunch.windows_path, mmap_mode="r")

    def _load_raw(self, k):
        # numpy window (the former loader passed a torch tensor to the numpy estimator)
        i = int(self.meta_all["window_index"].iloc[k])
        return self._windows[i][..., self.bunch.bands]


__all__ = ["PlacentaDataset", "fetch_placenta"]
