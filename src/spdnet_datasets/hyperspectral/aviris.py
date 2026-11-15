"""AVIRIS-C CA TS14: AVIRIS-Classic windows over California (LISTIC, local dataset).

Two versions built at LISTIC (13/11/2026) from the ORNL DAAC L2 reflectance
products, labels from the USDA NASS Cropland Data Layer:

- ``version="fine"`` (``AVIRIS-C_CA_TS14_FINE``): 1 590 windows of 600 m
  (32 x 32 pixels), 12 fine land-cover classes (purity >= 70 %),
  classification;
- ``version="mix"`` (``AVIRIS-C_CA_TS14_MIX``): 2 034 windows, target = the
  proportions of 13 classes (12 + ``autre``), regression
  (``task="regression"``); ``task="fine"`` uses the dominant class.

Each window is a time series of 14 dates ``(T, 224, 32, 32)`` float16
(reflectance 0-1, NaN = no data); ``meta/date_mask.npz`` flags the dates that
passed the quality control.

Covariance ("static"): the 166 clean bands (``meta/bands.csv``), one in
``band_step`` (default 2): p = 83. The observations are the finite, non-zero
pixels of **all the QC-valid dates pooled** (reflectance clipped to
[0, 1.5]): n = 1 024 x (number of valid dates) >= 2p = 166. Windows without
any valid date, or with fewer than 2p valid pixels, are discarded and
counted. Pooling mixes acquisition dates (seasons, years): the covariance
describes the spectral variability of the window over time, not one date.
``return_raw`` gives the clipped pixels of the valid dates, (T_ok, 1024, p),
NaN kept.

Splits: ``"fold5"`` (atlas folds by 10 km blocks, column ``fold5``; test =
fold ``fold``, val = fold ``fold + 1``) = ``"recommended"``, and
``"block10km"`` (StratifiedGroupKFold on the 10 km blocks). No buffer: 3-5 %
of the test windows are less than 1 km from a training window (fiche).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import dataset_path
from ..core.estimators import check_sample_size
from ..core.fetch import Bunch, run_pipeline, verify_local
from ..manager import DatasetManager
from ._common import (
    HSIDataset,
    covariance_stage,
    group_kfold_split,
    load_covariance_dir,
    valid_pixels,
    write_json,
)

DOMAIN = "HYPERSPECTRAL"
NAMES = {"fine": "AVIRIS-C_CA_TS14_FINE", "mix": "AVIRIS-C_CA_TS14_MIX"}
FILES = {}  # built at LISTIC, not published
PROPORTIONS = ["amandiers", "arbustes", "eau", "foret_coniferes", "foret_mixte",
               "prairie_paturage", "sol_nu", "urbain_dense", "urbain_faible",
               "urbain_moyen", "urbain_ouvert", "vignes", "autre"]
CLIP = (0.0, 1.5)


def _bands(root, band_step):
    clean = pd.read_csv(root / "meta" / "bands.csv")["clean"].to_numpy() == 1
    return np.flatnonzero(clean)[::band_step]


def _window_file(root, row, version):
    name = row["cls"] if version == "fine" else row["dominant"]
    return root / "windows" / f"{int(row['window_id'])}_{name}.npz"


def window_dates(path, ok, bands):
    """Clipped pixels of the valid dates of one window: (T_ok, 1024, p) float32."""
    cube = np.load(path)["cube"][np.flatnonzero(ok)][:, bands]  # (T_ok, p, 32, 32)
    cube = np.moveaxis(cube.astype(np.float32), 1, -1).reshape(len(cube), -1, len(bands))
    with np.errstate(invalid="ignore"):
        return np.clip(cube, *CLIP)


def fetch_aviris(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                 verify="size", *, version="fine", band_step=2, fold=0, seed=0):
    """Static covariances of AVIRIS-C CA TS14 FINE or MIX (one per window).

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means``, ``labels``
    (class, or dominant class for MIX), ``targets`` (MIX: (B, 13)
    proportions, else None), ``target_names``, ``groups`` (``block10km``),
    ``meta``, ``info`` and ``splits`` (``"fold5"`` = ``"recommended"``,
    ``"block10km"``).
    """
    if version not in NAMES:
        raise ValueError(f"version must be one of {sorted(NAMES)}")
    name = NAMES[version]
    root = dataset_path(DOMAIN, name, data_dir)
    verify_local(root, verify=verify,
                 required=("meta/labels.csv", "meta/date_mask.npz", "meta/bands.csv"))
    bands = _bands(root, band_step)
    check_sample_size(32 * 32, len(bands))
    params = {"band_step": band_step, "estimator": estimator, "clip": CLIP, "version": 1}

    def process(_, out_dir):
        labels = pd.read_csv(root / "meta" / "labels.csv")
        mask = np.load(root / "meta" / "date_mask.npz")
        ok = dict(zip(mask["window_id"].tolist(), mask["ok"], strict=True))
        rows, no_date = [], 0
        for _, row in labels.iterrows():
            valid = ok[int(row["window_id"])]
            if not valid.any():
                no_date += 1
                continue
            rec = row.to_dict()
            rec.update(file=_window_file(root, row, version).name,
                       n_ok_dates_mask=int(valid.sum()),
                       ok_dates="".join("1" if v else "0" for v in valid))
            rows.append(rec)
        meta = pd.DataFrame(rows)
        meta["label"] = meta["cls"] if version == "fine" else meta["dominant"]
        meta.to_csv(Path(out_dir) / "meta.csv", index=False)
        write_json(Path(out_dir) / "info.json", {
            "n_windows": len(labels), "excluded_no_valid_date": no_date,
            "p": len(bands), "pooling": "all QC-valid dates"})

    def pixels(proc_dir, meta):
        for _, row in meta.iterrows():
            ok = np.array([c == "1" for c in str(row["ok_dates"]).zfill(14)])
            X = window_dates(root / "windows" / row["file"], ok, bands)
            yield valid_pixels(X.reshape(-1, len(bands)))

    res = run_pipeline(name, params=params, process_fn=process,
                       covariance_fn=lambda p, o: covariance_stage(p, o, pixels, estimator),
                       cache_dir=cache_dir, keep=keep, force=force)
    out = load_covariance_dir(res.covariance_dir)
    meta = out.meta
    labels = meta["label"].to_numpy()
    folds = meta["fold5"].to_numpy()
    fold5 = {"train": np.flatnonzero((folds != fold) & (folds != (fold + 1) % 5)),
             "val": np.flatnonzero(folds == (fold + 1) % 5),
             "test": np.flatnonzero(folds == fold)}
    blocks = meta["block10km"].to_numpy()
    targets = meta[[f"p_{c}" for c in PROPORTIONS]].to_numpy(np.float64) \
        if version == "mix" else None
    return Bunch(
        covariances=out.covariances, means=out.means, n_obs=out.n_obs, labels=labels,
        targets=targets, target_names=PROPORTIONS if version == "mix" else None,
        groups=blocks, meta=meta, info=out.info, bands=bands, root=root,
        splits={"fold5": fold5, "recommended": fold5,
                "block10km": group_kfold_split(labels, blocks, fold=fold, seed=seed)},
        covariance_path=res.covariance_dir,
    )


@DatasetManager.register_dataset("aviris")
class AVIRISDataset(HSIDataset):
    """AVIRIS-C CA TS14 static covariances (83 x 83), FINE or MIX.

    Parameters
    ----------
    version : {"fine", "mix"}
        FINE: 12-class classification. MIX: ``task="regression"`` on the 13
        proportions, or ``task="fine"`` on the dominant class.
    band_step, fold
        See :func:`fetch_aviris`.
    split, subset
        See :class:`~spdnet_datasets.hyperspectral._common.HSIDataset`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    def __init__(self, data_dir=None, *, version="fine", band_step=2, fold=0,
                 split="recommended", subset=None, cache_dir=None, keep=None, force=False,
                 verify="size", **kwargs):
        if version not in NAMES:
            raise ValueError(f"version must be one of {sorted(NAMES)}")
        if version == "fine" and kwargs.get("task") == "regression":
            raise ValueError("version='fine' has no regression target; use version='mix'")
        root = dataset_path(DOMAIN, NAMES[version], data_dir)
        super().__init__(data_dir=str(root), **kwargs)
        self.version = version
        bunch = fetch_aviris(root, cache_dir, keep, self.estimator, force, verify,
                             version=version, band_step=band_step, fold=fold)
        self._setup(bunch, split, subset)

    def _load_raw(self, k):
        row = self.meta_all.iloc[k]
        ok = np.array([c == "1" for c in str(row["ok_dates"]).zfill(14)])
        return window_dates(self.bunch.root / "windows" / row["file"], ok, self.bunch.bands)


__all__ = ["AVIRISDataset", "fetch_aviris"]
