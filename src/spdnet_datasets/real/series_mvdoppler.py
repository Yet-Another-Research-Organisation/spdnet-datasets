"""
Covariance series from MVDoppler (atlas v2 sheet "mvdoppler"): mmWave Doppler snapshots of 1.28 s (128 Doppler bins x
256 frames, two radar views) of people walking with 4 patterns (normal, phone call, hands in pockets, texting).

- Token (atlas series "S1_fenetres64"): view x, log amplitude, the 64 central Doppler bins averaged by 4 (p = 16),
  Ledoit-Wolf covariance over windows of 64 frames, step 32 (T = 7 time windows, n/p = 4).
- ``pooled`` (baseline B3, no series): one covariance over the 256 frames, (N, 1, 16, 16).
- Official subject-disjoint folds (design_table.csv): folds 1 + 2 train, fold 0 validation, fold 3 test, each
  capped by a stratified subsample drawn once (seed 0): ``caps`` = (train, val, test). Snapshots of one recording
  overlap by 50 % but never cross folds. Files without the HDF5 signature are skipped (two zero-filled files in the
  BERTHA copy, 9/11: phone_call_02515, texting_12478).
"""

from multiprocessing import Pool

import numpy as np
import pandas as pd

from .series import covariance, data_root
from .series_breizhcrops import _cap_stratified

DOMAIN, NAME = "RADAR_DOPPLER", "MVDoppler"
MVDOPPLER_CLASSES = ["normal", "phone_call", "pockets", "texting"]
FOLDS = {1: 0, 2: 0, 0: 1, 3: 2}  # official fold -> train / val / test


def _readable(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n"


def _snapshot_series(args: tuple) -> np.ndarray:
    import h5py

    path, pooled = args
    with h5py.File(path, "r") as f:
        x = np.log(np.clip(f["radar_dat"][32:96, :, 0], 1e-12, None)).astype(np.float64)  # (64 bins, 256 frames)
    bands = x.reshape(16, 4, -1).mean(1).T  # (256 frames, 16)
    if pooled:
        return covariance(bands)[None]
    return np.stack([covariance(bands[s:s + 64]) for s in range(0, 256 - 64 + 1, 32)])


def mvdoppler_series(root: str | None = None, caps: tuple[int, int, int] = (20000, 5000, 10000), pooled: bool = False,
                     jobs: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(N, 7, 16, 16) covariances (or (N, 1, 16, 16) pooled), patterns 0..3, frames per window, {"split"}."""
    root = data_root(root, DOMAIN, NAME)
    design = pd.read_csv(f"{root}/Labels_and_metadata/design_table.csv")
    design = design[[_readable(f"{root}/Data/{name}.h5") for name in design.fname]]
    rng = np.random.default_rng(0)
    paths, labels, split = [], [], []
    for role in range(3):
        rows = design[design.fold.map(FOLDS) == role].reset_index(drop=True)
        y = rows.pattern.map(MVDOPPLER_CLASSES.index).to_numpy()
        keep = _cap_stratified(y, caps[role], rng)
        paths += [f"{root}/Data/{name}.h5" for name in rows.fname[keep]]
        labels.append(y[keep])
        split.append(np.full(len(keep), role))
    with Pool(jobs) as pool:
        data = np.stack(pool.map(_snapshot_series, [(p, pooled) for p in paths], chunksize=64))
    return data, np.concatenate(labels), np.full(len(paths), 256 if pooled else 64), {"split": np.concatenate(split)}
