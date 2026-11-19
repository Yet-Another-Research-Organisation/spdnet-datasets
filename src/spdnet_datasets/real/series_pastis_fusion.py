"""
Covariance series from PASTIS-R fusing Sentinel-2 and Sentinel-1 (user request 9/11, variant "B" of the R17 bilan):
one series per agricultural parcel, the observations of a covariance being the pixels of the parcel.

- Periods: ``n_periods`` equal intervals of the span common to all the patches and the three sensors (T = 12, about
  one month each). Per pixel and period: median over the acquisitions of the period (robust to cloudy S2 dates)
  of the 10 S2 bands and of (VV, VH) in dB of the ascending and descending S1 orbits (the third S1 channel,
  VV - VH, is their difference and is dropped): p = 14 variables.
- Each variable is divided by its standard deviation over fold 1 (NORM_*_patch.json of the dataset, a training
  fold), so that the Ledoit-Wolf shrinkage towards a multiple of the identity does not erase the S2 bands
  (reflectances in 1e4 units against dB).
- Token = Ledoit-Wolf covariance of the 14 variables over the parcel pixels (at least ``min_pixels`` = 30 > 2p).
  A period without any acquisition of one of the sensors is an identity token (masked, E8).
- ``pooled`` (baseline B3, no series): one covariance over the pixels x periods, (N, 1, 14, 14).
- Official folds: 1-3 train, 4 validation, 5 test, each capped by a stratified subsample drawn once (seed 0):
  ``caps`` = (train, val, test).

.. note::
   :class:`spdnet_datasets.series.CovarianceSeries` now builds these series on top of the main
   loaders (their samples, splits and groups) for the S1 + S2 fusion (``CovarianceSeries("pastis_r", "s1s2_fusion")``). See the guide page
   "Covariance series". This module is kept to reproduce earlier results.
"""

import ast
import datetime
import json
import os
from multiprocessing import Pool

import numpy as np

from .series import covariance, data_root
from .series_breizhcrops import _cap_stratified

SENSORS = ("S2", "S1A", "S1D")
CHANNELS = {"S2": list(range(10)), "S1A": [0, 1], "S1D": [0, 1]}


def _days(date: int) -> int:
    return datetime.date(date // 10000, date // 100 % 100, date % 100).toordinal()


def _patch_series(args: tuple) -> list[np.ndarray]:
    """One patch -> the covariance series of its kept parcels (in the order of ``parcels``)."""
    root, pid, dates, edges, scale, parcels, pooled = args
    ids = np.load(os.path.join(root, "ANNOTATIONS", f"ParcelIDs_{pid}.npy"))
    stacks, valid = [], np.ones(len(edges) - 1, bool)
    for sensor in SENSORS:
        cube = np.load(os.path.join(root, f"DATA_{sensor}", f"{sensor}_{pid}.npy"))[:, CHANNELS[sensor]]
        period = np.searchsorted(edges, dates[sensor], side="right") - 1
        frames = []
        for t in range(len(edges) - 1):
            chosen = np.flatnonzero(period == t)
            valid[t] &= len(chosen) > 0
            frames.append(np.median(cube[chosen].astype(np.float64), 0) if len(chosen) else
                          np.zeros(cube.shape[1:]))
        stacks.append(np.stack(frames) / scale[sensor][None, :, None, None])  # (T, c, 128, 128)
    cube = np.concatenate(stacks, 1)  # (T, 14, 128, 128)
    out = []
    for parcel in parcels:
        pixels = cube[:, :, ids == parcel].transpose(0, 2, 1)  # (T, n, 14)
        if pooled:
            out.append(covariance(pixels[valid].reshape(-1, pixels.shape[-1]))[None])
        else:
            series = np.tile(np.eye(pixels.shape[-1]), (len(pixels), 1, 1))
            for t in np.flatnonzero(valid):
                series[t] = covariance(pixels[t])
            out.append(series)
    return out


def pastis_fusion_series(root: str | None = None, n_periods: int = 12, min_pixels: int = 30,
                         caps: tuple[int, int, int] = (20000, 5000, 10000), pooled: bool = False,
                         jobs: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(N, 12, 14, 14) covariances (or (N, 1, 14, 14) pooled), labels 0..17, pixels per parcel, {"split"}."""
    root = data_root(root, "SITS", "PASTIS-R", "PASTIS-R")
    with open(os.path.join(root, "metadata.geojson")) as f:
        patches = [x["properties"] for x in json.load(f)["features"]]
    acquisitions = lambda v: v if isinstance(v, dict) else ast.literal_eval(v)  # noqa: E731
    dates = {p["ID_PATCH"]: {s: np.array([_days(d) for d in acquisitions(p[f"dates-{s}"]).values()])
                             for s in SENSORS} for p in patches}
    start = max(d[s].min() for d in dates.values() for s in SENSORS)
    end = min(d[s].max() for d in dates.values() for s in SENSORS)
    edges = np.linspace(start, end + 1, n_periods + 1)
    scale = {}
    for sensor in SENSORS:
        with open(os.path.join(root, f"NORM_{sensor}_patch.json")) as f:
            scale[sensor] = np.array(json.load(f)["Fold_1"]["std"])[CHANNELS[sensor]]

    candidates = []  # (patch, parcel, label, role, pixels)
    for patch in patches:
        pid = patch["ID_PATCH"]
        target = np.load(os.path.join(root, "ANNOTATIONS", f"TARGET_{pid}.npy"))[0].ravel()
        ids, inverse, counts = np.unique(np.load(os.path.join(root, "ANNOTATIONS", f"ParcelIDs_{pid}.npy")).ravel(),
                                         return_inverse=True, return_counts=True)
        for k, (parcel, count) in enumerate(zip(ids, counts, strict=True)):
            label = np.bincount(target[inverse == k]).argmax()  # majority semantic class of the parcel
            if parcel != 0 and count >= min_pixels and label not in (0, 19):
                fold = int(patch["Fold"])
                candidates.append((pid, parcel, label - 1, 0 if fold <= 3 else fold - 3, count))
    rng = np.random.default_rng(0)
    kept = []
    for role in range(3):
        rows = [c for c in candidates if c[3] == role]
        kept += [rows[i] for i in _cap_stratified(np.array([c[2] for c in rows]), caps[role], rng)]
    kept.sort(key=lambda c: (c[0], c[1]))  # one read of each patch

    by_patch = {}
    for c in kept:
        by_patch.setdefault(c[0], []).append(c)
    jobs_args = [(root, pid, dates[pid], edges, scale, [c[1] for c in rows], pooled) for pid, rows in by_patch.items()]
    with Pool(jobs) as pool:
        series = [s for out in pool.map(_patch_series, jobs_args) for s in out]
    rows = [c for pid in by_patch for c in by_patch[pid]]
    return (np.stack(series), np.array([c[2] for c in rows]), np.array([c[4] for c in rows]),
            {"split": np.array([c[3] for c in rows])})
