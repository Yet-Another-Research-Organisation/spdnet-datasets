"""
Covariance series from TreeSatAI-Time-Series (data atlas, chapter 21): one series per 60 m patch, one token per
calendar month (T = 12, position = month). A token is the 10 x 10 Ledoit-Wolf covariance of the Sentinel-2
bands B02-B12 over the clear pixels (snow and cloud probability both 0, 10 bands finite and positive) of all the
acquisitions of that month (36 pixels per date), diagonally loaded by 1e-6 tr(C) / p. A month with fewer than
20 = 2p clear pixels is missing and encoded as the identity. Reflectances are kept raw (L2A digital numbers,
~1e2-1e4): the scale does not matter after recentring.

Labels: the json gives genus proportions per patch; a patch is kept when its dominant genus covers >= 0.7 of it
(only 18 948 patches have a single genus, and the json drops small proportions anyway). The ``n_classes`` most
frequent dominant genera are kept, "Cleared" excluded (not a tree genus). Split: the official lists (split/);
up to ``max_per_class`` patches per class are drawn from official train + val and, separately, from official
test; 20 % of the first set, stratified by class, becomes the validation set.
"""

import json
import os

import numpy as np

from ..estimator.covariance import EstimateCovariance

N_BANDS = 10
MIN_PIXELS = 2 * N_BANDS


def treesat_patch(path: str, estimator: EstimateCovariance, pooled: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One .h5 patch -> (12, 10, 10) monthly covariances (identity when < 20 clear pixels), clear pixels per month.
    ``pooled``: (1, 10, 10), one covariance over the clear pixels of the valid months (baseline B3, no series)."""
    import h5py

    with h5py.File(path, "r") as h5:
        data = h5["sen-2-data"][:].astype(np.float64)  # (T, 10, 6, 6)
        masks = h5["sen-2-masks"][:]  # (T, 2, 6, 6): snow, cloud probabilities
        products = h5["sen-2-products"][:]
    months = np.array([int(p.decode().split("_")[2][4:6]) - 1 for p in products])  # S2A_MSIL2A_YYYYMMDDT..
    pixels = data.reshape(len(data), N_BANDS, -1).transpose(0, 2, 1)  # (T, 36, 10)
    clear = (masks.reshape(len(masks), 2, -1) == 0).all(1)
    clear &= (np.isfinite(pixels) & (pixels > 0)).all(-1)
    covariances = np.broadcast_to(np.eye(N_BANDS), (12, N_BANDS, N_BANDS)).copy()
    counts = np.zeros(12, int)
    for m in range(12):
        x = pixels[months == m][clear[months == m]]
        counts[m] = len(x)
        if counts[m] >= MIN_PIXELS:
            c = estimator(x)
            covariances[m] = c + 1e-6 * np.trace(c) / N_BANDS * np.eye(N_BANDS)
    if pooled:
        x = np.concatenate([pixels[months == m][clear[months == m]] for m in range(12) if counts[m] >= MIN_PIXELS]
                           or [pixels[:0].reshape(0, N_BANDS)])
        c = estimator(x) if len(x) >= MIN_PIXELS else np.eye(N_BANDS)
        covariances = (c + 1e-6 * np.trace(c) / N_BANDS * np.eye(N_BANDS))[None]
    return covariances, counts


def treesat_series(
    root: str = "/DATA/TreeSatAI-TS",
    max_per_class: int = 400,
    n_classes: int = 6,
    seed: int = 0,
    min_months: int = 6,
    min_proportion: float = 0.7,
    pooled: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(n_patches, 12, 10, 10) monthly covariances, labels in 0..n_classes-1 (most frequent first), median clear
    pixels per valid month, {"split": 0 train / 1 validation / 2 test}."""
    with open(os.path.join(root, "TreeSatBA_v9_60m_multi_labels.json")) as f:
        proportions = json.load(f)
    official = {}
    for part in ("train", "val", "test"):
        with open(os.path.join(root, "split", f"{part}_filenames.lst")) as f:
            official.update({line.strip(): part for line in f if line.strip()})
    files = {}  # official name (.tif) -> .h5 path (the h5 name adds the year of the series)
    for name in os.listdir(os.path.join(root, "sentinel-ts")):
        files[name.rsplit("_", 1)[0] + ".tif"] = os.path.join(root, "sentinel-ts", name)

    dominant = {}
    for name, genera in proportions.items():
        genus, share = max(genera, key=lambda g: g[1])
        if share >= min_proportion and genus != "Cleared" and name in files and name in official:
            dominant[name] = genus
    genera, counts = np.unique(list(dominant.values()), return_counts=True)
    classes = list(genera[np.argsort(-counts, kind="stable")][:n_classes])

    estimator = EstimateCovariance("ledoit_wolf", remove_mean=True, exclude_zero=False)
    rng = np.random.default_rng(seed)
    data, labels, n_obs, split = [], [], [], []
    for test in (False, True):
        for label, genus in enumerate(classes):
            names = sorted(n for n, g in dominant.items() if g == genus and (official[n] == "test") == test)
            kept = []
            for i in rng.permutation(len(names)):  # draw until max_per_class series with enough valid months
                covariances, pixels = treesat_patch(files[names[i]], estimator, pooled)
                valid = pixels >= MIN_PIXELS
                if valid.sum() >= min_months:
                    kept.append((covariances, np.median(pixels[valid])))
                    if len(kept) == max_per_class:
                        break
            parts = np.full(len(kept), 2)
            if not test:  # 20 % of the official train + val draw becomes the validation set, per class
                parts[:] = 0
                parts[rng.permutation(len(kept))[:round(0.2 * len(kept))]] = 1
            data += [c for c, _ in kept]
            n_obs += [n for _, n in kept]
            labels += [label] * len(kept)
            split += list(parts)
    return np.stack(data), np.array(labels), np.array(n_obs), {"split": np.array(split)}
