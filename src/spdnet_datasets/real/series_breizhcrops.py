"""
Covariance series from BreizhCrops (atlas v2 sheet "breizhcrops"; Russwurm et al. 2020): Sentinel-2 L2A time series
of mean parcel reflectances, Brittany 2017, 9 crop classes (class mapping of the BreizhCrops paper).

- Token = Ledoit-Wolf covariance of the 4 bands B2, B3, B4, B8 (visible + near infrared) over the cloud-free dates
  (CLD = 0) of one calendar quarter: T = 4 positions identified by the season. A quarter with fewer than
  ``min_dates`` clear dates is an identity token (masked, E8); parcels without any valid quarter are dropped.
- ``pooled`` (baseline B3, no series): one covariance over all the clear dates of the year, (N, 1, 4, 4).
- Official regional split (paper): frh01 + frh02 train, frh03 validation, frh04 test (distinct departments, a real
  spatial shift), each capped by a stratified subsample drawn once (seed 0): ``caps`` = (train, val, test).
"""

import numpy as np
import pandas as pd

ROOT = "/DATA/BreizhCrops/L2A_h5"
BREIZHCROPS_CLASSES = ["barley", "wheat", "rapeseed", "corn", "sunflower", "orchards", "nuts", "permanent meadows",
                       "temporary meadows"]
CODES = {"ORH": 0, "ORP": 0, "BTH": 1, "BTP": 1, "CZH": 2, "CZP": 2, "MID": 3, "MIE": 3, "MIS": 3, "TRN": 4,
         "AGR": 5, "PFR": 5, "PWT": 5, "VRG": 5, "CAB": 6, "CTG": 6, "NOS": 6, "NOX": 6, "PIS": 6,
         "PPH": 7, "PRL": 7, "PTR": 8, "RGA": 8}  # breizhcrops_classmapping.csv of the paper
BANDS = [0, 1, 2, 6]  # B2, B3, B4, B8 among B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
REGIONS = {"frh01": 0, "frh02": 0, "frh03": 1, "frh04": 2}


def _cap_stratified(labels: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted indices of a stratified subsample of size <= cap (proportional quotas, at least one per class)."""
    if len(labels) <= cap:
        return np.arange(len(labels))
    classes, counts = np.unique(labels, return_counts=True)
    quotas = np.maximum(1, np.floor(cap * counts / len(labels))).astype(int)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == c), q, replace=False)
                                   for c, q in zip(classes, quotas)]))


def _ledoit_wolf(x: np.ndarray, load: float = 1e-6) -> np.ndarray:
    from sklearn.covariance import ledoit_wolf

    covariance = ledoit_wolf(x)[0]
    return covariance + load * np.trace(covariance) / len(covariance) * np.eye(len(covariance))


def breizhcrops_series(root: str = ROOT, caps: tuple[int, int, int] = (20000, 5000, 10000), min_dates: int = 3,
                       pooled: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(N, 4, 4, 4) quarterly covariances (or (N, 1, 4, 4) pooled), labels 0..8, clear dates per token (median over
    the valid quarters; all of them when pooled), {"split": 0 train / 1 val / 2 test}."""
    import h5py

    rng = np.random.default_rng(0)
    data, labels, n_obs, split = [], [], [], []
    for role in range(3):
        index = pd.concat([pd.read_csv(f"{root}/{region}.csv").assign(region=region)
                           for region, r in REGIONS.items() if r == role], ignore_index=True)
        index = index[index.CODE_CULTU.isin(list(CODES))].reset_index(drop=True)
        y = index.CODE_CULTU.map(CODES).to_numpy()
        files = {region: h5py.File(f"{root}/{region}.h5", "r") for region in index.region.unique()}
        for i in _cap_stratified(y, caps[role], rng):
            row = index.iloc[i]
            a = files[row.region][row.path.lstrip("/")][:]
            clear = a[:, 11] == 0
            x = a[clear][:, 1:11][:, BANDS] / 1e4
            quarter = (pd.to_datetime(a[clear, 0].astype("int64")).month.to_numpy() - 1) // 3
            valid = [q for q in range(4) if (quarter == q).sum() >= min_dates]
            if not valid:  # no valid token: dropped from both versions (same samples with and without series)
                continue
            if pooled:
                series = _ledoit_wolf(x)[None]
            else:
                series = np.tile(np.eye(4), (4, 1, 1))
                for q in valid:
                    series[q] = _ledoit_wolf(x[quarter == q])
            per_token = len(x) if pooled else np.median([(quarter == q).sum() for q in valid])
            data.append(series); labels.append(y[i]); n_obs.append(per_token); split.append(role)
        for f in files.values():
            f.close()
    return np.stack(data), np.array(labels), np.array(n_obs), {"split": np.array(split)}
