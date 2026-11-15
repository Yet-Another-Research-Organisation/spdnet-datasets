"""
Covariance series from the UEA multivariate archive (EEG, R22): one covariance of the channels per time window.

- Token: channels standardised by their standard deviation over the training trials (statistics of the train rows only,
  after the validation carve), centred within each window, Ledoit-Wolf covariance over the window samples.
- Official TRAIN / TEST files: TEST is the test split; 20 % of TRAIN (stratified by class, drawn once with seed 0, the
  same for every training seed) is the validation split. The archive carries no subject identifier: MotorImagery,
  FingerMovements and SelfRegulationSCP1 come from a single subject each (BCI competition recordings), so the
  protocol is trial-independent, with train and test trials from different sessions / days, not subject-independent.
"""

import numpy as np

from .series_breizhcrops import _ledoit_wolf

UEA_ROOT = "/DATA/DATASET/TIMESERIES/UEA_MTS_2018/Multivariate_ts"
UEA_SHAPES = {  # name: (window, step, T)
    "MotorImagery": (250, 250, 12),  # 3 000 steps, 64 channels: n/p = 3.9
    "FingerMovements": (20, 4, 8),  # 50 steps, 28 channels: n/p = 0.71, windows overlap by 80 %; shrinkage does the rest
    "SelfRegulationSCP1": (56, 56, 16),  # 896 steps, 6 channels: n/p = 9.3
}
UEA_CLASSES = {"MotorImagery": ["finger", "tongue"], "FingerMovements": ["left", "right"],
               "SelfRegulationSCP1": ["negativity", "positivity"]}


def read_ts(path: str, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """sktime .ts file with equal lengths and no missing values -> (N, channels, length), labels."""
    x, y = [], []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "@")):
                continue
            *channels, label = line.strip().split(":")
            x.append([np.array(c.split(","), dtype=np.float64) for c in channels])
            y.append(classes.index(label))
    return np.array(x), np.array(y)


def uea_series(name: str, root: str = UEA_ROOT, channel_step: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(N, T, p, p) covariances per time window, labels, samples per window, {"split": 0 train / 1 val / 2 test}.
    ``channel_step``: keep every k-th channel (reduced variant, p = 64 -> 32)."""
    window, step, T = UEA_SHAPES[name]
    classes = UEA_CLASSES[name]
    x_train, y_train = read_ts(f"{root}/{name}/{name}_TRAIN.ts", classes)
    x_test, y_test = read_ts(f"{root}/{name}/{name}_TEST.ts", classes)
    rng = np.random.default_rng(0)
    val = np.zeros(len(y_train), bool)
    for c in np.unique(y_train):
        idx = rng.permutation(np.flatnonzero(y_train == c))
        val[idx[:round(0.2 * len(idx))]] = True
    x = np.concatenate([x_train, x_test])[:, ::channel_step]
    y = np.concatenate([y_train, y_test])
    split = np.concatenate([np.where(val, 1, 0), np.full(len(y_test), 2)])
    fit = x[split == 0]  # per-channel scaling from the training trials only
    x = (x - fit.mean((0, 2))[None, :, None]) / fit.std((0, 2))[None, :, None]
    data = np.stack([np.stack([_ledoit_wolf(trial[:, t * step:t * step + window].T - trial[:, t * step:t * step + window].T.mean(0))
                               for t in range(T)]) for trial in x])
    return data, y, np.full(len(y), window), {"split": split}




PTBXL_ROOT = "/DATA/DATASET/BIOSIGNAL/PTB-XL"
PTBXL_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]  # diagnostic superclasses


def read_wfdb16(path: str) -> np.ndarray:
    """WFDB format 16 record (``path`` without extension) -> (samples, leads) in physical units (mV), numpy only."""
    with open(path + ".hea") as f:
        lines = [line.split() for line in f if line.strip() and not line.startswith("#")]
    n_leads, n_samples = int(lines[0][1]), int(lines[0][3])
    gain, baseline = [], []
    for fields in lines[1:1 + n_leads]:
        assert fields[1].startswith("16"), fields
        gain.append(float(fields[2].split("/")[0].split("(")[0]))
        baseline.append(float(fields[2].split("(")[1].split(")")[0]) if "(" in fields[2] else 0.0)
    raw = np.fromfile(path + ".dat", dtype="<i2").reshape(n_samples, n_leads).astype(np.float64)
    return (raw - np.array(baseline)) / np.array(gain)


def ptbxl_series(root: str = PTBXL_ROOT, window: int = 100, T: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """PTB-XL 100 Hz, 12 leads: (N, T = 10, 12, 12) Ledoit-Wolf covariances of non-overlapping 1 s windows, labels
    0..4 (NORM, MI, STTC, CD, HYP), samples per window, {"split": 0 train / 1 val / 2 test}.
    Single-label records only (exactly one diagnostic superclass in ``scp_codes``); official ``strat_fold``: 1-8 train,
    9 validation, 10 test. Leads standardised by their std over the training records; windows centred."""
    import ast

    import pandas as pd

    db = pd.read_csv(f"{root}/ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(f"{root}/scp_statements.csv", index_col=0)
    superclass = scp[scp.diagnostic == 1].diagnostic_class.to_dict()
    ids, y = [], []
    for ecg_id, codes in db.scp_codes.items():
        classes = {superclass[c] for c in ast.literal_eval(codes) if c in superclass}
        if len(classes) == 1:
            ids.append(ecg_id); y.append(PTBXL_CLASSES.index(classes.pop()))
    y = np.array(y)
    fold = db.loc[ids, "strat_fold"].to_numpy()
    split = np.where(fold <= 8, 0, np.where(fold == 9, 1, 2))
    x = np.stack([read_wfdb16(f"{root}/{db.loc[i, 'filename_lr']}")[:window * T].T for i in ids])  # (N, 12, 1000)
    fit = x[split == 0]
    x = (x - fit.mean((0, 2))[None, :, None]) / fit.std((0, 2))[None, :, None]
    data = np.stack([np.stack([_ledoit_wolf(r[:, t * window:(t + 1) * window].T - r[:, t * window:(t + 1) * window].T.mean(0))
                               for t in range(T)]) for r in x])
    return data, y, np.full(len(y), window), {"split": split}
