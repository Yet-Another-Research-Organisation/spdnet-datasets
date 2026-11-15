"""
R23 (user request, 10/11): covariance series with larger p, to separate the matrix size from the kind of token.

True time series (tokens are instants / time windows):
- :func:`ntu_window_series`: NTU RGB+D skeletons of ``ntu_frame`` (same sequences, classes, subjects, split), but the
  token is the covariance of the 3 x (number of joints) coordinates (25 joints x 3 = 75; 15 joints x 3 = 45) over
  ``n_windows`` time windows of a quarter of the sequence, evenly spread (they overlap). The root joint is kept
  (coordinates not centred on it; the mean of each window is removed by the covariance). Ledoit-Wolf, n/p << 1.
- :func:`audio_mel_series`: ESC-50 / GTZAN clips of ``esc50_series`` / ``gtzan_series`` (same audio front end,
  windows, T, groups), but the token is the Ledoit-Wolf covariance of the 64 log-mel bands (dB, floor max - 80 dB)
  instead of the 20 MFCC.
- :func:`mvdoppler_window_series`: MVDoppler snapshots of ``mvdoppler_series`` (same files, caps, split), but the
  64 central Doppler bins are kept without averaging (p = 64) over the same windows of 64 frames (n/p = 1).
"""

import glob
import hashlib

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

from .series_audio import AUDIO_ROOT, _decode, _drop_silent, _mel_filters
from .series_breizhcrops import _cap_stratified, _ledoit_wolf
from .series_mvdoppler import FOLDS, MVDOPPLER_CLASSES, ROOT, _readable  # noqa: F401
from .series_ntu import ntu_skeletons

NTU_JOINTS_15 = (0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18)
"""Base of the spine, neck, head, shoulders, elbows, wrists, hips, knees and ankles (no spine middle, hands, feet,
hand tips or thumbs), 0-based indices of the 25 NTU joints."""


def ntu_window_series(joints: tuple[int, ...] | None = None, n_windows: int = 8, **kwargs):
    """(n, n_windows, 3 J, 3 J) covariances, labels, frames per window, {"split", "subject", "camera"}."""
    skeletons, labels, extra = ntu_skeletons(centre=False, **kwargs)
    data, n_obs = [], []
    for x in skeletons:
        x = x if joints is None else x[:, list(joints)]
        x = x.reshape(len(x), -1)
        width = max(len(x) // 4, 3)
        starts = np.round(np.linspace(0, len(x) - width, n_windows)).astype(int)
        data.append(np.stack([_ledoit_wolf(x[s:s + width]) for s in starts]))
        n_obs.append(width)
    return np.stack(data), labels, np.array(n_obs), extra


def _mel_window_series(x: np.ndarray, sr: int, window_s: float, t_max: int) -> np.ndarray:
    n_fft = int(2 ** round(np.log2(0.046 * sr)))
    hop = n_fft // 4
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    index = np.arange(n_fft)[None] + hop * np.arange(n_frames)[:, None]
    power = np.abs(np.fft.rfft(x[index] * np.hanning(n_fft)[None], axis=1)) ** 2
    db = 10 * np.log10(np.maximum(power @ _mel_filters(sr, n_fft).T, 1e-10))
    db = np.maximum(db, db.max() - 80.0)  # (frames, 64)
    w = max(int(round(window_s * sr / hop)), 2)
    series = np.tile(np.eye(64), (t_max, 1, 1))
    for t, start in enumerate(range(0, max(n_frames - w, 0) + 1, w // 2)):
        if t == t_max:
            break
        samples = x[start * hop:(start + w - 1) * hop + n_fft]
        covariance = ledoit_wolf(db[start:start + w])[0]
        trace = np.trace(covariance) / 64
        if (samples == 0).mean() <= 0.5 and trace > 0:
            series[t] = covariance + 1e-6 * trace * np.eye(64)
    return series


def _job(args: tuple) -> np.ndarray:
    blob, window_s, t_max = args
    x, sr = _decode(blob)
    return _mel_window_series(x, sr, window_s, t_max)


def audio_mel_series(dataset: str, jobs: int = 3):
    """dataset "esc50" or "gtzan": (N, T, 64, 64) covariances of the log-mel bands, labels, frames per window, groups
    (identical to ``esc50_series`` / ``gtzan_series``)."""
    from multiprocessing import Pool

    import pyarrow as pa
    import pyarrow.parquet as pq

    root, window_s, t_max, label_column, n_frames = {
        "esc50": (f"{AUDIO_ROOT}/ESC-50", 1.0, 8, "target", 86), "gtzan": (f"{AUDIO_ROOT}/GTZAN", 3.0, 19, "genre", 258)}[dataset]
    table = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(f"{root}/*.parquet"))])
    blobs = [a["bytes"] for a in table.column("audio").to_pylist()]
    with Pool(jobs) as pool:
        data = np.stack(pool.map(_job, [(b, window_s, t_max) for b in blobs], chunksize=8))
    if dataset == "esc50":
        groups = np.unique(table.column("src_file").to_numpy(), return_inverse=True)[1]
    else:
        fingerprints = [hashlib.md5(np.round(_decode(b)[0], 4).tobytes()).hexdigest() for b in blobs]
        groups = np.unique(fingerprints, return_inverse=True)[1]
    return _drop_silent(data, table.column(label_column).to_numpy(), np.full(len(blobs), n_frames), groups)


def _doppler_job(path: str) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        x = np.log(np.clip(f["radar_dat"][32:96, :, 0], 1e-12, None)).astype(np.float64).T  # (256 frames, 64 bins)
    return np.stack([_ledoit_wolf(x[s:s + 64]) for s in range(0, 256 - 64 + 1, 32)])


def mvdoppler_window_series(root: str = ROOT, caps: tuple[int, int, int] = (20000, 5000, 10000), stride: int = 3,
                           jobs: int = 3):
    """(N, 7, 64, 64) covariances (view x, 64 Doppler bins, windows of 64 frames step 32). The files, labels and split are
    those of ``mvdoppler_series``, of which one snapshot out of ``stride`` is kept (35 000 x 7 x 64 x 64 float64 is
    8 GB); ``parent_index`` = row in the ``mvdoppler`` cache."""
    from multiprocessing import Pool

    design = pd.read_csv(f"{root}/Labels_and_metadata/design_table.csv")
    design = design[[_readable(f"{root}/Data/{name}.h5") for name in design.fname]]
    rng = np.random.default_rng(0)
    paths, labels, split = [], [], []
    for role in range(3):
        rows = design[design.fold.map(FOLDS) == role].reset_index(drop=True)
        y = rows.pattern.map(MVDOPPLER_CLASSES.index).to_numpy()
        keep = _cap_stratified(y, caps[role], rng)
        paths += [f"{root}/Data/{name}.h5" for name in rows.fname[keep]]
        labels.append(y[keep]); split.append(np.full(len(keep), role))
    keep = np.arange(0, len(paths), stride)
    with Pool(jobs) as pool:
        data = np.stack(pool.map(_doppler_job, [paths[i] for i in keep], chunksize=64))
    return (data, np.concatenate(labels)[keep], np.full(len(keep), 64),
            {"split": np.concatenate(split)[keep], "parent_index": keep})
