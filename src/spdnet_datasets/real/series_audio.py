"""
Covariance series from audio clips (atlas v2, sheets ESC-50, GTZAN and BirdSet XCM; same chain as the atlas
script ``audio_common.py``): log-mel spectrogram (64 bands, window ~46 ms, hop = window / 4, floor at max - 80 dB),
20 MFCC (orthonormal DCT-II of the log-mel bands), then one Ledoit-Wolf covariance of the 20 MFCC per time window
of ``window_s`` seconds (step = half a window). Positions are time windows: they carry no identity of their own
(a sound event can fall in any window).

- ``pooled`` (baseline B3, no series): one covariance over the frames of the valid windows instead.
- Clips without any valid window are dropped. Windows whose samples are more than half exact zeros (padding silence, 216 ESC-50 clips are mostly silent) or
  whose covariance has a null trace become identity tokens (masked, E8); series are padded with the identity up to
  ``t_max``.
- ESC-50: 2 000 clips of 5 s, 50 classes, 1 s windows (T = 8, n = 86 frames, n/p = 4.3); groups = source
  recording (``src_file``, the official folds never split one).
- GTZAN: 999 excerpts of 30 s, 10 genres, 3 s windows (T = 19, n = 258 frames, n/p = 13); groups = exact
  duplicates merged (14 pairs, Sturm 2013), the artist is not available.
- UrbanSound8K (R17): 8 732 slices of <= 4 s, 10 classes, resampled to 22.05 kHz, 1 s windows (T = 6, n = 86 frames,
  n/p = 4.3); groups = source recording.
- BirdSet XCM (21 downloaded shards): 5 s crop centred on the first detected event, 1 s windows (T = 8, n = 62
  frames, n/p = 3.1); species with >= ``min_recordings`` recordings; groups = recordist.
"""

import glob
import hashlib
import io
from multiprocessing import Pool

import numpy as np
from scipy.fft import dct
from sklearn.covariance import ledoit_wolf

AUDIO_ROOT = "/DATA/DATASET/AUDIO"


def _mel_filters(sr: int, n_fft: int, n_mels: int = 64, fmin: float = 50.0) -> np.ndarray:
    hz_to_mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    mel_to_hz = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(sr / 2), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    filters = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        left, centre, right = edges[i:i + 3]
        filters[i] = np.maximum(0, np.minimum((bins - left) / (centre - left), (right - bins) / (right - centre)))
    return filters


def _decode(blob: bytes, start_s: float | None = None, duration_s: float | None = None) -> tuple[np.ndarray, int]:
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(blob)) as f:
        sr = f.samplerate
        if start_s is not None:
            f.seek(int(max(0, min(start_s * sr, max(f.frames - duration_s * sr, 0)))))
            x = f.read(int(duration_s * sr), dtype="float32", always_2d=True)
        else:
            x = f.read(dtype="float32", always_2d=True)
    return x.mean(1).astype(np.float64), sr


def mfcc_window_series(x: np.ndarray, sr: int, window_s: float, t_max: int, pooled: bool = False) -> np.ndarray:
    """Signal -> (t_max, 20, 20) Ledoit-Wolf covariances of the 20 MFCC per time window, identity for silent windows
    and after the last window. ``pooled``: (1, 20, 20), one covariance over the frames of the valid windows
    (baseline B3, no series; identity when none is valid)."""
    n_fft = int(2 ** round(np.log2(0.046 * sr)))
    hop = n_fft // 4
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    index = np.arange(n_fft)[None] + hop * np.arange(n_frames)[:, None]
    power = np.abs(np.fft.rfft(x[index] * np.hanning(n_fft)[None], axis=1)) ** 2
    db = 10 * np.log10(np.maximum(power @ _mel_filters(sr, n_fft).T, 1e-10))
    db = np.maximum(db, db.max() - 80.0)  # (frames, 64)
    mfcc = dct(db, type=2, axis=1, norm="ortho")[:, :20]
    w = max(int(round(window_s * sr / hop)), 2)
    series = np.tile(np.eye(20), (t_max, 1, 1))
    frames = np.zeros(n_frames, bool)
    for t, start in enumerate(range(0, max(n_frames - w, 0) + 1, w // 2)):
        if t == t_max:
            break
        samples = x[start * hop:(start + w - 1) * hop + n_fft]
        covariance = ledoit_wolf(mfcc[start:start + w])[0]
        trace = np.trace(covariance) / 20
        if (samples == 0).mean() <= 0.5 and trace > 0:
            series[t] = covariance + 1e-6 * trace * np.eye(20)
            frames[start:start + w] = True
    if pooled:
        if not frames.any():
            return np.eye(20)[None]
        covariance = ledoit_wolf(mfcc[frames])[0]
        return (covariance + 1e-6 * np.trace(covariance) / 20 * np.eye(20))[None]
    return series


def _series_job(args: tuple) -> np.ndarray:
    blob, crop, window_s, t_max, pooled, sr_target = args
    x, sr = _decode(blob) if crop is None else _decode(blob, *crop)
    if sr_target is not None and sr != sr_target:  # UrbanSound8K: rates of 8 to 96 kHz, tied to the source recording
        from math import gcd
        from scipy.signal import resample_poly

        x, sr = resample_poly(x, sr_target // gcd(sr, sr_target), sr // gcd(sr, sr_target)), sr_target
    return mfcc_window_series(x, sr, window_s, t_max, pooled)


def _build(blobs: list, crops: list, window_s: float, t_max: int, jobs: int, pooled: bool = False,
           sr_target: int | None = None) -> np.ndarray:
    with Pool(jobs) as pool:
        args = [(b, c, window_s, t_max, pooled, sr_target) for b, c in zip(blobs, crops)]
        return np.stack(pool.map(_series_job, args, chunksize=8))


def _drop_silent(data: np.ndarray, *arrays: np.ndarray) -> tuple:
    """Drops the series whose windows are all silent (identity only: 4 ESC-50 clips)."""
    keep = (data != np.eye(data.shape[-1])).any((-1, -2)).any(1)
    return (data[keep], *(np.asarray(a)[keep] for a in arrays))


def esc50_series(root: str = f"{AUDIO_ROOT}/ESC-50", window_s: float = 1.0, t_max: int = 8, jobs: int = 3, pooled: bool = False):
    """(2000, 8, 20, 20) covariances, labels 0..49, frames per window, groups = source recording."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(f"{root}/*.parquet"))])
    blobs = [a["bytes"] for a in table.column("audio").to_pylist()]
    data = _build(blobs, [None] * len(blobs), window_s, t_max, jobs, pooled)
    groups = np.unique(table.column("src_file").to_numpy(), return_inverse=True)[1]
    return _drop_silent(data, table.column("target").to_numpy(), np.full(len(blobs), 86), groups)


def gtzan_series(root: str = f"{AUDIO_ROOT}/GTZAN", window_s: float = 3.0, t_max: int = 19, jobs: int = 3, pooled: bool = False):
    """(999, 19, 20, 20) covariances, genres 0..9, frames per window, groups = exact duplicates merged."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(f"{root}/*.parquet"))])
    blobs = [a["bytes"] for a in table.column("audio").to_pylist()]
    data = _build(blobs, [None] * len(blobs), window_s, t_max, jobs, pooled)
    fingerprints = [hashlib.md5(np.round(_decode(b)[0], 4).tobytes()).hexdigest() for b in blobs]
    groups = np.unique(fingerprints, return_inverse=True)[1]
    return _drop_silent(data, table.column("genre").to_numpy(), np.full(len(blobs), 258), groups)


def birdset_series(root: str = f"{AUDIO_ROOT}/BirdSet_XCM", window_s: float = 1.0, t_max: int = 8,
                   min_recordings: int = 10, jobs: int = 3, pooled: bool = False):
    """(N, 8, 20, 20) covariances of a 5 s crop centred on the first detected event, species 0..C-1 with at least
    ``min_recordings`` recordings, frames per window, groups = recordist."""
    import pyarrow.parquet as pq

    blobs, crops, species, recordists = [], [], [], []
    for f in sorted(glob.glob(f"{root}/xcm_*.parquet")):
        table = pq.read_table(f, columns=["audio", "ebird_code", "detected_events", "recordist"])
        for audio, code, events, recordist in zip(*(table.column(c).to_pylist() for c in table.column_names)):
            start = max(0.0, (events[0][0] + events[0][1]) / 2 - 2.5) if events else 0.0
            blobs.append(audio["bytes"]); crops.append((start, 5.0)); species.append(code)
            recordists.append(recordist or "NA")
    codes, counts = np.unique(species, return_counts=True)
    kept = np.isin(species, codes[counts >= min_recordings])
    index = np.nonzero(kept)[0]
    data = _build([blobs[i] for i in index], [crops[i] for i in index], window_s, t_max, jobs, pooled)
    labels = np.unique(np.asarray(species)[index], return_inverse=True)[1]
    groups = np.unique(np.asarray(recordists)[index], return_inverse=True)[1]
    return _drop_silent(data, labels, np.full(len(index), 62), groups)


def urbansound8k_series(root: str = f"{AUDIO_ROOT}/UrbanSound8K", window_s: float = 1.0, t_max: int = 6, jobs: int = 3,
                        pooled: bool = False):
    """(8732, 6, 20, 20) covariances of the urban sound slices (<= 4 s, Salamon et al. 2014), resampled to 22.05 kHz,
    classes 0..9, frames per window, groups = source recording (``fsID``: slices of one Freesound recording, never
    split by the official folds). A slice shorter than a window gives one token over its frames."""
    import pandas as pd

    table = pd.read_csv(f"{root}/metadata/UrbanSound8K.csv")
    blobs = [open(f"{root}/audio/fold{fold}/{name}", "rb").read() for name, fold in zip(table.slice_file_name, table.fold)]
    data = _build(blobs, [None] * len(blobs), window_s, t_max, jobs, pooled, sr_target=22050)
    groups = np.unique(table.fsID.to_numpy(), return_inverse=True)[1]
    return _drop_silent(data, table.classID.to_numpy(), np.full(len(blobs), 86), groups)
