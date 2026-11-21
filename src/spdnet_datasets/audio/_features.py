"""Frame features of an audio signal: log-mel bands and MFCC.

Same chain as the atlas script ``audio_common.py`` and the covariance-series
module ``real/series_audio.py`` (branch ``feat/covariance-series``):

- power spectrum on Hann frames of ``n_fft = 2 ** round(log2(0.046 * sr))``
  samples (about 46 ms) with a hop of ``n_fft / 4``;
- ``n_mels = 64`` triangular mel filters from 50 Hz to Nyquist, in dB with a
  floor at ``max - 80 dB``;
- MFCC = orthonormal DCT-II of the log-mel bands, first ``n_mfcc`` coefficients.

Frames more than half made of exact zeros (digital silence, padding) are
dropped: they carry no information and make the covariance singular.
"""

import io

import numpy as np

FEATURES = ("mfcc", "logmel")


def mel_filters(sr, n_fft, n_mels=64, fmin=50.0):
    """Triangular mel filter bank of shape ``(n_mels, n_fft // 2 + 1)``."""
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(sr / 2), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    filters = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        left, centre, right = edges[i:i + 3]
        filters[i] = np.maximum(0, np.minimum((bins - left) / (centre - left),
                                              (right - bins) / (right - centre)))
    return filters


def decode(blob, start_s=None, duration_s=None):
    """Decode an encoded audio file (WAV, OGG, ...) to a mono float64 signal.

    Returns ``(x, sr)``. With ``start_s`` and ``duration_s`` only that excerpt
    is read (clipped to the file).
    """
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(blob)) as f:
        sr = f.samplerate
        if start_s is not None:
            start = int(max(0, min(start_s * sr, max(f.frames - duration_s * sr, 0))))
            f.seek(start)
            x = f.read(int(duration_s * sr), dtype="float32", always_2d=True)
        else:
            x = f.read(dtype="float32", always_2d=True)
    return x.mean(axis=1).astype(np.float64), sr


def frame_features(x, sr, features="mfcc", n_mfcc=20, n_mels=64):
    """Features of the non-silent frames of signal ``x``.

    Parameters
    ----------
    x : array (n_samples,)
    sr : int
        Sampling rate (Hz).
    features : {"mfcc", "logmel"}
        ``mfcc``: ``n_mfcc`` coefficients (p = n_mfcc); ``logmel``: the
        ``n_mels`` log-mel bands in dB (p = n_mels).

    Returns
    -------
    array (n_frames, p) float64
        One row per non-silent frame.
    """
    from scipy.fft import dct

    if features not in FEATURES:
        raise ValueError(f"features must be one of {FEATURES}, got {features!r}")
    n_fft = int(2 ** round(np.log2(0.046 * sr)))
    hop = n_fft // 4
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    index = np.arange(n_fft)[None] + hop * np.arange(n_frames)[:, None]
    frames = x[index]
    silent = (frames == 0).mean(axis=1) > 0.5
    power = np.abs(np.fft.rfft(frames * np.hanning(n_fft)[None], axis=1)) ** 2
    db = 10 * np.log10(np.maximum(power @ mel_filters(sr, n_fft, n_mels).T, 1e-10))
    db = np.maximum(db, db.max() - 80.0)
    out = db if features == "logmel" else dct(db, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return out[~silent]


def iter_parquet_rows(files, columns, positions=None):
    """Yield ``(position, row dict)`` of the parquet ``files`` read in order.

    ``positions`` (sorted global row numbers) restricts the rows that are
    materialised; the others are skipped batch by batch.
    """
    import pyarrow.parquet as pq

    wanted = None if positions is None else {int(i) for i in positions}
    start = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=columns, batch_size=64):
            stop = start + batch.num_rows
            if wanted is None or any(start <= i < stop for i in wanted):
                for k, row in enumerate(batch.to_pylist()):
                    if wanted is None or start + k in wanted:
                        yield start + k, row
            start = stop


def n_features(features, n_mfcc=20, n_mels=64):
    """Dimension ``p`` of the frame features."""
    return n_mfcc if features == "mfcc" else n_mels


def process_recordings(records, out_dir, *, features="mfcc", n_mfcc=20, n_mels=64,
                       min_frames=None):
    """Processed stage of an audio dataset.

    Parameters
    ----------
    records : iterable of dict
        One dict per recording with ``blob`` (encoded audio), optionally
        ``crop`` (``(start_s, duration_s)``), and the index columns (``id``,
        ``label``, ``group``, ...).
    out_dir : Path
        Processed stage directory (see :func:`._common.write_processed`).

    Recordings with fewer than ``min_frames`` non-silent frames (default ``2p``) are excluded
    and counted. A smaller ``min_frames`` (at least 1: silent recordings stay excluded) stores
    the short recordings too, for the covariance series (their own ``p x p`` covariance is then
    excluded at the covariance stage).
    """
    from ._common import write_processed

    p = n_features(features, n_mfcc, n_mels)
    least = 2 * p if min_frames is None else max(int(min_frames), 1)
    feats, index, excluded = [], [], []
    for rec in records:
        blob, crop = rec.pop("blob"), rec.pop("crop", None)
        x, sr = decode(blob, *crop) if crop else decode(blob)
        F = frame_features(x, sr, features, n_mfcc, n_mels)
        if len(F) < least:
            reason = (f"n < 2p ({len(F)} non-silent frames, 2p = {2 * p})" if min_frames is None
                      else "silent (no non-silent frame)" if len(F) == 0
                      else f"n < {least} ({len(F)} non-silent frames)")
            excluded.append({"id": rec["id"], "label": rec["label"], "reason": reason})
            continue
        rec["sr"] = sr
        feats.append(F.astype(np.float32))
        index.append(rec)
    write_processed(out_dir, feats, index, excluded, p)
