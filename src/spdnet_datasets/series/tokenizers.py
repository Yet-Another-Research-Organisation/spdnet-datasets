"""Tokenizers: cut one structured sample into the observation matrices of its tokens.

Every tokenizer is a pure function of an array. It returns ``(observations, positions)``:

- ``observations``: list of ``(n_t, p)`` float arrays, one per token, in series order; the
  covariance of token ``k`` is estimated from ``observations[k]``;
- ``positions``: list of the same length; ``positions[k]`` gives, for every axis of the input,
  the ``(start, stop)`` interval covered by token ``k`` (``(0, size)`` for an axis that is not
  cut). :func:`parts` gives the tuple of variable indices of each group instead.

Windows along one axis follow :func:`spdnet_datasets.real.series.window_starts`: ``window`` and
``step`` (default ``window // 2``, 50 % overlap) with a last window flush with the end when the
regular starts leave it uncovered, or ``n_windows`` windows spread evenly. With ``n_windows`` and
no ``window``, the window is ``2 n / (n_windows + 1)``: ``n_windows`` windows that overlap by
50 % and cover the whole axis. An axis shorter than ``window`` gives a single window over the
whole axis.

The variables of a token are the last axis of the input (frequency bins, MFCC, bands), possibly
restricted to a band window (:func:`spectral`); every other axis becomes observations.
"""

import numpy as np

from ..real.series import window_starts

__all__ = ["axis_windows", "n_windows", "parts", "pool_channels", "spatial", "spatio_spectral",
           "spatio_temporal", "spectral", "spectro_temporal", "temporal"]


def axis_windows(n, window=None, step=None, n_windows=None, cover=True):
    """``(start, stop)`` of the windows along an axis of ``n`` positions (see module docstring).

    ``cover=False`` keeps the regular starts only (the end of the axis may stay uncovered), as
    :func:`~spdnet_datasets.real.series.sliding_covariances` does.
    """
    if window is None:
        if n_windows is None:
            raise ValueError("give window or n_windows")
        window = max(1, int(2 * n / (n_windows + 1)))
    window = int(window)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if n <= window:
        return [(0, n)]
    if n_windows is None:
        step = int(step) if step is not None else max(1, window // 2)
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        starts = window_starts(n, window, step, cover=cover)
    else:
        starts = window_starts(n, window, n_windows=int(n_windows))
    return [(s, s + window) for s in starts]


def n_windows(n, window=None, step=None, n_windows=None, cover=True):
    """Number of windows :func:`axis_windows` gives along an axis of ``n`` positions."""
    return len(axis_windows(n, window, step, n_windows, cover))


def pool_channels(x, n_channels=None):
    """Average the last axis of ``x`` into ``n_channels`` contiguous groups (``None``: unchanged).

    Groups have equal sizes when ``n_channels`` divides the axis, sizes differing by one otherwise.
    """
    if n_channels is None or n_channels == x.shape[-1]:
        return x
    if not 1 <= n_channels <= x.shape[-1]:
        raise ValueError(f"cannot pool {x.shape[-1]} channels into {n_channels}")
    edges = np.linspace(0, x.shape[-1], n_channels + 1).round().astype(int)
    return np.add.reduceat(np.asarray(x, dtype=np.float64), edges[:-1], axis=-1) / np.diff(edges)


def _full(x):
    return [(0, n) for n in x.shape]


def temporal(x, window=None, step=None, n_windows=None, cover=True):
    """Windows along the first axis of ``x`` (T, ..., p); observations = the frames of the window
    (and every middle axis), variables = the last axis."""
    obs, pos = [], []
    for a, b in axis_windows(x.shape[0], window, step, n_windows, cover):
        obs.append(x[a:b].reshape(-1, x.shape[-1]))
        pos.append([(a, b), *_full(x)[1:]])
    return obs, pos


def spectral(x, band_window, band_step=None, n_windows=None, cover=True):
    """Windows along the last axis of ``x`` (..., B); observations = every other position.

    Same windows as :func:`spdnet_datasets.real.series.spectral_windows` when ``band_step``
    tiles the axis; otherwise a last window flush with the end is added (unless ``cover=False``,
    which gives exactly the windows of ``spectral_windows``).
    """
    rows = x.reshape(-1, x.shape[-1])
    obs, pos = [], []
    for a, b in axis_windows(x.shape[-1], band_window, band_step, n_windows, cover):
        obs.append(rows[:, a:b])
        pos.append([*_full(x)[:-1], (a, b)])
    return obs, pos


def spatial(x, window, step=None, n_windows=None):
    """Grid of spatial windows on ``x`` (H, W, C), row-major; observations = the pixels of the
    window, variables = the C channels.

    ``window``, ``step`` and ``n_windows`` are ``(rows, columns)`` pairs (``step`` or
    ``n_windows`` may be ``None``).
    """
    step = step if step is not None else (None, None)
    counts = n_windows if n_windows is not None else (None, None)
    rows = axis_windows(x.shape[0], window[0], step[0], counts[0])
    cols = axis_windows(x.shape[1], window[1], step[1], counts[1])
    obs, pos = [], []
    for r0, r1 in rows:
        for c0, c1 in cols:
            obs.append(x[r0:r1, c0:c1].reshape(-1, x.shape[-1]))
            pos.append([(r0, r1), (c0, c1), (0, x.shape[-1])])
    return obs, pos


def spatio_spectral(x, spatial_params, spectral_params):
    """Spatial windows x band windows on ``x`` (H, W, B), spatial-major: the band windows of the
    first spatial window, then those of the second, ... ``spatial_params`` and
    ``spectral_params`` are the keyword arguments of :func:`spatial` and :func:`spectral`."""
    obs, pos = [], []
    for o, (r, c, _) in zip(*spatial(x, **spatial_params), strict=True):
        window = o.reshape(r[1] - r[0], c[1] - c[0], x.shape[-1])
        for o2, p2 in zip(*spectral(window, **spectral_params), strict=True):
            obs.append(o2)
            pos.append([r, c, p2[-1]])
    return obs, pos


def spectro_temporal(x, temporal_params, spectral_params):
    """Time windows x frequency sub-bands on a spectrogram ``x`` (T, F), time-major: every
    sub-band of the first time window, then of the second, ..."""
    obs, pos = [], []
    for o, (t, _) in zip(*temporal(x, **temporal_params), strict=True):
        for o2, p2 in zip(*spectral(o, **spectral_params), strict=True):
            obs.append(o2)
            pos.append([t, p2[-1]])
    return obs, pos


def spatio_temporal(x, temporal_params, spatial_params):
    """Time windows x spatial windows on ``x`` (T, H, W, C), time-major; observations = the
    frames of the time window times the pixels of the spatial window."""
    obs, pos = [], []
    for o, (t, *_) in zip(*temporal(x, **temporal_params), strict=True):
        block = np.moveaxis(o.reshape(t[1] - t[0], *x.shape[1:]), 0, 2)  # (H, W, t, C)
        for o2, (r, c, _) in zip(*spatial(block.reshape(*x.shape[1:3], -1), **spatial_params),
                                 strict=True):
            obs.append(o2.reshape(-1, x.shape[-1]))
            pos.append([t, r, c, (0, x.shape[-1])])
    return obs, pos


def parts(x, groups):
    """Groups of variables of ``x`` (..., p): token ``k`` keeps the channels ``groups[k]`` (body
    parts, sensors, polarisations); observations = every other position."""
    rows = x.reshape(-1, x.shape[-1])
    groups = [tuple(int(i) for i in g) for g in groups]
    return [rows[:, list(g)] for g in groups], groups
