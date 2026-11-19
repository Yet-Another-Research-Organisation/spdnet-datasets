"""Registry of the covariance-series modes of each dataset.

``SERIES_MODES[dataset][mode]`` is a :class:`SeriesMode`; the first mode of a dataset is its
default. A mode says where the structured sample comes from (``source``, by default the
``structured_raw(i)`` hook of the loader), how it is cut into tokens (``tokenize``), and the
default parameters, chosen by rules only (no benchmark):

1. ``n_t >= 2p`` for every token, ``>= 4p`` when the data allow it;
2. tokens that cover the whole sample, with 0 or 50 % overlap;
3. a number of tokens S between 4 and 32;
4. ``p`` from the atlas or the user's recommendations when they exist;
5. a degenerate token (too few observations, constant, not SPD) becomes a masked identity token.

Every default can be overridden: ``CovarianceSeries(name, mode, **params)``.

Importing this module reads no data and imports no loader (the documentation builds the
tables of :doc:`/datasets/series_overview` from it).
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from . import tokenizers as tok

__all__ = ["FAMILIES", "SERIES_MODES", "SeriesMode", "get_mode"]

FAMILIES = ("temporal", "vision")


def structured_raw(ds, i):
    """Default source: the ``structured_raw(i)`` hook of the wrapped loader."""
    return ds.structured_raw(i)


@dataclass(frozen=True)
class SeriesMode:
    """One way of cutting the samples of a dataset into a covariance series.

    Attributes
    ----------
    name : str
        Mode name (``"temporal"``, ``"spectral"``, ...).
    tokenize : callable(x, **params) -> (observations, positions)
        Built on :mod:`spdnet_datasets.series.tokenizers`.
    axes : tuple of str
        Names of the axes of the structured sample (``("T", "F")``, ``("H", "W", "B")``).
    defaults : dict
        Default parameters of ``tokenize`` (all of them; any can be overridden).
    description : str
        One line: what a token is.
    example_shape : tuple of int
        Typical shape of a structured sample, used by the documentation tables and diagrams.
    loader_defaults : dict
        Loader arguments the mode needs (e.g. ``features="logmel"``): used when the loader is
        built by name, and checked against a loader instance.
    family : {"temporal", "vision"}
        ``"temporal"``: a time series of covariances (tokens ordered along time: audio frames,
        radar time windows, dates). ``"vision"``: tokens "as in a vision transformer", windows
        of one measurement (spatial and/or spectral windows of a hyperspectral cube, sub-bands
        of a whole recording). The two families are never mixed in the tables.
    estimator : str
        Default covariance estimator of the tokens.
    allow_small_n : bool
        Default of ``allow_small_n``: ``True`` only for the documented derogations of R23
        (large-p tokens with ``n_t < 2p``, Ledoit-Wolf).
    source : callable(ds, i) -> array
        Structured sample ``i`` of the wrapped loader (unscaled).
    drop_invalid : bool
        Drop the observations that are all zero or not finite (image background) before
        estimating each token.
    min_n_over_p : float
        Smallest ``n_t / p`` accepted without ``allow_small_n``.
    """

    name: str
    tokenize: Callable
    axes: tuple
    defaults: dict
    description: str
    example_shape: tuple
    family: str = "vision"
    estimator: str = "scm"
    allow_small_n: bool = False
    loader_defaults: dict = field(default_factory=dict)
    source: Callable = structured_raw
    drop_invalid: bool = False
    min_n_over_p: float = 2.0

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")

    def resolve(self, **params):
        """Default parameters updated with ``params`` (unknown names raise ``ValueError``)."""
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"mode {self.name!r} has no parameter {sorted(unknown)}; "
                             f"parameters: {sorted(self.defaults)}")
        return {**self.defaults, **params}

    def nominal(self, shape=None, **params):
        """``(S, p, n_t)`` for a sample of ``shape`` (default ``example_shape``), every observation
        valid; ``n_t`` is the smallest token."""
        x = np.zeros(shape or self.example_shape, dtype=np.float32)
        obs, _ = self.tokenize(x, **self.resolve(**params))
        return len(obs), obs[0].shape[1], min(len(o) for o in obs)


# ---------------------------------------------------------------------------
# Tokenize functions of the modes (keyword parameters = the mode defaults)


def _temporal(x, *, window=None, step=None, n_windows=None, n_channels=None, cover=True):
    return tok.temporal(tok.pool_channels(x, n_channels), window, step, n_windows, cover)


def _spectral(x, *, band_window, band_step=None, n_windows=None, cover=True):
    return tok.spectral(x, band_window, band_step, n_windows, cover)


def _spectro_temporal(x, *, n_windows=4, window=None, band_window=16, band_step=None):
    return tok.spectro_temporal(x, {"window": window, "n_windows": n_windows},
                                {"band_window": band_window, "band_step": band_step})


def _spatial(x, *, window, step=None, n_windows=None, n_channels=None):
    return tok.spatial(tok.pool_channels(x, n_channels), window, step, n_windows)


def _spatio_spectral(x, *, window, step=None, n_windows=None, band_window, band_step=None,
                     band_n_windows=None):
    return tok.spatio_spectral(x, {"window": window, "step": step, "n_windows": n_windows},
                               {"band_window": band_window, "band_step": band_step,
                                "n_windows": band_n_windows})


# ---------------------------------------------------------------------------
# Families of datasets


def _spectrogram_modes(shape, *, window=None, step=None, n_windows=None, n_channels,
                       band_step=8, loader_spectral=None, estimator="scm", unit="frames",
                       cover=True):
    """Modes of a (T, F) sample: audio frames x MFCC / log-mel, micro-Doppler time x bins."""
    loader_spectral = loader_spectral or {}
    spectral_shape = (shape[0], 64) if loader_spectral else shape
    cut = (f"{n_windows} windows of {window} {unit}" if n_windows
           else f"window of {window} {unit} (step {step}"
           + (", end not padded)" if not cover else ")"))
    return {
        "temporal": SeriesMode(
            "temporal", _temporal, ("T", "F"),
            {"window": window, "step": step, "n_windows": n_windows, "n_channels": n_channels,
             "cover": cover},
            f"{cut}; variables = the F channels"
            + (f" pooled into {n_channels}" if n_channels else ""), shape,
            family="temporal", estimator=estimator),
        "spectral": SeriesMode(
            "spectral", _spectral, ("T", "F"),
            {"band_window": 16, "band_step": band_step, "n_windows": None, "cover": True},
            f"sub-band of 16 channels (step {band_step}) over the whole recording; "
            f"observations = every {unit[:-1]}", spectral_shape, family="vision",
            estimator=estimator, loader_defaults=loader_spectral),
        "spectro_temporal": SeriesMode(
            "spectro_temporal", _spectro_temporal, ("T", "F"),
            {"n_windows": 4, "window": None, "band_window": 16, "band_step": 16},
            "4 time windows (50 % overlap) x sub-bands of 16 channels, time-major",
            spectral_shape, family="temporal", estimator=estimator,
            loader_defaults=loader_spectral),
    }


def _patch_modes(shape, *, band_window=16, band_step=8, window, step, n_channels, half,
                 band_window_ss=16, estimator="scm"):
    """Modes of an (H, W, B) hyperspectral patch (all of the "vision" family)."""
    return {
        "spectral": SeriesMode(
            "spectral", _spectral, ("H", "W", "B"),
            {"band_window": band_window, "band_step": band_step, "n_windows": None,
             "cover": True},
            f"window of {band_window} bands (step {band_step}); observations = the pixels",
            shape, estimator=estimator, drop_invalid=True),
        "spatial": SeriesMode(
            "spatial", _spatial, ("H", "W", "B"),
            {"window": window, "step": step, "n_windows": None, "n_channels": n_channels},
            f"{window[0]} x {window[1]} sub-window (step {step[0]}); variables = the bands"
            + (f" pooled into {n_channels}" if n_channels else ""), shape,
            estimator=estimator, drop_invalid=True),
        "spatio_spectral": SeriesMode(
            "spatio_spectral", _spatio_spectral, ("H", "W", "B"),
            {"window": half, "step": half, "n_windows": None, "band_window": band_window_ss,
             "band_step": band_window_ss, "band_n_windows": None},
            f"{half[0]} x {half[1]} quadrant x window of {band_window_ss} bands, spatial-major",
            shape, estimator=estimator, drop_invalid=True),
    }


def _variant(mode, name, description, **changes):
    """Copy of ``mode`` under another name, with other defaults (``defaults=``) or attributes."""
    defaults = {**mode.defaults, **changes.pop("defaults", {})}
    return replace(mode, name=name, description=description, defaults=defaults, **changes)


def _large_p_spectral(modes, windows):
    """R23 (``real/series_large_p.py``, ``experiments/real_data.py``): spectral windows of
    48 / 64 / 96 bands, ``{p: band_step}``."""
    for p, step in windows.items():
        modes[f"spectral_w{p}"] = _variant(
            modes["spectral"], f"spectral_w{p}",
            f"R23 large p: window of {p} bands (step {step}, end not padded, as in R23); "
            "observations = the pixels",
            defaults={"band_window": p, "band_step": step, "cover": False})
    return modes


_AUDIO_LOGMEL = {"features": "logmel"}


def _audio_modes(shape, window, step):
    modes = _spectrogram_modes(shape, window=window, step=step, n_channels=None,
                               loader_spectral=_AUDIO_LOGMEL)
    modes["temporal_mel64"] = _variant(
        modes["temporal"], "temporal_mel64",
        f"R23 large p: window of {window} frames (step {step}), 64 log-mel bands",
        example_shape=(shape[0], 64), loader_defaults=_AUDIO_LOGMEL, estimator="ledoit_wolf",
        allow_small_n=2 * 64 > window)
    return modes


def _dopnet_bands(ds, i):
    """Dop-NET source: 128 central Doppler bins in 16 bands (``real.series.dopnet_series``)."""
    return ds.doppler_bands(i)


SERIES_MODES = {
    # audio: frames x 20 MFCC (temporal) or x 64 log-mel bands (other modes); 1 s / 3 s windows
    "esc50": _audio_modes((427, 20), 86, 43),
    "gtzan": _audio_modes((2584, 20), 258, 129),
    "birdset_xcm": _spectrogram_modes((312, 20), window=62, step=31, n_channels=None,
                                      loader_spectral=_AUDIO_LOGMEL),
    # micro-Doppler: time x Doppler bins of the loader
    # CI4R (real.series.ci4r_series): central half of the Doppler axis in 16 bands, 13 windows
    # of 200 columns, Ledoit-Wolf
    "ci4r": _spectrogram_modes((1475, 64), window=200, step=100, n_channels=16,
                               estimator="ledoit_wolf", unit="columns", cover=False),
    "mvdoppler": _spectrogram_modes((256, 64), window=64, step=32, n_channels=16, band_step=16),
    "dopnet": _spectrogram_modes((180, 48), window=48, step=24, n_channels=16, cover=False),
    # hyperspectral patches: H x W x bands of the loader
    "gsoff": _patch_modes((15, 15, 63), window=(8, 8), step=(4, 4), n_channels=16, half=(8, 8)),
    "chikusei": _patch_modes((15, 15, 64), window=(8, 8), step=(4, 4), n_channels=16,
                             half=(8, 8)),
    "uav": _large_p_spectral(
        _patch_modes((15, 15, 100), window=(8, 8), step=(4, 4), n_channels=16, half=(8, 8),
                     estimator="ledoit_wolf"), {48: 8, 64: 6}),
    "kaggle_wheat": _large_p_spectral(
        _patch_modes((32, 32, 107), window=(16, 16), step=(8, 8), n_channels=16,
                     half=(16, 16), estimator="ledoit_wolf"), {48: 8, 64: 8}),
    "placenta": _patch_modes((24, 24, 37), band_window=8, band_step=4, window=(12, 12),
                             step=(6, 6), n_channels=None, half=(12, 12), band_window_ss=8),
}

# Dop-NET temporal (real.series.dopnet_series): log power of the 128 central bins in 16 bands,
# read again from the .mat files; windows of 48 frames, step 24; identity padding
# (collate_series). The other modes cut the 48 energy-band channels of the loader.
SERIES_MODES["dopnet"]["temporal"] = _variant(
    SERIES_MODES["dopnet"]["temporal"], "temporal",
    "window of 48 frames (step 24); variables = the 128 central Doppler bins in 16 bands",
    defaults={"n_channels": None}, source=_dopnet_bands, example_shape=(180, 16))

# MVDoppler, R23 large p: the 64 channels (2 views x 32 bins) without pooling, Ledoit-Wolf
SERIES_MODES["mvdoppler"]["temporal_w64"] = _variant(
    SERIES_MODES["mvdoppler"]["temporal"], "temporal_w64",
    "R23 large p: window of 64 frames (step 32), the 64 channels (2 views x 32 bins)",
    defaults={"n_channels": None}, estimator="ledoit_wolf", allow_small_n=True)

# HyperLeaf: whole image 48 x 352 x 204, user example first (6 spatial x 4 band windows)
_HYPERLEAF = (48, 352, 204)
SERIES_MODES["hyperleaf"] = _large_p_spectral({
    "spatio_spectral": SeriesMode(
        "spatio_spectral", _spatio_spectral, ("H", "W", "B"),
        {"window": (48, 58), "step": None, "n_windows": (1, 6), "band_window": 48,
         "band_step": None, "band_n_windows": 4},
        "6 windows of 48 x 58 pixels along the leaf x 4 windows of 48 bands, spatial-major; "
        "observations = the leaf pixels", _HYPERLEAF, drop_invalid=True),
    "spatial": SeriesMode(
        "spatial", _spatial, ("H", "W", "B"),
        {"window": (48, 58), "step": None, "n_windows": (1, 6), "n_channels": None},
        "6 windows of 48 x 58 pixels along the leaf, all 204 bands; observations = the leaf "
        "pixels", _HYPERLEAF, drop_invalid=True),
    "spectral": SeriesMode(
        "spectral", _spectral, ("H", "W", "B"),
        {"band_window": 16, "band_step": 8, "n_windows": None, "cover": True},
        "window of 16 bands (step 8); observations = the leaf pixels", _HYPERLEAF,
        drop_invalid=True),
}, {48: 16, 64: 16, 96: 16})


def get_mode(dataset, mode=None):
    """The :class:`SeriesMode` ``mode`` of ``dataset`` (its default mode when ``None``)."""
    if dataset not in SERIES_MODES:
        raise ValueError(f"No covariance-series mode for {dataset!r}; available: "
                         f"{sorted(SERIES_MODES)}")
    modes = SERIES_MODES[dataset]
    if mode is None:
        return next(iter(modes.values()))
    if mode not in modes:
        raise ValueError(f"{dataset!r} has no mode {mode!r}; modes: {list(modes)}")
    return modes[mode]
