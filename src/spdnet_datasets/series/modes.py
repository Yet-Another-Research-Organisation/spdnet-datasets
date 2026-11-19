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
    tokenize : callable
        ``tokenize(x, **params) -> (observations, positions)``,
        built on :mod:`spdnet_datasets.series.tokenizers`.
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
    prepare : callable(x) -> x, optional
        Per-sample transform applied before the standardisation and the tokenizer
        (skeleton centred on its root joint, rotated into the body frame...).
    standardize : {None, "last", "element"}
        Divide by standard deviations fitted on the training samples of the split (after
        removing the means): ``"last"`` one per variable of the last axis, ``"element"`` one
        per element of the trailing axes (per joint and coordinate).
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
    prepare: Callable | None = None
    standardize: str | None = None
    drop_invalid: bool = False
    min_n_over_p: float = 2.0

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")
        if self.standardize not in (None, "last", "element"):
            raise ValueError(f"standardize must be None, 'last' or 'element', got "
                             f"{self.standardize!r}")

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
        if self.prepare is not None:
            x = self.prepare(x)
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


# ---------------------------------------------------------------------------
# Skeletons (NTU RGB+D 120; HDM05 has no raw trajectories): sample = (T, 25, 3) camera
# coordinates of body 0. Parameters of real/series_ntu.py and real/series_large_p.py.

from ..real.series_large_p import NTU_JOINTS_15  # noqa: E402
from ..real.series_ntu import (  # noqa: E402
    NTU_JOINTS_16,
    NTU_LONG_ACTIONS,
    NTU_PARTS,
    body_frame,
)

NTU_PARTS_15 = {"trunk": (0, 1, 20, 2, 3), "left arm": (4, 5, 6, 7, 21),
                "right arm": (8, 9, 10, 11, 23), "left leg": (0, 12, 13, 14, 15),
                "right leg": (0, 16, 17, 18, 19)}
"""Five body parts of five joints (p = 15 each; the root joint is shared by the trunk and the
legs, the thumbs are left out), camera coordinates: tokens of the ``part_temporal`` mode."""


def _root_centred(x):
    """(T, 25, 3) -> (T, 24, 3): root joint subtracted, then dropped (``ntu_skeletons``)."""
    return (x - x[:, :1])[:, 1:]


def _root_centred_25(x):
    return x - x[:, :1]


def _body_frame_16(x):
    return body_frame(x - x[:, :1])[:, list(NTU_JOINTS_16)]


def _centred_16(x):
    return (x - x[:, :1])[:, list(NTU_JOINTS_16)]


def _joints(x, *, joints=None):
    """One token per joint: its trajectory over every frame, p = 3."""
    joints = list(joints) if joints is not None else range(x.shape[1])
    return ([x[:, j] for j in joints],
            [[(0, len(x)), (j, j + 1), (0, x.shape[2])] for j in joints])


def _frames(x, *, n_windows=12):
    """``n_windows`` equal consecutive windows; observations = every joint of every frame."""
    edges = np.linspace(0, len(x), n_windows + 1).round().astype(int)
    obs = [x[a:b].reshape(-1, x.shape[-1]) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    return obs, [[(a, b), (0, x.shape[1]), (0, 3)]
                 for a, b in zip(edges[:-1], edges[1:], strict=True)]


def _width_windows(T, width, n_windows, minimum=4):
    w = max(int(round(T * width)), minimum)
    return tok.axis_windows(T, w, n_windows=n_windows)


def _parts_in_windows(x, *, parts, width=1 / 3, n_windows=6):
    """Window-major: every body part of the first window, then of the second..."""
    obs, pos = [], []
    for a, b in _width_windows(len(x), width, n_windows):
        w = x[a:b]
        for joints in parts.values():
            obs.append(w[:, list(joints)].reshape(b - a, -1))
            pos.append([(a, b), list(joints), (0, 3)])  # list: a set of joints
    return obs, pos


def _joint_covariance(x, *, width=1 / 3, n_windows=6):
    """Covariance between joints over a window: the 3 axes are 3 observations of the
    J-dimensional variable, window mean removed per joint and axis (n = 3 x frames)."""
    obs, pos = [], []
    for a, b in _width_windows(len(x), width, n_windows):
        v = x[a:b] - x[a:b].mean(0)
        obs.append(v.transpose(0, 2, 1).reshape(-1, x.shape[1]))
        pos.append([(a, b), (0, x.shape[1]), (0, 3)])
    return obs, pos


def _axis_covariance(x, *, width=1 / 3, n_windows=4):
    """Per window and per body axis (lateral, vertical, sagittal): covariance between joints."""
    obs, pos = [], []
    w = int(len(x) * width)
    for a, b in tok.axis_windows(len(x), max(w, 1), n_windows=n_windows):
        for axis in range(3):
            obs.append(x[a:b, :, axis])
            pos.append([(a, b), (0, x.shape[1]), (axis, axis + 1)])
    return obs, pos


def _windows_all_joints(x, *, joints=None, n_windows=8):
    """R23 ``ntu_window_series``: windows of a quarter of the sequence, evenly spread, over
    the 3 x J coordinates (camera axes, not centred)."""
    J = list(joints) if joints is not None else (0, x.shape[1])
    x = x if joints is None else x[:, list(joints)]
    obs, pos = tok.temporal(x.reshape(len(x), -1), max(len(x) // 4, 3), n_windows=n_windows)
    return obs, [[p[0], J, (0, 3)] for p in pos]


def _skeleton_modes(shape, *, ntu=True):
    modes = {
        "joint": SeriesMode(
            "joint", _joints, ("T", "J", "C"), {"joints": None},
            "one token per joint (root-centred, root dropped): its 3 coordinates over every "
            "frame", shape, prepare=_root_centred),
        "frame": SeriesMode(
            "frame", _frames, ("T", "J", "C"), {"n_windows": 12},
            "12 equal consecutive windows; observations = every joint of every frame, p = 3",
            shape, family="temporal", prepare=_root_centred),
        "window": SeriesMode(
            "window", _windows_all_joints, ("T", "J", "C"),
            {"joints": None, "n_windows": 8},
            f"R23 large p: 8 windows of a quarter of the sequence, the {3 * shape[1]} camera "
            "coordinates (n_t < 2p, Ledoit-Wolf)", shape, family="temporal",
            estimator="ledoit_wolf", allow_small_n=True),
    }
    if not ntu:
        return modes
    modes.update({
        "part": SeriesMode(
            "part", _parts_in_windows, ("T", "J", "C"),
            {"parts": NTU_PARTS, "width": 1 / 3, "n_windows": 6},
            "6 windows of a third of the sequence x 5 body parts of 4 joints (p = 12), "
            "root-centred and standardised coordinates, window-major", shape,
            family="temporal", estimator="ledoit_wolf", prepare=_root_centred_25,
            standardize="element"),
        "jointcov": SeriesMode(
            "jointcov", _joint_covariance, ("T", "J", "C"),
            {"width": 1 / 3, "n_windows": 6},
            "6 windows of a third of the sequence; covariance between 16 joints, the 3 axes as "
            "observations (n = 3 x frames)", shape, family="temporal",
            estimator="ledoit_wolf", prepare=_centred_16, standardize="element"),
        "axis": SeriesMode(
            "axis", _axis_covariance, ("T", "J", "C"),
            {"width": 1 / 3, "n_windows": 4},
            "4 windows x 3 body axes (lateral, vertical, sagittal): covariance between 16 "
            "joints in the body frame, window-major", shape, family="temporal",
            estimator="ledoit_wolf", prepare=_body_frame_16, standardize="element"),
        "part_temporal": SeriesMode(
            "part_temporal", _parts_in_windows, ("T", "J", "C"),
            {"parts": NTU_PARTS_15, "width": 0.4, "n_windows": 4},
            "4 windows (50 % overlap) x 5 body parts of 5 joints (p = 15), camera "
            "coordinates, window-major; a token with n_t < 2p is masked", shape,
            family="temporal"),
        "window_p45": SeriesMode(
            "window_p45", _windows_all_joints, ("T", "J", "C"),
            {"joints": NTU_JOINTS_15, "n_windows": 8},
            "R23 large p: 8 windows of a quarter of the sequence, 15 joints (p = 45, n_t < 2p, "
            "Ledoit-Wolf)", shape, family="temporal", estimator="ledoit_wolf",
            allow_small_n=True),
    })
    window = modes.pop("window")
    modes["window_p75"] = replace(window, name="window_p75")
    modes["window_p45"] = modes.pop("window_p45")
    return modes


# Sequences of real/series_ntu.py: actions A001-A010, at most 300 per class (median 76
# frames); the axis mode keeps the 11 long actions (median 123 frames), as ntu_axis_series.
_NTU_SELECTION = {"classes": [f"A{a:03d}" for a in range(1, 11)], "max_samples_per_class": 300}
_NTU_LONG = {"classes": [f"A{a:03d}" for a in NTU_LONG_ACTIONS], "max_samples_per_class": None}
SERIES_MODES["ntu120"] = {
    name: replace(mode, loader_defaults=_NTU_LONG if name == "axis" else _NTU_SELECTION,
                  example_shape=(123, 25, 3) if name == "axis" else mode.example_shape)
    for name, mode in _skeleton_modes((76, 25, 3)).items()}
# HDM05: same generic modes on (T, 31, 3); the copies hold no trajectories (structured_raw raises)
SERIES_MODES["hdm05"] = _skeleton_modes((251, 31, 3), ntu=False)

# ---------------------------------------------------------------------------
# PASTIS-R parcels: (dates or periods, pixels, variables)


def _pastis_s2(ds, i):
    return ds.s2_dates(i, 24)


def _pastis_fusion(ds, i):
    return ds.s1s2_periods(i, 12)


SERIES_MODES["pastis_r"] = {
    "s2_dates": SeriesMode(
        "s2_dates", _temporal, ("T", "N", "B"),
        {"window": 1, "step": 1, "n_windows": None, "n_channels": None, "cover": False},
        "one token per date of a 24-date calendar grid (nearest S2 acquisition of the tile): "
        "covariance of the 10 bands over the parcel pixels", (24, 66, 10), family="temporal",
        estimator="ledoit_wolf", source=_pastis_s2),
    "s2_months": SeriesMode(
        "s2_months", _temporal, ("T", "N", "B"),
        {"window": 2, "step": 2, "n_windows": None, "n_channels": None, "cover": False},
        "12 periods of two consecutive grid dates (about a month): 10 bands over the pixels x "
        "2 dates", (24, 66, 10), family="temporal", estimator="ledoit_wolf",
        source=_pastis_s2),
    "s1s2_fusion": SeriesMode(
        "s1s2_fusion", _temporal, ("T", "N", "B"),
        {"window": 1, "step": 1, "n_windows": None, "n_channels": None, "cover": False},
        "12 equal periods, per-pixel median of the 10 S2 bands and (VV, VH) of S1A and S1D "
        "(p = 14), standardised on the training parcels", (12, 66, 14), family="temporal",
        estimator="ledoit_wolf", source=_pastis_fusion, standardize="last"),
}

# ---------------------------------------------------------------------------
# Objects (rice grains, fruits): bounding box (h, w, B), 0 outside the object

SERIES_MODES["rices90"] = {
    "spectral": SeriesMode(
        "spectral", _spectral, ("H", "W", "B"),
        {"band_window": 16, "band_step": 8, "n_windows": None, "cover": True},
        "window of 16 binned bands (step 8); observations = the grain pixels", (21, 17, 64),
        drop_invalid=True),
    "spatial": SeriesMode(
        "spatial", _spatial, ("H", "W", "B"),
        {"window": (0.5, 0.5), "step": (0.5, 0.5), "n_windows": None, "n_channels": 16},
        "4 quadrants of the grain box, the 64 bands pooled into 16 (p = 64 gives about 100 "
        "pixels for 2p = 128)", (21, 17, 64), drop_invalid=True),
    "spatio_spectral": SeriesMode(
        "spatio_spectral", _spatio_spectral, ("H", "W", "B"),
        {"window": (0.5, 0.5), "step": (0.5, 0.5), "n_windows": None, "band_window": 16,
         "band_step": 16, "band_n_windows": None},
        "4 quadrants of the grain box x windows of 16 bands, spatial-major", (21, 17, 64),
        drop_invalid=True),
}
SERIES_MODES["deephsfruit"] = {
    "spectral": SeriesMode(
        "spectral", _spectral, ("H", "W", "B"),
        {"band_window": 16, "band_step": 8, "n_windows": None, "cover": True},
        "window of 16 bands (step 8, T = 27); observations = the fruit pixels",
        (180, 140, 224), drop_invalid=True,
        loader_defaults={"classification_mode": "ripeness"}),
    "spatial": SeriesMode(
        "spatial", _spatial, ("H", "W", "B"),
        {"window": (0.5, 0.5), "step": (0.5, 0.5), "n_windows": None, "n_channels": 16},
        "4 quadrants of the fruit box, the 224 bands pooled into 16", (180, 140, 224),
        drop_invalid=True, loader_defaults={"classification_mode": "ripeness"}),
    "spatio_spectral": SeriesMode(
        "spatio_spectral", _spatio_spectral, ("H", "W", "B"),
        {"window": (0.5, 0.5), "step": (0.5, 0.5), "n_windows": None, "band_window": 32,
         "band_step": 32, "band_n_windows": None},
        "4 quadrants of the fruit box x windows of 32 bands, spatial-major", (180, 140, 224),
        drop_invalid=True, loader_defaults={"classification_mode": "ripeness"}),
}

# ---------------------------------------------------------------------------
# Pol-InSAR-Island (real.series_sar.polinsar_logpower_series): 4 x 4 patches of 8 x 8 m of
# a 32 x 32 cell, the 12 log-powers (6 per band, L and S), standardised, Ledoit-Wolf

SERIES_MODES["pol_insar_island"] = {
    "spatial": SeriesMode(
        "spatial", _spatial, ("H", "W", "F"),
        {"window": (8, 8), "step": (8, 8), "n_windows": None, "n_channels": None},
        "4 x 4 patches of 8 x 8 pixels, row-major; variables = the 12 log-powers (L and S)",
        (32, 32, 12), estimator="ledoit_wolf", standardize="last"),
}


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
