"""
Covariance series from real datasets (strategies of the data atlas, chapter "Séries de covariances").

Output contract of every ``*_series`` builder (``real/series*.py``)::

    data, labels, n_obs, groups_or_splits[, mean]

- ``data``: float64 (N, T, p, p), one series of T SPD tokens per sample. Missing tokens (padding, masked dates
  or windows) are identity matrices (convention E8 of the atlas).
- ``labels``: (N,) integer classes, or float targets for a regression.
- ``n_obs``: (N,) number of observations behind the covariances of each sample (the smallest or the median over
  its tokens when they differ; documented per builder), so that ``n >= 2p`` can be audited.
- ``groups_or_splits``: (N,) group ids for a grouped split (subject, session, spatial block, ...), or a dict
  with ``"split"`` (0 train / 1 validation / 2 test) and other per-sample arrays.
- ``mean`` (only with ``return_mean=True``, where available): float64 (N, T, p), the per-channel mean of the
  observations of each token.

Estimation goes through :func:`covariance` (:func:`spdnet_datasets.core.estimate_covariance`, centred, then a
relative diagonal loading ``load * tr(C) / p * I``, 1e-6 everywhere unless stated). ``n >= 2p`` is enforced
(:class:`spdnet_datasets.core.InsufficientSamplesError`); the few builders where it does not hold by design
take ``allow_small_n=False`` and must be called with ``allow_small_n=True`` (documented derogation). Data roots
default to ``<SPDNET_DATA_ROOT>/<DOMAIN>/<Name>`` (:func:`spdnet_datasets.core.dataset_path`); every builder
takes ``root`` to override it. Statistics used for normalising the data are fitted on the training samples
only (``fit_mask`` or the internal split, see each builder).

- :func:`sliding_covariances`: generic sliding window over the observation axis.
- :func:`fmcw_series`: FMCW 24 GHz (figshare, simulated activities). Coherent integration of the 102 rows,
  STFT (128 / 32), log power pooled into p frequency bands, covariances over T windows of frames spread
  evenly over the whole recording. The recordings last 8 000, 10 000 or 15 000 samples depending on the
  class: a fixed step would give a class-dependent number of tokens (a trivial leak), hence a fixed T.
- :func:`hyperleaf_series`: HyperLeaf2024 cubes, spectral windows of b bands (step s), observations = leaf
  pixels (non-zero), one b x b covariance per window: a series ordered by wavelength. Optional ``zones=(r, c)``
  (R39): the leaf pixels are cut into r x c spatial zones at the quantiles of their row and column indices
  (median for 2), each zone gives its own series of windows, concatenated zone by zone (row-major): T = r c T_spectral,
  and the pixel count returned is the smallest zone. A zone whose window is (almost) constant (saturated sensor,
  pixels at 65535 in the far-NIR bands: trace < 1 % of the whole-leaf window, 23 tokens of 10 cubes in HyperLeaf,
  43 of them exactly zero) takes the covariance of the whole-leaf window instead: the zero matrix is not SPD and a
  token 10^-3 to 10^-8 times smaller than the others makes the eigendecompositions of the models fail.
  Default (1, 1): unchanged.
- :func:`deephs_series`: DeepHS-Fruit, VIS camera (224 bands, reflectance), ripeness in 3 classes; same
  spectral windows, observations = fruit pixels (Otsu threshold on the 760-880 nm mean). Also returns the
  physical fruit of each cube (front and back views share it) for a split without leakage.
- :func:`ci4r_series`: CI4R-MULTI3, 77 GHz micro-Doppler spectrograms rendered in jet colours: the colour map
  is inverted (the luminance of jet is not monotonic), the central half of the Doppler axis is pooled into p
  bands, and Ledoit-Wolf covariances are taken over windows of columns (time): a series ordered in time.
  Also returns the recording session (file prefix, ~20 takes of one activity by one person) for the split.
- :func:`pastis_series`: PASTIS-R, one series per agricultural parcel: covariance of the 10 Sentinel-2 bands
  over the parcel pixels at each date of a common calendar grid (nearest acquisition of the tile), with the
  official folds (1-3 train, 4 validation, 5 test) and at most a fixed number of parcels per class and fold.
- :func:`dopnet_series`: Dop-NET hand gestures (4 cells of the training files, persons A-F), log power of the
  central Doppler bins pooled into p bands, covariances over windows of frames: variable lengths, padded with
  the identity (E8). Also returns the person of each gesture for a split across persons.
- :func:`uav_series`: UAV-HSI-Crop, pre-extracted 15 x 15 windows of one scene (100 bands): spectral windows of
  b bands over the 225 pixels, as for HyperLeaf. Windows overlap on the source image, so the split unit is a
  spatial block (block x class), and windows not entirely inside their block are dropped (no shared pixel
  between blocks); classes with too few windows are dropped and the large ones capped. Ledoit-Wolf: with the
  empirical estimator, 5 % of the tokens sit at the floor of the diagonal loading (kappa ~ 10^7.2).
- :func:`kaggle_wheat_series`: Kaggle wheat (Health / Other / Rust), 32 x 32 hyperspectral patches: spectral
  windows of b bands over the 1 024 pixels of the first 125 bands; the 23 all-zero patches are dropped.
  Ledoit-Wolf, for the same reason (0.7 % of the empirical tokens at the loading floor).
"""

import glob
import os
import warnings

import numpy as np

from ..core.config import dataset_path
from ..core.estimators import check_sample_size, check_spd, estimate_covariance

FMCW_CLASSES = ["B", "K", "P", "G", "W", "SU", "SD", "STW", "WTS", "WTF", "FTW"]
HYPERLEAF_CULTIVARS = ["Heerup", "Kvium", "Rembrandt", "Sheriff"]
DEEPHS_RIPENESS = ["unripe", "perfect", "overripe"]  # "ripe" is merged into "perfect", as in deephsfruit.py
DOPNET_CELLS = ["geste 0", "geste 1", "geste 2", "geste 3"]  # class names of the cells are ambiguous (atlas)
PASTIS_CLASSES = ["Meadow", "Soft winter wheat", "Corn", "Winter barley", "Winter rapeseed", "Spring barley",
                  "Sunflower", "Grapevine", "Beet", "Winter triticale", "Winter durum wheat",
                  "Fruits, vegetables, flowers", "Potatoes", "Leguminous fodder", "Soybeans", "Orchard",
                  "Mixed cereal", "Sorghum"]  # semantic classes 1-18 (0 background and 19 void are dropped)
KAGGLE_WHEAT_CLASSES = ["Health", "Other", "Rust"]
CI4R_CLASSES = ["Away", "Bend", "Crawl", "Kneel", "Limp", "Pick", "SStep", "Scissor", "Sit", "Toes", "Towards"]


def data_root(root, domain: str, name: str, *parts: str) -> str:
    """``root`` if given, else ``<SPDNET_DATA_ROOT>/<domain>/<name>/<parts...>`` (resolved at call time)."""
    if root is not None:
        return str(root)
    return str(dataset_path(domain, name).joinpath(*parts))


def covariance(
    x: np.ndarray,
    estimator: str = "ledoit_wolf",
    load: float = 1e-6,
    allow_small_n: bool = False,
    spd: bool = True,
) -> np.ndarray:
    """
    One (p, p) token from observations ``x`` (n, p): :func:`spdnet_datasets.core.estimate_covariance` (centred,
    ``estimator`` in scm / ledoit_wolf / tyler / student), plus the relative diagonal loading
    ``load * tr(C) / p * I``.

    ``n >= 2p`` is checked first unless ``allow_small_n`` (documented derogations only). ``spd``: check the
    loaded token is SPD (:class:`spdnet_datasets.core.NotSPDError`); callers that replace degenerate tokens
    themselves (constant windows -> identity or fallback) pass ``spd=False``.
    """
    x = np.asarray(x, dtype=np.float64)
    n, p = x.shape
    if not allow_small_n:
        check_sample_size(n, p)
    c = estimate_covariance(x, estimator, check=False)
    c = c + load * np.trace(c) / p * np.eye(p)
    if spd:
        check_spd(c)
    return c


def resolve_fit_mask(fit_mask, n: int, *args) -> np.ndarray:
    """Boolean (n,) mask of the samples that fit the normalisation statistics. ``fit_mask``: None (every
    sample, with a warning: test statistics leak into the normalisation), a boolean / index array, or a
    callable applied to ``args`` (e.g. labels and groups) that returns one."""
    if fit_mask is None:
        warnings.warn("normalisation statistics fitted on all the samples, test included: pass fit_mask "
                      "(training samples) to avoid the leak", UserWarning, stacklevel=3)
        return np.ones(n, bool)
    if callable(fit_mask):
        fit_mask = fit_mask(*args)
    fit_mask = np.asarray(fit_mask)
    if fit_mask.dtype != bool:
        mask = np.zeros(n, bool)
        mask[fit_mask] = True
        fit_mask = mask
    if fit_mask.shape != (n,) or not fit_mask.any():
        raise ValueError(f"fit_mask must select at least one of the {n} samples")
    return fit_mask


def window_starts(
    n: int,
    window: int,
    step: int | None = None,
    n_windows: int | None = None,
    cover: bool = False,
) -> list[int]:
    """
    Starts of the windows ``[s, s + window)`` over ``n`` positions: every ``step`` (``range(0, n - window + 1,
    step)``), or ``n_windows`` starts spread evenly over ``[0, n - window]``. ``cover`` (``step`` only): add a
    last window ending at ``n`` when the regular starts leave the end uncovered. Empty when ``n < window``.
    """
    if n < window:
        return []
    if n_windows is not None:
        return np.linspace(0, n - window, n_windows).round().astype(int).tolist()
    starts = list(range(0, n - window + 1, step))
    if cover and starts[-1] + window < n:
        starts.append(n - window)
    return starts


def sliding_covariances(
    observations: np.ndarray,
    window: int,
    step: int | None,
    estimator: str = "scm",
    load: float = 1e-6,
    n_windows: int | None = None,
    return_mean: bool = False,
    spd: bool = True,
):
    """
    observations (n, p) ordered along n -> covariances (T, p, p) of the windows [s, s + window): starts every
    ``step`` (T = floor((n - window) / step) + 1), or ``n_windows`` starts spread evenly over [0, n - window].
    Each covariance is diagonally loaded by load * tr(C) / p (:func:`covariance`). ``return_mean``: also the
    (T, p) window means. Raises ``ValueError`` when ``n < window`` (no complete window).
    """
    n, p = observations.shape
    if n < window:
        raise ValueError(f"{n} observations for windows of {window}: no complete window")
    starts = window_starts(n, window, step, n_windows)
    covariances = np.stack([covariance(observations[s:s + window], estimator, load, spd=spd) for s in starts])
    if return_mean:
        return covariances, np.stack([observations[s:s + window].mean(0) for s in starts])
    return covariances


def fmcw_series(
    root: str | None = None,
    nperseg: int = 128,
    hop: int = 32,
    n_bands: int = 16,
    window: int = 64,
    n_windows: int = 8,
    files_per_class: int | None = None,
    return_mean: bool = False,
) -> tuple:
    """(n_files, T, n_bands, n_bands) SCM covariances of log-power spectrogram bands, labels, observations per
    covariance (``window``); with ``return_mean``, also the (n_files, T, n_bands) window means."""
    import scipy.io as sio
    from scipy.signal import stft

    root = data_root(root, "RADAR_DOPPLER", "FMCW24GHz-figshare")
    data, labels, means = [], [], []
    for label, name in enumerate(FMCW_CLASSES):
        files = sorted(glob.glob(os.path.join(root, name, "*.mat")))[:files_per_class]
        for path in files:
            signal = sio.loadmat(path)["received_time_domain_signal"].mean(0)  # coherent integration -> (8000,)
            _, _, spectrum = stft(signal, nperseg=nperseg, noverlap=nperseg - hop, return_onesided=False, boundary=None)
            power = np.abs(np.fft.fftshift(spectrum, 0)) ** 2  # (nperseg, frames)
            bands = power.reshape(n_bands, nperseg // n_bands, -1).mean(1)  # (n_bands, frames)
            features = np.log(bands + 1e-12 * bands.max()).T  # (frames, n_bands)
            covariances, mean = sliding_covariances(features, window, None, "scm", n_windows=n_windows,
                                                    return_mean=True)
            data.append(covariances)
            means.append(mean)
            labels.append(label)
    out = (np.stack(data), np.array(labels), window)
    return (*out, np.stack(means)) if return_mean else out


def hyperleaf_series(
    root: str | None = None,
    band_window: int = 16,
    band_step: int = 8,
    max_images: int | None = None,
    zones: tuple[int, int] = (1, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n_images, T, b, b) spectral-window covariances of the leaf pixels, cultivar labels, pixels per image
    (per image: the smallest zone when ``zones`` != (1, 1))."""
    import pandas as pd
    import tifffile

    root = data_root(root, "HYPERSPECTRAL", "HyperLeaf2024")
    table = pd.read_csv(os.path.join(root, "train.csv"), dtype={"ImageId": str})[:max_images]
    data, labels, n_pixels = [], [], []
    for row in table.itertuples():
        cube = tifffile.imread(os.path.join(root, "images", f"{row.ImageId}.tiff")).astype(np.float64) / 65535.0
        pixels = cube.reshape(cube.shape[0], -1).T  # (pixels, 204 bands)
        leaf = pixels.sum(1) > 0  # leaf mask: 91 % of the pixels are background zeros
        if zones == (1, 1):
            groups = [pixels[leaf]]
        else:
            rows, cols = np.divmod(np.nonzero(leaf)[0], cube.shape[2])  # row-major flattening of (H, W)
            zone = np.zeros(len(rows), dtype=int)
            for n_cuts, index in ((zones[0], rows), (zones[1], cols)):
                cuts = np.quantile(index, np.arange(1, n_cuts) / n_cuts)
                zone = zone * n_cuts + (index[:, None] > cuts).sum(1)  # ties go to the lower zone
            groups = [pixels[leaf][zone == z] for z in range(zones[0] * zones[1])]
        # loaded tokens: the trace test below compares loaded traces, the same ratio as unloaded ones
        covariances = np.concatenate([spectral_windows(group, band_window, band_step, "scm", spd=False)
                                      for group in groups])
        if zones != (1, 1):  # (almost) constant zone window: whole-leaf window instead
            whole = np.tile(spectral_windows(pixels[leaf], band_window, band_step, "scm"), (len(groups), 1, 1))
            trace = np.trace(covariances, axis1=-2, axis2=-1)[:, None, None]
            constant = trace < 1e-2 * np.trace(whole, axis1=-2, axis2=-1)[:, None, None]
            covariances = np.where(constant, whole, covariances)
        check_spd(covariances)
        data.append(covariances)
        labels.append(int(np.argmax([getattr(row, c) for c in HYPERLEAF_CULTIVARS])))
        n_pixels.append(min(len(group) for group in groups))
    return np.stack(data), np.array(labels), np.array(n_pixels)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Threshold maximising the between-class variance of the histogram (Otsu, 1979): the low class is the
    bins up to k, so the threshold is the upper edge of bin k (values > threshold form the high class)."""
    hist, edges = np.histogram(values, bins)
    centres = (edges[:-1] + edges[1:]) / 2
    weight, moment = np.cumsum(hist), np.cumsum(hist * centres)
    between = (moment[-1] * weight / weight[-1] - moment) ** 2 / np.maximum(weight * (weight[-1] - weight), 1)
    return float(edges[np.argmax(between) + 1])


def deephs_series(
    root: str | None = None,
    band_window: int = 16,
    band_step: int = 8,
    max_images: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(n_cubes, T, b, b) spectral-window covariances of the fruit pixels, ripeness labels, pixels per cube,
    physical fruit id per cube (fruit, day, number)."""
    import json

    from spectral.io import envi

    root = data_root(root, "HYPERSPECTRAL", "DeepHS-Fruit")
    with open(os.path.join(root, "annotations", "merged_annotations.json")) as f:
        annotations = json.load(f)
    ripeness = {a["record_id"]: a["ripeness_state"] for a in annotations["annotations"]}
    ripeness = {k: "perfect" if v == "ripe" else v for k, v in ripeness.items()}
    records = [r for r in annotations["records"]
               if r["camera_type"] == "VIS" and ripeness.get(r["id"]) in DEEPHS_RIPENESS][:max_images]
    data, labels, n_pixels, fruits = [], [], [], []
    for record in records:
        header = os.path.join(root, record["files"]["header_file"])
        cube = np.asarray(envi.open(header, header.replace(".hdr", ".bin")).load(), dtype=np.float64)
        nir = cube[..., 150:200].mean(-1)
        pixels = cube[nir > otsu_threshold(nir)]  # (fruit pixels, 224 bands)
        data.append(spectral_windows(pixels, band_window, band_step, "scm"))
        labels.append(DEEPHS_RIPENESS.index(ripeness[record["id"]]))
        n_pixels.append(len(pixels))
        fruits.append(os.path.basename(header).rsplit("_", 1)[0])  # e.g. avocado_day_01_01
    _, groups = np.unique(fruits, return_inverse=True)
    return np.stack(data), np.array(labels), np.array(n_pixels), groups


def jet_to_scalar(rgb: np.ndarray) -> np.ndarray:
    """Invert a jet colour map: (h, w, 3) uint8 RGB -> (h, w) values in [0, 1] (nearest jet colour)."""
    from matplotlib import colormaps
    from scipy.spatial import cKDTree

    table = colormaps["jet"](np.linspace(0, 1, 256))[:, :3] * 255
    _, index = cKDTree(table).query(rgb.reshape(-1, 3).astype(np.float64))
    return (index / 255.0).reshape(rgb.shape[:2])


def ci4r_series(
    root: str | None = None,
    n_bands: int = 16,
    window: int = 200,
    step: int = 100,
    max_images: int | None = None,
    pooled_cov: bool = False,
    return_mean: bool = False,
) -> tuple:
    """(n_images, T, p, p) covariances of the 77 GHz spectrograms over time windows, activity labels, columns
    per covariance (n_images,), recording session per image. ``pooled_cov``: one covariance over all the columns,
    (n_images, 1, p, p) (baseline B3, no series); ``n_obs`` is then the number of columns of each image.
    Every column is kept (no silent removal of the all-zero ones); a constant window (null covariance) becomes an
    identity token (masked, E8), with a warning giving their count. ``return_mean``: also the (n_images, T, p)
    window means."""
    from PIL import Image

    root = data_root(root, "RADAR_DOPPLER", "CI4R-MULTI3")
    files = sorted(f for f in glob.glob(os.path.join(root, "11_class_activity_data", "77GHz", "*", "*.png"))
                   if "(copy)" not in f)[:max_images]  # the "(copy)" files duplicate other takes
    data, means, labels, sessions, n_obs = [], [], [], [], []
    n_constant = 0
    for path in files:
        image = jet_to_scalar(np.asarray(Image.open(path).convert("RGB")))  # (Doppler 804, time 1475)
        height = image.shape[0]
        band = image[height // 4: height // 4 + (height // 2) // n_bands * n_bands]  # central half of the Doppler axis
        pooled = band.reshape(n_bands, -1, band.shape[1]).mean(1).T  # (time, p)
        width = len(pooled) if pooled_cov else window
        covariances, mean = sliding_covariances(pooled, width, step, "ledoit_wolf", return_mean=True, spd=False)
        constant = np.trace(covariances, axis1=-2, axis2=-1) <= 0
        covariances[constant] = np.eye(n_bands)
        check_spd(covariances)
        n_constant += int(constant.sum())
        data.append(covariances)
        means.append(mean)
        n_obs.append(width)
        labels.append(CI4R_CLASSES.index(os.path.basename(os.path.dirname(path))))
        sessions.append(os.path.basename(path).split("_")[0])
    if n_constant:
        warnings.warn(f"CI4R: {n_constant} constant windows replaced by identity tokens", UserWarning, stacklevel=2)
    _, groups = np.unique(sessions, return_inverse=True)
    out = (np.stack(data), np.array(labels), np.array(n_obs), groups)
    return (*out, np.stack(means)) if return_mean else out


def pastis_series(
    root: str | None = None,
    n_dates: int = 24,
    min_pixels: int = 30,
    per_class_and_fold: int = 100,
    seed: int = 0,
    pooled: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(n_parcels, n_dates, 10, 10) covariances of the Sentinel-2 bands over each parcel, at the acquisitions
    nearest an evenly spaced calendar grid; labels in 0..17; pixels per parcel; {"split": 0 train / 1 val / 2 test}.
    ``pooled``: one covariance over the pixels of all the dates, (n_parcels, 1, 10, 10) (baseline B3, no series)."""
    import ast
    import datetime
    import json

    root = data_root(root, "SITS", "PASTIS-R", "PASTIS-R")
    with open(os.path.join(root, "metadata.geojson")) as f:
        patches = [x["properties"] for x in json.load(f)["features"]]

    def days(date: int) -> int:
        return datetime.date(date // 10000, date // 100 % 100, date % 100).toordinal()

    def acquisitions(value) -> dict:  # a dict in the GeoJSON, or its string form
        return value if isinstance(value, dict) else ast.literal_eval(value)

    dates = {p["ID_PATCH"]: np.array([days(d) for d in acquisitions(p["dates-S2"]).values()]) for p in patches}
    start, end = max(d.min() for d in dates.values()), min(d.max() for d in dates.values())
    grid = np.linspace(start, end, n_dates)

    candidates = []  # (patch, parcel id, label, fold, pixels)
    for patch in patches:
        pid = patch["ID_PATCH"]
        target = np.load(os.path.join(root, "ANNOTATIONS", f"TARGET_{pid}.npy"))[0].ravel()
        parcels = np.load(os.path.join(root, "ANNOTATIONS", f"ParcelIDs_{pid}.npy")).ravel()
        ids, inverse, counts = np.unique(parcels, return_inverse=True, return_counts=True)
        for k, (parcel, count) in enumerate(zip(ids, counts, strict=True)):
            label = np.bincount(target[inverse == k]).argmax()  # majority semantic class of the parcel
            if parcel != 0 and count >= min_pixels and label not in (0, 19):
                candidates.append((pid, parcel, label - 1, int(patch["Fold"]), count))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    kept, taken = [], {}
    for i in order:  # at most per_class_and_fold parcels of each class in each fold
        key = candidates[i][2:4]
        if taken.get(key, 0) < per_class_and_fold:
            taken[key] = taken.get(key, 0) + 1
            kept.append(candidates[i])
    kept.sort(key=lambda c: (c[0], c[1]))  # one read of each patch

    # Ledoit-Wolf: cloudy or saturated dates give nearly rank-one covariances (condition number ~1e7 even
    # after diagonal loading), which the per-position recentring turns into numerically singular tokens.
    data, labels, n_pixels, split = [], [], [], []
    current, s2, parcels = None, None, None
    for pid, parcel, label, fold, count in kept:
        if pid != current:
            current = pid
            nearest = np.abs(dates[pid][:, None] - grid[None, :]).argmin(0)  # acquisition nearest each grid date
            s2 = np.load(os.path.join(root, "DATA_S2", f"S2_{pid}.npy"))[nearest].astype(np.float64)  # (T, 10, 128, 128)
            parcels = np.load(os.path.join(root, "ANNOTATIONS", f"ParcelIDs_{pid}.npy"))
        pixels = s2[:, :, parcels == parcel]  # (T, 10, n)
        if pooled:
            covariances = covariance(pixels.transpose(0, 2, 1).reshape(-1, pixels.shape[1]))[None]
        else:
            covariances = np.stack([covariance(x.T) for x in pixels])
        data.append(covariances)
        labels.append(label)
        n_pixels.append(count)
        split.append(0 if fold <= 3 else fold - 3)
    return np.stack(data), np.array(labels), np.array(n_pixels), {"split": np.array(split)}


def dopnet_series(
    root: str | None = None,
    n_bands: int = 16,
    n_bins: int = 128,
    window: int = 48,
    step: int = 24,
    max_per_cell: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(n_gestures, T_max, p, p) covariances of the log Doppler power over windows of frames, identity after the
    last window; labels = cell index 0..3 of the training files (the class names of the cells are ambiguous,
    see the data atlas: only the index is used); frames per covariance (``min(window, gesture length)``);
    person index (A-F)."""
    from scipy.io import loadmat

    root = data_root(root, "RADAR_DOPPLER", "Dop-NET", "Training Data")
    series, labels, n_frames, persons = [], [], [], []
    for person, path in enumerate(sorted(glob.glob(os.path.join(root, "*Person_*.mat")))):
        cells = loadmat(path)["Data_Training"]["Doppler_Signals"][0, 0]
        for label in range(cells.shape[1]):
            for signal in cells[0, label].ravel()[:max_per_cell]:  # complex (800 Doppler bins, frames)
                centre = signal.shape[0] // 2
                power = np.abs(signal[centre - n_bins // 2: centre + n_bins // 2]) ** 2
                bands = np.log10(power.reshape(n_bands, -1, power.shape[1]).mean(1) + 1e-12)  # (p, frames)
                width = min(window, bands.shape[1])
                series.append(sliding_covariances(bands.T, width, step, "scm"))
                labels.append(label)
                n_frames.append(width)
                persons.append(person)
    t_max = max(len(x) for x in series)
    eye = np.eye(n_bands)
    data = np.stack([np.concatenate([x, np.broadcast_to(eye, (t_max - len(x), n_bands, n_bands))]) for x in series])
    return data, np.array(labels), np.array(n_frames), np.array(persons)


def spectral_windows(
    pixels: np.ndarray,
    band_window: int,
    band_step: int,
    estimator: str = "ledoit_wolf",
    return_mean: bool = False,
    spd: bool = True,
):
    """pixels (n, bands) -> (T, b, b) covariances of the spectral windows, diagonally loaded by 1e-6 tr(C) / b
    (:func:`covariance`); ``return_mean``: also the (T, b) window means."""
    starts = window_starts(pixels.shape[1], band_window, band_step)
    covariances = np.stack([covariance(pixels[:, b:b + band_window], estimator, spd=spd) for b in starts])
    if return_mean:
        return covariances, np.stack([pixels[:, b:b + band_window].mean(0) for b in starts])
    return covariances


def uav_series(
    root: str | None = None,
    band_window: int = 16,
    band_step: int = 8,
    block: int = 128,
    min_windows: int = 50,
    max_per_class: int = 300,
    seed: int = 0,
    return_mean: bool = False,
) -> tuple:
    """(n_windows, T, b, b) spectral-window covariances, crop labels (0 .. C-1), pixels per window, spatial groups;
    with ``return_mean``, also the (n_windows, T, b) window means. Default root:
    ``<SPDNET_DATA_ROOT>/HYPERSPECTRAL/UAV-HSI-Crop/XJM_W15_T80`` (on BERTHA the prepared windows only exist
    outside the canonical copy: pass ``root``)."""
    root = data_root(root, "HYPERSPECTRAL", "UAV-HSI-Crop", "XJM_W15_T80")
    images = np.load(os.path.join(root, "uav_windows_data.npy"), mmap_mode="r")  # (N, w, w, bands)
    crops = np.load(os.path.join(root, "uav_windows_labels.npy")).astype(int)
    centres = np.load(os.path.join(root, "uav_windows_positions.npy"))
    w = images.shape[1]
    # window inside its block whether positions are centres or top-left corners (the metadata does not say)
    inside = ((centres % block >= w // 2) & (centres % block <= block - w)).all(1)
    inside &= np.array([np.any(images[i]) for i in range(len(images))])  # all-zero windows (6 in MJK_N, none in XJM)
    kept_crops = [c for c in np.unique(crops) if (inside & (crops == c)).sum() >= min_windows]
    rng = np.random.default_rng(seed)
    index = np.concatenate([rng.permutation(np.nonzero(inside & (crops == c))[0])[:max_per_class] for c in kept_crops])
    index.sort()
    tokens = [spectral_windows(np.asarray(images[i], np.float64).reshape(-1, images.shape[-1]), band_window,
                               band_step, "ledoit_wolf", return_mean=True) for i in index]
    data = np.stack([c for c, _ in tokens])
    labels = np.searchsorted(kept_crops, crops[index])
    blocks = centres[index] // block
    groups = np.unique(np.stack([blocks[:, 0], blocks[:, 1], labels], 1), axis=0, return_inverse=True)[1].ravel()
    out = (data, labels, np.full(len(index), images.shape[1] * images.shape[2]), groups)
    return (*out, np.stack([m for _, m in tokens])) if return_mean else out


def kaggle_wheat_series(
    root: str | None = None,
    band_window: int = 16,
    band_step: int = 8,
    n_bands: int = 125,
    return_mean: bool = False,
) -> tuple:
    """(n_patches, T, b, b) spectral-window covariances, labels (Health / Other / Rust), pixels per patch; with
    ``return_mean``, also the (n_patches, T, b) window means."""
    import tifffile

    root = data_root(root, "HYPERSPECTRAL", "Wheat_Disease_ICPR26", "train", "HS")
    data, labels, means = [], [], []
    for path in sorted(glob.glob(os.path.join(root, "*.tif"))):
        pixels = tifffile.imread(path).astype(np.float64)[..., :n_bands].reshape(-1, n_bands) / 65535.0
        if not pixels.any():  # 23 patches are entirely zero (atlas)
            continue
        covariances, mean = spectral_windows(pixels, band_window, band_step, "ledoit_wolf", return_mean=True)
        data.append(covariances)
        means.append(mean)
        labels.append(KAGGLE_WHEAT_CLASSES.index(os.path.basename(path).split("_")[0]))
    out = (np.stack(data), np.array(labels), np.full(len(labels), 32 * 32))
    return (*out, np.stack(means)) if return_mean else out
