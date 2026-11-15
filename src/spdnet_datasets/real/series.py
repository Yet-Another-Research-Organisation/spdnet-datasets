"""
Covariance series from real datasets (strategies of the data atlas, /DATA/_atlas, chapter "Séries de
covariances"). Every builder returns float64 arrays (n_series, T, p, p), integer labels and the
number of observations per covariance, so that the constraint n >= 2p of the specification can be checked.

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

import numpy as np

from ..estimator.covariance import EstimateCovariance

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


def sliding_covariances(
    observations: np.ndarray,
    window: int,
    step: int | None,
    estimator: EstimateCovariance,
    load: float = 1e-6,
    n_windows: int | None = None,
) -> np.ndarray:
    """
    observations (n, p) ordered along n -> covariances (T, p, p) of the windows [s, s + window): starts every
    ``step`` (T = floor((n - window) / step) + 1), or ``n_windows`` starts spread evenly over [0, n - window].
    Each covariance is diagonally loaded by load * tr(C) / p (SPD guaranteed).
    """
    n, p = observations.shape
    if n_windows is not None:
        starts = np.linspace(0, n - window, n_windows).round().astype(int)
    else:
        starts = range(0, n - window + 1, step)
    covariances = np.stack([estimator(observations[s:s + window]) for s in starts])
    trace = np.trace(covariances, axis1=-2, axis2=-1)[:, None, None] / p
    return covariances + load * trace * np.eye(p)


def fmcw_series(
    root: str = "/DATA/FMCW24GHz-figshare",
    nperseg: int = 128,
    hop: int = 32,
    n_bands: int = 16,
    window: int = 64,
    n_windows: int = 8,
    files_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """(n_files, T, n_bands, n_bands) covariances of log-power spectrogram bands, labels, observations per covariance."""
    import scipy.io as sio
    from scipy.signal import stft

    estimator = EstimateCovariance("scm", remove_mean=True, exclude_zero=False)
    data, labels = [], []
    for label, name in enumerate(FMCW_CLASSES):
        files = sorted(glob.glob(os.path.join(root, name, "*.mat")))[:files_per_class]
        for path in files:
            signal = sio.loadmat(path)["received_time_domain_signal"].mean(0)  # coherent integration -> (8000,)
            _, _, spectrum = stft(signal, nperseg=nperseg, noverlap=nperseg - hop, return_onesided=False, boundary=None)
            power = np.abs(np.fft.fftshift(spectrum, 0)) ** 2  # (nperseg, frames)
            bands = power.reshape(n_bands, nperseg // n_bands, -1).mean(1)  # (n_bands, frames)
            features = np.log(bands + 1e-12 * bands.max()).T  # (frames, n_bands)
            data.append(sliding_covariances(features, window, None, estimator, n_windows=n_windows))
            labels.append(label)
    return np.stack(data), np.array(labels), window


def hyperleaf_series(
    root: str = "/DATA/HyperLeaf2024",
    band_window: int = 16,
    band_step: int = 8,
    max_images: int | None = None,
    zones: tuple[int, int] = (1, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n_images, T, b, b) spectral-window covariances of the leaf pixels, cultivar labels, pixels per image
    (per image: the smallest zone when ``zones`` != (1, 1))."""
    import pandas as pd
    import tifffile

    table = pd.read_csv(os.path.join(root, "train.csv"), dtype={"ImageId": str})[:max_images]
    estimator = EstimateCovariance("scm", remove_mean=True, exclude_zero=False)
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
        n_bands = pixels.shape[1]
        windows = [
            estimator(group[:, b:b + band_window]) for group in groups
            for b in range(0, n_bands - band_window + 1, band_step)
        ]
        covariances = np.stack(windows)
        trace = np.trace(covariances, axis1=-2, axis2=-1)[:, None, None] / band_window
        if zones != (1, 1):  # (almost) constant zone window: whole-leaf window instead
            whole = np.stack([estimator(pixels[leaf][:, b:b + band_window])
                              for b in range(0, n_bands - band_window + 1, band_step)])
            whole = np.tile(whole, (len(groups), 1, 1))
            constant = trace < 1e-2 * np.trace(whole, axis1=-2, axis2=-1)[:, None, None] / band_window
            covariances = np.where(constant, whole, covariances)
            trace = np.trace(covariances, axis1=-2, axis2=-1)[:, None, None] / band_window
        data.append(covariances + 1e-6 * trace * np.eye(band_window))
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
    root: str = "/DATA/DeepHS-Fruit",
    band_window: int = 16,
    band_step: int = 8,
    max_images: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(n_cubes, T, b, b) spectral-window covariances of the fruit pixels, ripeness labels, pixels per cube,
    physical fruit id per cube (fruit, day, number)."""
    import json

    from spectral.io import envi

    with open(os.path.join(root, "annotations", "merged_annotations.json")) as f:
        annotations = json.load(f)
    ripeness = {a["record_id"]: a["ripeness_state"] for a in annotations["annotations"]}
    ripeness = {k: "perfect" if v == "ripe" else v for k, v in ripeness.items()}
    records = [r for r in annotations["records"]
               if r["camera_type"] == "VIS" and ripeness.get(r["id"]) in DEEPHS_RIPENESS][:max_images]
    estimator = EstimateCovariance("scm", remove_mean=True, exclude_zero=False)
    data, labels, n_pixels, fruits = [], [], [], []
    for record in records:
        header = os.path.join(root, record["files"]["header_file"])
        cube = np.asarray(envi.open(header, header.replace(".hdr", ".bin")).load(), dtype=np.float64)
        nir = cube[..., 150:200].mean(-1)
        pixels = cube[nir > otsu_threshold(nir)]  # (fruit pixels, 224 bands)
        windows = [
            estimator(pixels[:, b:b + band_window]) for b in range(0, pixels.shape[1] - band_window + 1, band_step)
        ]
        covariances = np.stack(windows)
        trace = np.trace(covariances, axis1=-2, axis2=-1)[:, None, None] / band_window
        data.append(covariances + 1e-6 * trace * np.eye(band_window))
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
    root: str = "/DATA/CI4R-MULTI3",
    n_bands: int = 16,
    window: int = 200,
    step: int = 100,
    max_images: int | None = None,
    pooled_cov: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(n_images, T, p, p) covariances of the 77 GHz spectrograms over time windows, activity labels, columns
    per window, recording session per image. ``pooled_cov``: one covariance over all the columns, (n_images, 1, p, p)
    (baseline B3, no series)."""
    from PIL import Image

    files = sorted(f for f in glob.glob(os.path.join(root, "11_class_activity_data", "77GHz", "*", "*.png"))
                   if "(copy)" not in f)[:max_images]  # the "(copy)" files duplicate other takes
    estimator = EstimateCovariance("ledoit_wolf", remove_mean=True)
    data, labels, sessions = [], [], []
    for path in files:
        image = jet_to_scalar(np.asarray(Image.open(path).convert("RGB")))  # (Doppler 804, time 1475)
        height = image.shape[0]
        band = image[height // 4: height // 4 + (height // 2) // n_bands * n_bands]  # central half of the Doppler axis
        pooled = band.reshape(n_bands, -1, band.shape[1]).mean(1)  # (p, time)
        data.append(sliding_covariances(pooled.T, pooled.shape[1] if pooled_cov else window, step, estimator))
        labels.append(CI4R_CLASSES.index(os.path.basename(os.path.dirname(path))))
        sessions.append(os.path.basename(path).split("_")[0])
    _, groups = np.unique(sessions, return_inverse=True)
    return np.stack(data), np.array(labels), np.array(window), groups


def pastis_series(
    root: str = "/DATA/PASTIS-R/PASTIS-R",
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
        for k, (parcel, count) in enumerate(zip(ids, counts)):
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
    estimator = EstimateCovariance("ledoit_wolf", remove_mean=True, exclude_zero=False)
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
            covariances = estimator(pixels.transpose(0, 2, 1).reshape(-1, pixels.shape[1]))[None]
        else:
            covariances = np.stack([estimator(x.T) for x in pixels])
        trace = np.trace(covariances, axis1=-2, axis2=-1)[:, None, None] / covariances.shape[-1]
        data.append(covariances + 1e-6 * trace * np.eye(covariances.shape[-1]))
        labels.append(label)
        n_pixels.append(count)
        split.append(0 if fold <= 3 else fold - 3)
    return np.stack(data), np.array(labels), np.array(n_pixels), {"split": np.array(split)}


def dopnet_series(
    root: str = "/DATA/Dop-NET/Training Data",
    n_bands: int = 16,
    n_bins: int = 128,
    window: int = 48,
    step: int = 24,
    max_per_cell: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(n_gestures, T_max, p, p) covariances of the log Doppler power over windows of frames, identity after the
    last window; labels = cell index 0..3 of the training files (the class names of the cells are ambiguous,
    see the data atlas: only the index is used); frames per gesture; person index (A-F)."""
    from scipy.io import loadmat

    estimator = EstimateCovariance("scm", remove_mean=True, exclude_zero=False)
    series, labels, n_frames, persons = [], [], [], []
    for person, path in enumerate(sorted(glob.glob(os.path.join(root, "*Person_*.mat")))):
        cells = loadmat(path)["Data_Training"]["Doppler_Signals"][0, 0]
        for label in range(cells.shape[1]):
            for signal in cells[0, label].ravel()[:max_per_cell]:  # complex (800 Doppler bins, frames)
                centre = signal.shape[0] // 2
                power = np.abs(signal[centre - n_bins // 2: centre + n_bins // 2]) ** 2
                bands = np.log10(power.reshape(n_bands, -1, power.shape[1]).mean(1) + 1e-12)  # (p, frames)
                length = bands.shape[1]
                series.append(sliding_covariances(bands.T, min(window, length), step, estimator))
                labels.append(label)
                n_frames.append(length)
                persons.append(person)
    t_max = max(len(x) for x in series)
    eye = np.eye(n_bands)
    data = np.stack([np.concatenate([x, np.broadcast_to(eye, (t_max - len(x), n_bands, n_bands))]) for x in series])
    return data, np.array(labels), np.array(n_frames), np.array(persons)


def spectral_windows(pixels: np.ndarray, band_window: int, band_step: int, estimator: EstimateCovariance) -> np.ndarray:
    """pixels (n, bands) -> (T, b, b) covariances of the spectral windows, diagonally loaded by 1e-6 tr(C) / b."""
    windows = [estimator(pixels[:, b:b + band_window])
               for b in range(0, pixels.shape[1] - band_window + 1, band_step)]
    covariances = np.stack(windows)
    trace = np.trace(covariances, axis1=-2, axis2=-1)[:, None, None] / band_window
    return covariances + 1e-6 * trace * np.eye(band_window)


def uav_series(
    root: str = "/DATA/Hyperspectral/UAV-HSI-Crop/XJM_W15_T80",
    band_window: int = 16,
    band_step: int = 8,
    block: int = 128,
    min_windows: int = 50,
    max_per_class: int = 300,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(n_windows, T, b, b) spectral-window covariances, crop labels (0 .. C-1), pixels per window, spatial groups."""
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
    estimator = EstimateCovariance("ledoit_wolf", remove_mean=True, exclude_zero=False)
    data = np.stack([spectral_windows(np.asarray(images[i], np.float64).reshape(-1, images.shape[-1]), band_window,
                                      band_step, estimator) for i in index])
    labels = np.searchsorted(kept_crops, crops[index])
    blocks = centres[index] // block
    groups = np.unique(np.stack([blocks[:, 0], blocks[:, 1], labels], 1), axis=0, return_inverse=True)[1].ravel()
    return data, labels, np.full(len(index), images.shape[1] * images.shape[2]), groups


def kaggle_wheat_series(
    root: str = "/DATA/Kaggle_Prepared/train/HS",
    band_window: int = 16,
    band_step: int = 8,
    n_bands: int = 125,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n_patches, T, b, b) spectral-window covariances, labels (Health / Other / Rust), pixels per patch."""
    import tifffile

    estimator = EstimateCovariance("ledoit_wolf", remove_mean=True, exclude_zero=False)
    data, labels = [], []
    for path in sorted(glob.glob(os.path.join(root, "*.tif"))):
        pixels = tifffile.imread(path).astype(np.float64)[..., :n_bands].reshape(-1, n_bands) / 65535.0
        if not pixels.any():  # 23 patches are entirely zero (atlas)
            continue
        data.append(spectral_windows(pixels, band_window, band_step, estimator))
        labels.append(KAGGLE_WHEAT_CLASSES.index(os.path.basename(path).split("_")[0]))
    return np.stack(data), np.array(labels), np.full(len(labels), 32 * 32)
