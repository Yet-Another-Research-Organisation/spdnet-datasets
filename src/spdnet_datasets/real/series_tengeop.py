"""
Covariance series from TenGeoP-SARwv (Sentinel-1A wave mode, VV, 2016; 10 ocean phenomena; atlas chapter 11).

Each vignette (uint16 intensity quick-look, (463..493, 431..573) pixels, GDAL ``SCALE`` / ``OFFSET`` applied) is
centre-cropped to a common square, cut into a row-major grid of windows, and each window gives the Ledoit-Wolf
covariance of its vectorised overlapping ``patch x patch`` patches. Unlike the other series, the positions have
no fixed identity: an ocean phenomenon can appear anywhere in the vignette (control for the hypothesis "the SPD
model wins when each position has its own identity").

The label is the class folder (F .. O), identical to the label file ``58683.txt``. Vignettes of one datatake
(``<mission>-...-<datatake id>-<index>`` in the file name) are grouped so that a split never separates them
(atlas recommendation).
"""

import glob
import os
import re

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..estimator.covariance import EstimateCovariance

TENGEOP_CLASSES = ["Pure Ocean Waves", "Wind Streaks", "Micro Convective Cells", "Rain Cells", "Biological Slicks",
                   "Sea Ice", "Iceberg", "Low Wind Area", "Atmospheric Front", "Oceanic Front"]  # folders F .. O
TENGEOP_FOLDERS = "FGHIJKLMNO"
TENGEOP_MIN_SIDE = 431  # smallest side over the 37 553 vignettes (shapes range from (463, 431) to (493, 573))


def read_vignette(path: str) -> np.ndarray:
    """uint16 GeoTIFF -> float64 intensity, value * SCALE + OFFSET from the GDAL metadata (as in the atlas)."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        image = page.asarray().astype(np.float64)
        metadata = page.tags[42112].value if 42112 in page.tags else ""
    scale = re.search(r'name="SCALE"[^>]*>([^<]+)<', metadata)
    offset = re.search(r'name="OFFSET"[^>]*>([^<]+)<', metadata)
    return image * (float(scale.group(1)) if scale else 1.0) + (float(offset.group(1)) if offset else 0.0)


def patch_window_covariances(image: np.ndarray, window: int, step: int, patch: int,
                             estimator: EstimateCovariance) -> np.ndarray:
    """(H, W) image -> (T, patch^2, patch^2) covariances of the windows of a row-major grid (window, step),
    diagonally loaded by 1e-6 tr(C) / p."""
    p = patch * patch
    covariances = []
    for y in range(0, image.shape[0] - window + 1, step):
        for x in range(0, image.shape[1] - window + 1, step):
            patches = sliding_window_view(image[y:y + window, x:x + window], (patch, patch)).reshape(-1, p)
            covariances.append(estimator(patches))
    covariances = np.stack(covariances)
    trace = np.trace(covariances, axis1=-2, axis2=-1)[:, None, None] / p
    return covariances + 1e-6 * trace * np.eye(p)


def tengeop_series(
    root: str = "/DATA/TenGeoP-SARwv",
    window: int = 96,
    step: int = 96,
    patch: int = 4,
    max_per_class: int = 300,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(N, T, patch^2, patch^2) patch covariances over a grid of windows, labels 0..9 (F .. O), patches per window,
    {"groups": datatake of each vignette}. T = g^2 with g = (431 - window) // step + 1 (4 x 4 = 16 by default)."""
    side = (TENGEOP_MIN_SIDE - window) // step + 1
    crop = (side - 1) * step + window  # common centred square crop
    rng = np.random.default_rng(seed)
    files, labels = [], []
    for label, folder in enumerate(TENGEOP_FOLDERS):
        paths = sorted(glob.glob(os.path.join(root, folder, "*.tiff")))
        if len(paths) < max_per_class:
            print(f"TenGeoP: class {folder} ({TENGEOP_CLASSES[label]}) has only {len(paths)} vignettes")
        chosen = rng.permutation(len(paths))[:max_per_class]
        files += [paths[i] for i in sorted(chosen)]
        labels += [label] * len(chosen)
    estimator = EstimateCovariance("ledoit_wolf", remove_mean=True, exclude_zero=False)
    data = []
    for path in files:
        image = read_vignette(path)
        top, left = (image.shape[0] - crop) // 2, (image.shape[1] - crop) // 2
        data.append(patch_window_covariances(image[top:top + crop, left:left + crop], window, step, patch, estimator))
    # s1a-wv1-slc-vv-<start>-<stop>-<orbit>-<datatake>-<index>.tiff
    datatakes = ["-".join(os.path.basename(f).split("-")[i] for i in (0, 7)) for f in files]
    _, groups = np.unique(datatakes, return_inverse=True)
    n_obs = np.full(len(files), (window - patch + 1) ** 2)
    return np.stack(data), np.array(labels), n_obs, {"groups": groups}
