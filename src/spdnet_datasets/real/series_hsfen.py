"""
Covariance series from hyperspectral windows centred on labelled pixels (atlas chapter "Hyperspectral
(fenêtres Chikusei, GSOFF, UAV-HSI-Crop)"): each window gives one covariance per spectral window of
``band_window`` bands (step ``band_step``), Ledoit-Wolf over its pixels, as for UAV-HSI-Crop (R6). Positions are
spectral windows: each has its own identity.

- Chikusei v2 (re-extraction of 30/10/2026, scene read in the right orientation): 4 844 windows of 23 x 23
  pixels, 19 classes, official spatial split by 320-pixel blocks with a buffer (fixed). ``n_pixels`` < 529 keeps
  a random subset of the window pixels (fixed per window): the number of observations per covariance changes,
  not the content of the window (controlled n/p factor, R10).
- GSOFF (Gaofeng): 1 106 non-overlapping 16 x 16 windows (top-left corners in ``positions``, checked against the
  source scene); windows with at least ``min_labelled`` labelled pixels in the ground truth and classes with at
  least ``min_windows`` of them; groups = (64-pixel block, class), as UAV-HSI-Crop.

.. note::
   :class:`spdnet_datasets.series.CovarianceSeries` now builds these series on top of the main
   loaders (their samples, splits and groups) for Chikusei and GSOFF (``CovarianceSeries("chikusei", "spectral")``). See the guide page
   "Covariance series". This module is kept to reproduce earlier results.
"""

import os

import numpy as np

from .series import data_root, spectral_windows


def chikusei_series(
    path: str | None = None,
    n_pixels: int = 529,
    band_window: int = 16,
    band_step: int = 8,
    seed: int = 0,
    return_mean: bool = False,
) -> tuple:
    """(N, T, b, b) covariances (T = 15), labels 0 .. 18, pixels per covariance, {"split": 0 / 1 / 2}; with
    ``return_mean``, also the (N, T, b) window means. ``path``: the ``chikusei_v2.h5`` file (default
    ``<SPDNET_DATA_ROOT>/HYPERSPECTRAL/Chikusei/prepared_v2/chikusei_v2.h5``)."""
    import h5py

    path = data_root(path, "HYPERSPECTRAL", "Chikusei", "prepared_v2", "chikusei_v2.h5")

    with h5py.File(path, "r") as f:
        split = f["w23/split"][:]
        kept = np.nonzero(split >= 0)[0]  # -1: buffer between blocks of different splits
        windows = f["w23/windows"]
        labels = f["w23/labels"][:][kept].astype(int) - 1
        rng = np.random.default_rng(seed)
        data, means = [], []
        for i in kept:
            pixels = np.asarray(windows[i], np.float64).reshape(-1, windows.shape[-1])
            if n_pixels < len(pixels):
                pixels = pixels[np.sort(rng.choice(len(pixels), n_pixels, replace=False))]
            covariances, mean = spectral_windows(pixels, band_window, band_step, return_mean=True)
            data.append(covariances)
            means.append(mean)
    out = (np.stack(data), labels, np.full(len(kept), min(n_pixels, 529)), {"split": split[kept].astype(np.int64)})
    return (*out, np.stack(means)) if return_mean else out


def gsoff_series(
    root: str | None = None,
    band_window: int = 16,
    band_step: int = 8,
    min_labelled: float = 0.5,
    min_windows: int = 30,
    block: int = 64,
    return_mean: bool = False,
) -> tuple:
    """(N, T, b, b) covariances (T = 14), labels 0 .. C-1, pixels per covariance (256), spatial groups; with
    ``return_mean``, also the (N, T, b) window means."""
    import tifffile

    root = data_root(root, "HYPERSPECTRAL", "GSOFF")

    images = np.load(os.path.join(root, "gaofeng_windows_data.npy"), mmap_mode="r")  # (N, 16, 16, 125)
    classes = np.load(os.path.join(root, "gaofeng_windows_labels.npy")).astype(int)
    corners = np.load(os.path.join(root, "gaofeng_windows_positions.npy"))
    truth = tifffile.imread(os.path.join(root, "source", "data", "testarea2_new", "sg_quac_test2_gt.tif"))
    w = images.shape[1]
    labelled = np.array([(truth[r:r + w, c:c + w] > 0).mean() for r, c in corners])
    keep = labelled >= min_labelled
    kept_classes = [c for c in np.unique(classes) if (keep & (classes == c)).sum() >= min_windows]
    index = np.nonzero(keep & np.isin(classes, kept_classes))[0]
    tokens = [spectral_windows(np.asarray(images[i], np.float64).reshape(-1, images.shape[-1]), band_window,
                               band_step, return_mean=True) for i in index]
    data = np.stack([c for c, _ in tokens])
    labels = np.searchsorted(kept_classes, classes[index])
    blocks = corners[index] // block
    groups = np.unique(np.stack([blocks[:, 0], blocks[:, 1], labels], 1), axis=0, return_inverse=True)[1].ravel()
    out = (data, labels, np.full(len(index), w * w), groups)
    return (*out, np.stack([m for _, m in tokens])) if return_mean else out
