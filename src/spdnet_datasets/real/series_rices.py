"""
Covariance series of rice grains (Rices_90, atlas v2 sheet: 90 varieties, 180 VNIR cubes of about 48 grains, two
bundles per variety). Same preparation as the atlas script ``rices_90.py``: dark reference of the session
subtracted, relative reflectance against the white panel of the cube (rows above it only), Otsu mask on the mean
reflectance, opening, connected components whose area lies in [0.4, 2] x the median area = grains. The 256 bands
are averaged by 4 (64 bands, atlas recommendation: n/p = 1.7 at 256 bands); each grain gives one Ledoit-Wolf
covariance per spectral window of ``band_window`` bands (step ``band_step``; ``band_window = 64``: the global
covariance, T = 1).

Split (fixed, official): train = bundle 1 (20 % of its grains drawn as validation), test = bundle 2. For 73
varieties both bundles come from the same session (variety confounded with the session); ``cross_session_test``
keeps in the test set only the 17 varieties whose bundles come from two sessions (the only unconfounded score).
"""

import os

import numpy as np

from .series import data_root, otsu_threshold, spectral_windows


def _read_envi(header: str) -> np.ndarray:
    """ENVI bil uint16 cube -> (lines, samples, bands) memmap view."""
    fields = {}
    for line in open(header, errors="ignore"):
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip().lower()] = value.strip()
    lines, samples, bands = int(fields["lines"]), int(fields["samples"]), int(fields["bands"])
    assert fields["interleave"] == "bil" and fields["data type"] == "12", header
    raw = np.memmap(header[:-4] + ".raw", dtype="<u2", mode="r", shape=(lines, bands, samples))
    return raw.transpose(0, 2, 1)


def _grains(cube: np.ndarray, dark: np.ndarray) -> list[np.ndarray]:
    """(lines, samples, 256) cube -> list of (n_pixels, 256) relative reflectances of the grains."""
    from scipy import ndimage

    cube = cube.astype(np.float32) - dark[None]
    profile = np.median(cube.mean(-1), 1)
    rows = np.nonzero(profile > 0.5 * profile.max())[0]  # white reference panel
    first, last = int(rows.min()), int(rows.max())
    white = cube[np.arange(first + 5, max(first + 6, last - 4))].mean(0)
    cube = cube[:max(first - 8, 10)] / np.maximum(white[None], 1.0)
    brightness = cube.mean(-1)
    mask = ndimage.binary_opening(brightness > otsu_threshold(brightness.ravel()), iterations=1)
    labels, n = ndimage.label(mask)
    areas = ndimage.sum(mask, labels, range(1, n + 1))
    median = np.median(areas[areas > 30]) if (areas > 30).any() else 0
    objects = ndimage.find_objects(labels)
    return [cube[objects[i]][labels[objects[i]] == i + 1].astype(np.float64)
            for i in range(n) if 0.4 * median <= areas[i] <= 2.0 * median]


def rices_series(
    root: str | None = None,
    band_window: int = 16,
    band_step: int = 8,
    cross_session_test: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(n_grains, T, b, b) covariances (T = 7 for 16 / 8, 1 for 64 / 64), varieties 0..89, pixels per grain,
    {"split": 0 train / 1 validation / 2 test / -1 dropped}."""
    import pandas as pd

    root = data_root(root, "HYPERSPECTRAL", "Rices_90")
    index = pd.read_csv(os.path.join(root, "index.csv"))
    varieties = sorted(index["Species Full Name"].unique())
    sessions = index.groupby("Species Full Name").Folder.nunique()
    darks = {}
    rng = np.random.default_rng(seed)
    data, labels, n_pixels, split = [], [], [], []
    for row in index.itertuples():
        folder, name, bundle = row.Folder, row[5], row[3]
        if folder not in darks:
            darks[folder] = np.asarray(_read_envi(os.path.join(root, folder, "black.hdr")), np.float32).mean(0)
        grains = _grains(np.asarray(_read_envi(os.path.join(root, folder, f"{name}.hdr"))), darks[folder])
        parts = np.full(len(grains), 2)
        if bundle == 1:
            parts[:] = 0
            parts[rng.permutation(len(grains))[:round(0.2 * len(grains))]] = 1
        elif cross_session_test and sessions[row[1]] < 2:
            parts[:] = -1
        for pixels in grains:
            binned = pixels.reshape(len(pixels), 64, 4).mean(-1)
            data.append(spectral_windows(binned, band_window, band_step))
            n_pixels.append(len(pixels))
        labels += [varieties.index(row[1])] * len(grains)
        split += list(parts)
    return np.stack(data), np.array(labels), np.array(n_pixels), {"split": np.array(split)}
