"""Pol-InSAR-Island (F-SAR, DLR; L and S bands over Baltrum; Hochstuhl et al., 2023) as ViT-like series of covariances.

Normalisation statistics (band powers, feature standardisation) are fitted on the cells selected by ``fit_mask``:
a boolean (N,) array in the output order, or a callable ``fit_mask(labels, groups) -> mask`` evaluated once the
cells are known (e.g. the training blocks of a grouped split). ``None`` fits them on every cell, test included,
with a warning.
"""

import re

import numpy as np

from .series import covariance, data_root, resolve_fit_mask

POLINSAR_CLASSES = ["tidal flat", "water", "coastal shrub", "dense high vegetation", "white dune", "peat bog", "grey dune",
                    "couch grass", "upper saltmarsh", "lower saltmarsh", "sand", "settlement"]


def _envi(path: str, dtype: str) -> np.ndarray:
    header = open(path + ".hdr").read()
    shape = (int(re.search(r"lines\s*=\s*(\d+)", header).group(1)), int(re.search(r"samples\s*=\s*(\d+)", header).group(1)))
    return np.fromfile(path, dtype=dtype).reshape(shape)


def _block_mean(raster: np.ndarray, size: int) -> np.ndarray:
    h, w = raster.shape[0] // size * size, raster.shape[1] // size * size
    return raster[:h, :w].reshape(h // size, size, w // size, size).mean((1, 3), dtype=np.float64)


def _t6(folder: str, size: int) -> np.ndarray:
    """6 x 6 complex coherency matrices averaged over non-overlapping ``size`` x ``size`` pixel blocks: (H, W, 6, 6)."""
    t6 = None
    for i in range(6):
        for j in range(i, 6):
            if i == j:
                value = _block_mean(_envi(f"{folder}/T{i + 1}{j + 1}.bin", "<f4"), size).astype(complex)
            else:
                value = (_block_mean(_envi(f"{folder}/T{i + 1}{j + 1}_real.bin", "<f4"), size)
                         + 1j * _block_mean(_envi(f"{folder}/T{i + 1}{j + 1}_imag.bin", "<f4"), size))
            if t6 is None:
                t6 = np.zeros((*value.shape, 6, 6), dtype=complex)
            t6[..., i, j] = value
            t6[..., j, i] = value.conj()
    return t6


def realify(hermitian: np.ndarray) -> np.ndarray:
    """Hermitian positive definite (..., p, p) -> real SPD (..., 2p, 2p), [[Re, -Im], [Im, Re]]: an injective algebra
    morphism (products, inverses, logarithms commute with it), each eigenvalue is doubled."""
    re_, im = hermitian.real, hermitian.imag
    return np.concatenate([np.concatenate([re_, -im], -1), np.concatenate([im, re_], -1)], -2)


def polinsar_island_series(root: str | None = None, cell: int = 32, patch: int = 8, purity: float = 0.75,
                           block: int = 512, bands: tuple[str, ...] = ("L", "S"), stack_bands: bool = False,
                           fit_mask=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ViT-like series: each labelled cell of ``cell`` x ``cell`` pixels (1 m) is cut into (cell / patch)^2 patches in
    raster order, and each patch gives one token per band: the T6 coherency matrix averaged over its patch^2 pixels,
    in its real 12 x 12 form. Returns (N, T = bands * (cell / patch)^2, 12, 12) with the tokens of a patch adjacent
    (L then S), labels 0..11, pixels per token, and spatial groups (blocks of ``block`` m in UTM coordinates, shared by
    the two flight paths, which overlap: a block never crosses the splits).
    Cells: non-overlapping, official train and test labels merged, kept when at least half the pixels are labelled and
    the majority class holds ``purity`` of them (definition of the atlas). Each band is divided by its mean power over
    the tokens of the ``fit_mask`` cells (one scalar per band, no label used), and tokens get a relative ridge of
    1e-6."""
    root = data_root(root, "SAR", "Pol-InSAR-Island")
    side = cell // patch
    data, labels, groups = [], [], []
    for path in ("FP1", "FP2"):
        label = np.maximum(_envi(f"{root}/label/{path}/label_train.bin", "u1"), _envi(f"{root}/label/{path}/label_test.bin", "u1"))
        header = open(f"{root}/label/{path}/label_train.bin.hdr").read()
        east, north = (float(v) for v in re.search(r"map info = \{UTM, 1, 1, ([\d.]+), ([\d.]+)", header).groups())
        tokens = np.stack([realify(_t6(f"{root}/data/{path}/{band}/T6", patch)) for band in bands], 2)  # (H, W, B, 12, 12)
        for r in range(label.shape[0] // cell):
            for c in range(label.shape[1] // cell):
                window = label[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
                counts = np.bincount(window[window > 0], minlength=13)
                if counts.sum() < cell * cell // 2 or counts.max() < purity * counts.sum():
                    continue
                data.append(tokens[r * side:(r + 1) * side, c * side:(c + 1) * side].reshape(-1, 12, 12))
                labels.append(counts.argmax() - 1)
                x, y = east + (c + 0.5) * cell, north - (r + 0.5) * cell
                groups.append(int(x // block) * 100000 + int(y // block))
    data = np.stack(data)
    labels, groups = np.array(labels), np.unique(groups, return_inverse=True)[1]
    fit = resolve_fit_mask(fit_mask, len(data), labels, groups)
    n_bands = len(bands)
    for b in range(n_bands):
        data[:, b::n_bands] /= np.trace(data[fit, b::n_bands], axis1=-2, axis2=-1).mean() / 12
    data += 1e-6 * (np.trace(data, axis1=-2, axis2=-1) / 12)[..., None, None] * np.eye(12)
    if stack_bands:  # one token per patch: block-diagonal of its bands, (N, T / bands, 12 * bands, 12 * bands)
        stacked = np.zeros((data.shape[0], data.shape[1] // n_bands, 12 * n_bands, 12 * n_bands))
        for b in range(n_bands):
            stacked[:, :, 12 * b:12 * (b + 1), 12 * b:12 * (b + 1)] = data[:, b::n_bands]
        data = stacked
    return data, labels, np.full(len(labels), patch * patch), groups


def calvo_oller(mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Gaussian N(mean, covariance) -> SPD (..., p + 1, p + 1), [[S + m m^T, m], [m^T, 1]] (Calvo and Oller, 1990);
    its determinant is det(S)."""
    p = mean.shape[-1]
    out = np.empty((*mean.shape[:-1], p + 1, p + 1))
    out[..., :p, :p] = covariance + mean[..., :, None] * mean[..., None, :]
    out[..., :p, p] = out[..., p, :p] = mean
    out[..., p, p] = 1.0
    return out


def polinsar_logpower_series(root: str | None = None, cell: int = 32, patch: int = 8, purity: float = 0.75,
                             block: int = 512, bands: tuple[str, ...] = ("L", "S"), mean: bool = False,
                             fit_mask=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Variant C of the atlas, ViT-like: the features of a pixel are the log-powers of the 6 diagonal terms of T6 in
    each band (12 real features: the bands are IN the covariance, with true cross-band terms); each patch of
    ``patch`` x ``patch`` pixels of a cell gives one token, the Ledoit-Wolf covariance of its pixels (12 x 12).
    With ``mean``, the token is the Calvo-Oller embedding of (patch mean, covariance), 13 x 13: the centred covariance
    alone forgets the backscatter level. Features are standardised by their mean and std over the pixels of the
    ``fit_mask`` cells (no label used). Same cells, labels and groups as ``polinsar_island_series``, in the same
    order. Returns (N, (cell / patch)^2, p, p), labels 0..11, pixels per token, spatial groups."""
    root = data_root(root, "SAR", "Pol-InSAR-Island")
    side = cell // patch
    pixels, labels, groups = [], [], []
    for path in ("FP1", "FP2"):
        label = np.maximum(_envi(f"{root}/label/{path}/label_train.bin", "u1"), _envi(f"{root}/label/{path}/label_test.bin", "u1"))
        header = open(f"{root}/label/{path}/label_train.bin.hdr").read()
        east, north = (float(v) for v in re.search(r"map info = \{UTM, 1, 1, ([\d.]+), ([\d.]+)", header).groups())
        power = np.stack([np.log(np.maximum(_envi(f"{root}/data/{path}/{band}/T6/T{i}{i}.bin", "<f4"), 1e-12))
                          for band in bands for i in range(1, 7)], -1)  # (H, W, 12)
        for r in range(label.shape[0] // cell):
            for c in range(label.shape[1] // cell):
                window = label[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
                counts = np.bincount(window[window > 0], minlength=13)
                if counts.sum() < cell * cell // 2 or counts.max() < purity * counts.sum():
                    continue
                x = power[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell].astype(np.float64)
                # (side, patch, side, patch, f) -> (side * side patches in raster order, patch^2 pixels, f)
                pixels.append(x.reshape(side, patch, side, patch, -1).transpose(0, 2, 1, 3, 4).reshape(side * side, patch * patch, -1))
                labels.append(counts.argmax() - 1)
                groups.append(int((east + (c + 0.5) * cell) // block) * 100000 + int((north - (r + 0.5) * cell) // block))
    pixels = np.stack(pixels)  # (N, T, n, f)
    labels, groups = np.array(labels), np.unique(groups, return_inverse=True)[1]
    fit = resolve_fit_mask(fit_mask, len(pixels), labels, groups)
    pixels = (pixels - pixels[fit].mean((0, 1, 2))) / pixels[fit].std((0, 1, 2))
    centre = pixels.mean(2)
    tokens = np.stack([np.stack([covariance(token) for token in series]) for series in pixels])
    data = calvo_oller(centre, tokens) if mean else tokens
    return data, labels, np.full(len(labels), patch * patch), groups
