"""
Covariance series from AVIRIS-C CA TS14 MIX (data atlas): 2034 windows of 600 m (32 x 32 pixels) over California,
14 acquisition dates, 224 bands (166 clean ones: 424-1293, 1483-1772 and 2038-2437 nm), label = proportions of 13
land-cover classes (CDL). Here a pure classification task with fewer classes (user request, 14/11): the proportions
are summed into 7 groups and a window is kept when its dominant group covers at least ``purity`` of it.

:func:`aviris_mix_series` builds "spectral ViT" / "spatial ViT" series: for every valid date, the window is cut into
``patches`` x ``patches`` spatial blocks and the clean bands (one in ``step``) into ``n_groups`` overlapping spectral
groups of ``width`` bands; one Ledoit-Wolf covariance (width x width) per (date, block, spectral group), estimated
over the pixels of the block. Tokens in the order date, block, spectral group; dates rejected by the quality control
are dropped and the series is padded at the end with identity tokens.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

AVIRIS_MIX_ROOT = "/DATA/DATASET/HYPERSPECTRAL/AVIRIS-C_CA_TS14_MIX"
AVIRIS_MIX_GROUPS = {"urbain": ("urbain_dense", "urbain_faible", "urbain_moyen", "urbain_ouvert"),
                     "foret": ("foret_coniferes", "foret_mixte"), "arbustes": ("arbustes",),
                     "prairie": ("prairie_paturage",), "vergers_vignes": ("amandiers", "vignes"), "eau": ("eau",),
                     "sol_nu": ("sol_nu",)}
AVIRIS_MIX_CLASSES = list(AVIRIS_MIX_GROUPS)
N_DATES = 14


AVIRIS_MIX_PROPORTIONS = [f"p_{name}" for names in AVIRIS_MIX_GROUPS.values() for name in names] + ["p_autre"]


def aviris_mix_selection(root: str = AVIRIS_MIX_ROOT, purity: float = 0.5, min_dates: int = 5,
                         target: str = "class") -> pd.DataFrame:
    """Windows kept for the classification: columns window_id, label (rank in AVIRIS_MIX_CLASSES), share (of the
    dominant group), block10km, file. The class "autre" is never a label (its share counts against the purity)."""
    labels = pd.read_csv(f"{root}/meta/labels.csv")
    shares = pd.DataFrame({group: labels[[f"p_{name}" for name in names]].sum(axis=1)
                           for group, names in AVIRIS_MIX_GROUPS.items()})
    out = labels[["window_id", "dominant", "block10km", "n_ok_dates"]].copy()
    out["label"], out["share"] = shares.to_numpy().argmax(1), shares.to_numpy().max(1)
    out["file"] = [f"{root}/windows/{w}_{d}.npz" for w, d in zip(labels.window_id, labels.dominant)]
    keep = (out.share >= purity) & (out.n_ok_dates >= min_dates) & (out.share >= labels.p_autre)
    if target == "proportions":  # R55: every window with enough dates, the 13 proportions as regression targets
        keep = out.n_ok_dates >= min_dates
        out[AVIRIS_MIX_PROPORTIONS] = labels[AVIRIS_MIX_PROPORTIONS]
    return out[keep].reset_index(drop=True)


def spectral_groups(n_bands: int, width: int, n_groups: int) -> np.ndarray:
    """Start index of ``n_groups`` groups of ``width`` consecutive bands, evenly spread over ``n_bands``."""
    return np.round(np.linspace(0, n_bands - width, n_groups)).astype(int)


def aviris_mix_series(root: str = AVIRIS_MIX_ROOT, purity: float = 0.5, min_dates: int = 5, width: int = 48,
                      n_groups: int = 4, step: int = 1, patches: int = 1, n_bands: int | None = None,
                      target: str = "class") -> tuple:
    """(N, 14 x patches^2 x n_groups, width, width) covariances, labels 0..6, median pixels per token, groups =
    10 km block (never across the splits). Reflectance clipped to [0, 1.5], each band divided by its global standard
    deviation (no label used); a (date, block) with fewer than 2 x width complete pixels is dropped with its date."""
    selection = aviris_mix_selection(root, purity, min_dates, target)
    bands = np.flatnonzero(pd.read_csv(f"{root}/meta/bands.csv").clean.to_numpy() == 1)[::step]
    if n_bands is not None:  # R55: decimation to n_bands clean bands evenly spread over the spectrum
        bands = bands[np.round(np.linspace(0, len(bands) - 1, n_bands)).astype(int)]
    starts = spectral_groups(len(bands), width, n_groups)
    mask = np.load(f"{root}/meta/date_mask.npz")
    ok = dict(zip(mask["window_id"].tolist(), mask["ok"]))
    side = 32 // patches
    per_date = patches * patches * n_groups

    def pixels(path: str) -> np.ndarray:  # (14, 32, 32, bands)
        cube = np.load(path)["cube"][:, bands].astype(np.float64)
        return np.clip(np.moveaxis(cube, 1, -1), 0.0, 1.5)

    sample = np.concatenate([pixels(path)[:, ::4, ::4].reshape(-1, len(bands))
                             for path in selection.file[:: max(len(selection) // 150, 1)]])
    scale = np.nanstd(sample, axis=0) + 1e-12
    data, n_obs = np.tile(np.eye(width), (len(selection), N_DATES * per_date, 1, 1)), []
    for i, (window, path) in enumerate(zip(selection.window_id, selection.file)):
        cube, t, counts = pixels(path) / scale, 0, []
        for date in np.flatnonzero(ok[window]):
            tokens = []
            for row in range(patches):
                for column in range(patches):
                    x = cube[date, row * side:(row + 1) * side, column * side:(column + 1) * side].reshape(-1, len(bands))
                    x = x[np.isfinite(x).all(1)]
                    if len(x) < 2 * width or np.ptp(x, axis=0).min() == 0:  # too few pixels or a constant band
                        tokens = None
                        break
                    counts.append(len(x))
                    for start in starts:
                        covariance = ledoit_wolf(x[:, start:start + width])[0]
                        tokens.append(covariance + 1e-6 * np.trace(covariance) / width * np.eye(width))
                if tokens is None:
                    break
            if tokens is not None:
                data[i, t * per_date:(t + 1) * per_date] = np.stack(tokens)
                t += 1
        n_obs.append(np.median(counts) if counts else 0)
    if target == "proportions":  # float targets (N, 13); strata = dominant group, for the stratified split
        return (data, selection[AVIRIS_MIX_PROPORTIONS].to_numpy(np.float64), np.array(n_obs),
                {"groups": selection.block10km.to_numpy(), "strata": selection.label.to_numpy()})
    return data, selection.label.to_numpy(), np.array(n_obs), selection.block10km.to_numpy()


def whiten_series(data: np.ndarray, period: int, load: float = 1e-3) -> np.ndarray:
    """Congruence of every token by the inverse square root of a reference: tokens at positions k, k + period, ...
    (same spectral group) share the reference M_k = log-Euclidean mean of the valid tokens of the whole dataset (no
    label used); token <- M_k^{-1/2} token M_k^{-1/2}, then diagonal loading ``load`` x trace / p. A congruence is an
    isometry of the affine-invariant metric: distances between tokens of a group are unchanged, but the tokens are
    brought around the identity and their condition number drops (AVIRIS: median 10^4.1 -> 10^2). Identity tokens
    (padding) are left unchanged."""
    out, p = data.copy(), data.shape[-1]
    valid = (data != np.eye(p)).any((-1, -2))
    for k in range(period):
        tokens = data[:, k::period][valid[:, k::period]]
        values, vectors = np.linalg.eigh(tokens)
        mean_log = ((vectors * np.log(values)[..., None, :]) @ vectors.swapaxes(-1, -2)).mean(0)
        values, vectors = np.linalg.eigh(mean_log)
        whitener = (vectors * np.exp(-0.5 * values)) @ vectors.T
        tokens = whitener @ tokens @ whitener
        tokens = (tokens + tokens.swapaxes(-1, -2)) / 2
        tokens = tokens + load * np.trace(tokens, axis1=-2, axis2=-1)[:, None, None] / p * np.eye(p)
        view = out[:, k::period]
        view[valid[:, k::period]] = tokens
    return out
