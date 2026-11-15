"""Building blocks shared by the hyperspectral dataset modules.

A hyperspectral sample is a set of pixels (a window, a grain, a leaf...) whose
``p`` spectral bands are the variables and whose ``n`` pixels are the
observations of one ``p x p`` covariance.

The modules follow the same staged pipeline (:func:`core.fetch.run_pipeline`):

- ``processed`` stage: the sample list (``meta.csv``: one row per sample, a
  ``label`` column, positions, groups...), ``info.json`` (counts of discarded
  samples) and, when the samples are extracted from a scene, ``windows.npy``
  (N, h, w, p) float32;
- ``covariance`` stage: ``covs.npy`` (B, p, p) unscaled float64, ``means.npy``
  (B, p), ``n_obs.npy`` (B,) and a copy of ``meta.csv`` / ``info.json``
  restricted to the samples whose estimate is SPD (column ``proc_index`` =
  row in the processed stage).

:class:`HSIDataset` turns the result of a ``fetch_<dataset>()`` into a
:class:`~spdnet_datasets.base.BaseDataset` (tasks, class limits, balancing,
normalisation fitted on the training split, output contract).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch

from ..base import BaseDataset
from ..core.estimators import (
    InsufficientSamplesError,
    NotSPDError,
    channel_mean,
    estimate_covariance,
)
from ..core.fetch import Bunch
from ..core.normalize import geometric_scale
from ..core.tasks import apply_task, balance_indices, class_weights

# ---------------------------------------------------------------------------
# Pixels and covariances


def valid_pixels(X):
    """Rows of ``X`` (n, p) that are finite and not identically zero."""
    X = np.asarray(X)
    keep = np.isfinite(X).all(axis=1) & (X != 0).any(axis=1)
    return X[keep]


def band_index(n_bands, band_step=1, start=0, stop=None):
    """Indices of the kept bands: ``range(start, stop, band_step)``."""
    if band_step < 1:
        raise ValueError("band_step must be >= 1")
    return np.arange(n_bands)[start:stop:band_step]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=_to_builtin))


def _to_builtin(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def covariance_stage(proc_dir, out_dir, pixels_fn, estimator="scm"):
    """Estimate one covariance per sample of the processed stage.

    Parameters
    ----------
    proc_dir, out_dir : Path
        Processed stage (``meta.csv``, ``info.json``) and covariance stage.
    pixels_fn : callable(proc_dir, meta) -> iterable of (n_i, p) arrays
        Observations of each sample, in the order of ``meta``.
    estimator : str
        One of :data:`core.estimators.ESTIMATORS`.

    Samples with fewer than ``2p`` observations, or whose estimate is not SPD
    (constant band, duplicated pixels...), are dropped and counted in
    ``info.json`` (``excluded_n_lt_2p_at_estimation``, ``excluded_not_spd``).
    """
    import pandas as pd

    proc_dir, out_dir = Path(proc_dir), Path(out_dir)
    meta = pd.read_csv(proc_dir / "meta.csv")
    info = json.loads((proc_dir / "info.json").read_text())
    covs, means, n_obs, kept, short = [], [], [], [], 0
    for i, X in enumerate(pixels_fn(proc_dir, meta)):
        X = np.asarray(X, dtype=np.float64)
        try:
            C = estimate_covariance(X, estimator)
        except InsufficientSamplesError:
            short += 1
            continue
        except NotSPDError:
            continue
        covs.append(C)
        means.append(channel_mean(X))
        n_obs.append(len(X))
        kept.append(i)
    if not kept:
        raise RuntimeError("No sample gave an SPD covariance")
    np.save(out_dir / "covs.npy", np.stack(covs))
    np.save(out_dir / "means.npy", np.stack(means))
    np.save(out_dir / "n_obs.npy", np.asarray(n_obs))
    info["excluded_n_lt_2p_at_estimation"] = short
    info["excluded_not_spd"] = len(meta) - len(kept) - short
    info["n_samples"] = len(kept)
    meta = meta.iloc[kept].copy()
    meta.insert(0, "proc_index", kept)
    meta.to_csv(out_dir / "meta.csv", index=False)
    write_json(out_dir / "info.json", info)


def window_pixels(proc_dir, meta):
    """Default ``pixels_fn``: valid pixels of ``windows.npy`` (N, h, w, p)."""
    windows = np.load(Path(proc_dir) / "windows.npy", mmap_mode="r")
    for i in range(len(meta)):
        w = np.asarray(windows[i])
        yield valid_pixels(w.reshape(-1, w.shape[-1]))


def load_covariance_dir(cov_dir):
    """Read a covariance stage: Bunch(covariances, means, n_obs, meta, info)."""
    import pandas as pd

    cov_dir = Path(cov_dir)
    return Bunch(
        covariances=np.load(cov_dir / "covs.npy", mmap_mode="r"),
        means=np.load(cov_dir / "means.npy"),
        n_obs=np.load(cov_dir / "n_obs.npy"),
        meta=pd.read_csv(cov_dir / "meta.csv"),
        info=json.loads((cov_dir / "info.json").read_text()),
    )


# ---------------------------------------------------------------------------
# Window extraction on a labelled scene


def _box_sums(mask, w):
    """Sum of ``mask`` over every ``w x w`` window, indexed by its top-left corner."""
    S = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    S[1:, 1:] = np.cumsum(np.cumsum(mask, axis=0), axis=1)
    return S[w:, w:] - S[:-w, w:] - S[w:, :-w] + S[:-w, :-w]


def purity_grid_windows(gt, window, threshold=0.8):
    """Non-overlapping windows per class on the grid offset that maximises them.

    For each class ``c`` (non-zero codes of ``gt``), every grid of step
    ``window`` with offset ``(oy, ox)`` in ``[0, window)^2`` is scanned; a
    window is *pure* for ``c`` when at least ``threshold`` of its labelled
    pixels (``gt > 0``) are ``c``. The offset with the most pure windows is
    kept (first one in row-major order on ties), as in the original GSOFF and
    UAV-HSI-Crop preprocessing scripts. Windows of different classes may
    therefore overlap.

    Returns
    -------
    positions : int array (k, 2)
        Top-left (row, column) of each window.
    labels : int array (k,)
        Class code of each window.
    purity : float array (k,)
        Fraction of the labelled pixels that belong to the class.
    """
    gt = np.asarray(gt)
    labelled = _box_sums(gt > 0, window)
    positions, labels, purity = [], [], []
    for c in np.unique(gt[gt > 0]):
        count = _box_sums(gt == c, window)
        pure = (labelled > 0) & (count >= threshold * labelled)
        best, best_offset = 0, (0, 0)
        for oy in range(window):
            for ox in range(window):
                n = int(pure[oy::window, ox::window].sum())
                if n > best:
                    best, best_offset = n, (oy, ox)
        oy, ox = best_offset
        grid = np.zeros_like(pure)
        grid[oy::window, ox::window] = True
        rows, cols = np.nonzero(pure & grid)
        positions.append(np.stack([rows, cols], 1))
        labels.append(np.full(len(rows), c))
        purity.append(count[rows, cols] / labelled[rows, cols])
    if not positions:
        return np.zeros((0, 2), int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(positions), np.concatenate(labels), np.concatenate(purity)


# ---------------------------------------------------------------------------
# Splits


def window_boxes(positions, window, scene=None, scene_offset=10**7):
    """Boxes ``(r0, c0, r1, c1)`` of square windows, scenes shifted apart."""
    pos = np.asarray(positions, dtype=np.int64).copy()
    if scene is not None:
        pos[:, 0] += np.asarray(scene, dtype=np.int64) * scene_offset
    return np.concatenate([pos, pos + window], axis=1)


def _near(coords_a, coords_b, gap):
    """Mask of ``coords_a`` rows at Chebyshev distance < ``gap`` from ``coords_b``."""
    if not len(coords_a) or not len(coords_b):
        return np.zeros(len(coords_a), bool)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(coords_b).query(coords_a, p=np.inf)
    return d < gap


def block_kfold_split(positions, y, *, window, block, n_splits=5, fold=0, seed=0,
                      scene=None):
    """Train/val/test split on square spatial blocks, without overlapping windows.

    Blocks of ``block`` pixels (per scene) are the groups of a
    :class:`~sklearn.model_selection.StratifiedGroupKFold`; test = fold
    ``fold``, val = fold ``fold + 1`` and train = the other folds. Val and test
    windows closer than ``window`` (Chebyshev distance between top-left
    corners, i.e. overlapping) to a training window are removed, and so are
    test windows overlapping a validation window.

    Returns
    -------
    dict with ``train``, ``val``, ``test`` index arrays and ``dropped``
    (number of windows removed by the buffer).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    pos = np.asarray(positions, dtype=np.int64)
    sc = np.zeros(len(pos), np.int64) if scene is None else np.asarray(scene, np.int64)
    groups = (sc * 10**6 + (pos[:, 0] // block)) * 10**6 + pos[:, 1] // block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # rare classes: fewer members than folds
        folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                          random_state=seed).split(pos, y, groups))
    test = folds[fold][1]
    val = folds[(fold + 1) % n_splits][1]
    train = np.setdiff1d(np.arange(len(pos)), np.concatenate([test, val]))
    coords = pos.astype(float)
    coords[:, 0] += sc * 1e7
    n_before = len(val) + len(test)
    val = val[~_near(coords[val], coords[train], window)]
    test = test[~_near(coords[test], coords[train], window)]
    test = test[~_near(coords[test], coords[val], window)]
    return {"train": train, "val": val, "test": test,
            "dropped": n_before - len(val) - len(test)}


def group_kfold_split(y, groups, *, n_splits=5, fold=0, seed=0):
    """Train/val/test split from a StratifiedGroupKFold (test = fold, val = fold + 1)."""
    from sklearn.model_selection import StratifiedGroupKFold

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                     .split(np.zeros(len(y)), y, groups))
    test = folds[fold][1]
    val = folds[(fold + 1) % n_splits][1]
    train = np.setdiff1d(np.arange(len(y)), np.concatenate([test, val]))
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Dataset


class HSIDataset(BaseDataset):
    """Hyperspectral covariance dataset built from a ``fetch_<dataset>()`` Bunch.

    Subclasses call :meth:`_setup` with the Bunch and implement
    :meth:`_load_raw` (preprocessed pixels of one sample) when they provide
    ``return_raw``.

    The dataset holds every sample of the fetched data; ``dataset.splits``
    maps split names to ``{"train", "val", "test"}`` indices into it.
    ``split`` names the split used for the normalisation scale and the
    balancing (``"recommended"`` by default); ``subset`` (``"train"``,
    ``"val"``, ``"test"``) optionally restricts the dataset to one part of it
    (``splits`` is then emptied).
    """

    COARSE_MAP = None

    def _setup(self, bunch, split="recommended", subset=None):
        self.bunch = bunch
        self.meta_all = bunch.meta
        self.info = bunch.get("info", {})
        labels = np.asarray(bunch.labels)
        index = np.arange(len(labels))

        if self.task == "regression":
            targets = bunch.get("targets")
            if targets is None:
                raise ValueError(f"{type(self).__name__} has no regression target")
            if self.drop_classes or self.merge_classes:
                raise ValueError("drop_classes / merge_classes do not apply to regression")
            y = np.asarray(targets, dtype=np.float64)
            self.classes = list(bunch.get("target_names", []))
        else:
            y, keep, names = apply_task(labels, task=self.task, coarse_map=self.COARSE_MAP,
                                        drop_classes=self.drop_classes,
                                        merge_classes=self.merge_classes)
            index = index[keep]
            selected = self._limit_classes(names)
            in_sel = np.isin(y, [names.index(c) for c in selected])
            remap = {names.index(c): i for i, c in enumerate(selected)}
            index, y = index[in_sel], np.array([remap[v] for v in y[in_sel]], dtype=np.int64)
            self.classes = list(selected)
            if self.max_samples_per_class:
                chosen = []
                for c in range(len(self.classes)):
                    members = np.flatnonzero(y == c)
                    if len(members) > self.max_samples_per_class:
                        members = np.sort(self.rng.choice(members, self.max_samples_per_class,
                                                          replace=False))
                    chosen.append(members)
                chosen = np.sort(np.concatenate(chosen)) if chosen else np.zeros(0, int)
                index, y = index[chosen], y[chosen]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        position = np.full(len(labels), -1)
        position[index] = np.arange(len(index))
        splits = {}
        for name, parts in bunch.splits.items():
            splits[name] = {}
            for part, idx in parts.items():
                new = position[np.asarray(idx, dtype=np.int64)]
                splits[name][part] = new[new >= 0]
        groups = np.asarray(bunch.groups)[index] if bunch.get("groups") is not None else None

        if split not in splits:
            raise ValueError(f"Unknown split {split!r}; available: {sorted(splits)}")
        self.split = split
        self.class_weights = None
        if self.balance in ("undersample", "oversample"):
            for parts in splits.values():
                train = parts["train"]
                parts["train"] = train[balance_indices(y[train], self.balance, seed=self.seed)]
        elif self.balance == "weights":
            self.class_weights = class_weights(y[splits[split]["train"]])
        elif self.balance is not None:
            raise ValueError("balance must be None, 'undersample', 'oversample' or 'weights'")

        covs = bunch.covariances
        train_idx = np.unique(splits[split]["train"])
        self.scale = (geometric_scale(np.asarray(covs[index[train_idx]]))
                      if self.normalize and len(train_idx) else 1.0)

        if subset is not None:
            if subset not in splits[split]:
                raise ValueError(f"subset must be one of {sorted(splits[split])}")
            sel = np.asarray(splits[split][subset])
            index, y = index[sel], y[sel]
            groups = groups[sel] if groups is not None else None
            splits = {}
        self.subset = subset
        self._index = index
        self.labels = y
        self.groups = groups
        self.splits = splits
        self.meta = bunch.meta.iloc[index].reset_index(drop=True)
        names = self.classes if self.task != "regression" else None
        self.samples = [
            {"index": int(k),
             "class_idx": int(v) if names is not None else None,
             "class_name": names[int(v)] if names is not None else None}
            for k, v in zip(index, y, strict=True)
        ]
        if self.verbose:
            print(f"{type(self).__name__}: {len(self)} samples, "
                  f"{len(self.classes)} classes, p={covs.shape[-1]}, split={split!r}")

    def _load_raw(self, k):
        """Preprocessed observations of sample ``k`` (row of the fetched data).

        Default: the window stored in the processed stage (``windows.npy``).
        """
        proc = self.bunch.get("processed_path")
        if proc is None or not (Path(proc) / "windows.npy").exists():
            raise ValueError("return_raw needs the processed stage (keep != 'keep_covariance')")
        windows = np.load(Path(proc) / "windows.npy", mmap_mode="r")
        return windows[int(self.meta_all["proc_index"].iloc[k])]

    def __getitem__(self, i):
        k = int(self._index[i])
        cov = torch.from_numpy(np.asarray(self.bunch.covariances[k], dtype=np.float64)
                               * self.scale)
        mean = torch.from_numpy(np.asarray(self.bunch.means[k], dtype=np.float64)) \
            if self.return_mean else None
        raw = torch.from_numpy(np.array(self._load_raw(k), dtype=np.float32)) \
            if self.return_raw else None
        x = self._pack(cov, mean, raw)
        if self.transform is not None:
            x = self.transform(x)
        y = self.labels[i]
        if self.task == "regression":
            return x, torch.as_tensor(y, dtype=torch.float64)
        return x, int(y)

    def _load_data(self):  # data are loaded by _setup
        pass
