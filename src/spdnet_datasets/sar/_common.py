"""Small helpers shared by the SAR dataset modules (task selection, splits).

They only combine :mod:`spdnet_datasets.core` functions; every dataset module
keeps its own reading, preprocessing and split logic.
"""

import numpy as np

from ..core.normalize import geometric_scale
from ..core.splits import assert_no_leakage
from ..core.tasks import apply_task, balance_indices, class_weights


def finalize_labels(ds, labels, groups, splits, *, coarse_map=None):
    """Apply task, class selection and balancing to a loaded dataset.

    Parameters
    ----------
    ds : BaseDataset
        Dataset being built (reads ``task``, ``drop_classes``, ``merge_classes``,
        ``balance``, ``seed``, ``max_samples_per_class`` and the class limits).
    labels : array (n,)
        Class names (fine level) of all loaded samples.
    groups : array (n,)
        Group of each sample (site, block, vessel, city...).
    splits : dict
        ``{name: {"train": idx, "val": idx, "test": idx}}`` on the loaded samples.
    coarse_map : dict, optional
        ``{fine_class: coarse_class}`` used by ``task="coarse"``.

    Returns
    -------
    keep : ndarray of int
        Indices (into the loaded samples) of the kept samples, in order. The
        caller subsets its arrays with it. ``ds.targets``, ``ds.classes``,
        ``ds.class_to_idx``, ``ds.groups``, ``ds.splits`` and
        ``ds.class_weights`` are set on the kept samples; the class weights
        are computed on the training part of ``splits[ds.split]`` (default
        ``"recommended"``), ``None`` if that split is absent or its training
        part is empty.
    """
    labels = np.asarray(labels)
    y, mask, names = apply_task(labels, task=ds.task, coarse_map=coarse_map,
                                drop_classes=ds.drop_classes, merge_classes=ds.merge_classes)
    names = list(names)
    selected = ds._limit_classes(names)
    if selected != names:
        sel = np.isin(np.asarray(names)[y], selected)
        full = np.flatnonzero(mask)
        mask = np.zeros(len(labels), bool)
        mask[full[sel]] = True
        names_y = np.asarray(names)[y[sel]]
        y = np.array([selected.index(v) for v in names_y], dtype=np.int64)
        names = selected
    keep = np.flatnonzero(mask)
    if ds.max_samples_per_class:
        rng = np.random.default_rng(ds.seed)
        sub = []
        for c in range(len(names)):
            idx = np.flatnonzero(y == c)
            if len(idx) > ds.max_samples_per_class:
                idx = np.sort(rng.choice(idx, ds.max_samples_per_class, replace=False))
            sub.append(idx)
        sub = np.sort(np.concatenate(sub)) if sub else np.zeros(0, int)
        keep, y = keep[sub], y[sub]

    remap = np.full(len(labels), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    new_splits = {}
    for name, parts in splits.items():
        new_parts = {}
        for part, idx in parts.items():
            idx = remap[np.asarray(idx, dtype=np.int64)]
            new_parts[part] = idx[idx >= 0]
        if ds.balance in ("undersample", "oversample") and len(new_parts.get("train", ())):
            tr = new_parts["train"]
            new_parts["train"] = tr[balance_indices(y[tr], ds.balance, seed=ds.seed)]
        new_splits[name] = new_parts

    ds.targets = y
    ds.classes = names
    ds.class_to_idx = {c: i for i, c in enumerate(names)}
    ds.groups = np.asarray(groups)[keep]
    ds.splits = new_splits
    train = new_splits.get(getattr(ds, "split", "recommended"), {}).get("train", ())
    ds.class_weights = (class_weights(y[train], n_classes=len(names)) if len(train)
                        else None)
    return keep


def check_split_groups(splits, groups):
    """Assert that no group is shared between the parts of every split."""
    groups = np.asarray(groups)
    for parts in splits.values():
        names = [k for k in ("train", "val", "test") if k in parts]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert_no_leakage(groups[parts[a]], groups[parts[b]])


def train_scale(covs, splits, split):
    """Geometric scale fitted on the training part of ``splits[split]``."""
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}; available: {sorted(splits)}")
    train = np.unique(splits[split]["train"])
    if not len(train):
        raise ValueError(f"Split {split!r} has an empty training part")
    return geometric_scale(covs[train])


def group_kfold_splits(y, groups, *, n_splits=5, seed=0, prefix="sgkf"):
    """``StratifiedGroupKFold`` folds with a grouped validation part.

    Fold ``k`` uses fold ``k`` as test; the validation part is one fold of a
    second ``StratifiedGroupKFold`` (``n_splits - 1`` folds) on the remaining
    groups. Returns ``{f"{prefix}{n_splits}_fold{k}": {...}}``.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y, groups = np.asarray(y), np.asarray(groups)
    idx = np.arange(len(y))
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = {}
    for k, (rest, test) in enumerate(outer.split(idx, y, groups)):
        inner = StratifiedGroupKFold(n_splits=max(2, n_splits - 1), shuffle=True,
                                     random_state=seed + 1)
        tr, va = next(inner.split(rest, y[rest], groups[rest]))
        splits[f"{prefix}{n_splits}_fold{k}"] = {
            "train": rest[tr], "val": rest[va], "test": test}
    return splits
