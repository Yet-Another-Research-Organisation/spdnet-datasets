"""Leakage-free splits and leakage checks (see ADR-0005).

Samples carry a group (subject, session, patch, spatial block, parcel, site,
...). Splits never separate a group; spatial splits additionally drop test
windows that overlap a training window.
"""

import numpy as np


def group_split(y, groups, *, val_size=0.1, test_size=0.2, seed=0):
    """Split indices into train/val/test without sharing a group.

    Uses :class:`~sklearn.model_selection.StratifiedGroupKFold` so that the class
    proportions stay close to the global ones; the fold sizes approximate
    ``test_size`` and ``val_size``.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y, groups = np.asarray(y), np.asarray(groups)
    idx = np.arange(len(y))

    def _cut(sub, frac, rs):
        k = max(2, int(round(1 / frac)))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=rs)
        a, b = next(sgkf.split(sub, y[sub], groups[sub]))
        return sub[a], sub[b]

    rest, test = _cut(idx, test_size, seed)
    if val_size:
        train, val = _cut(rest, val_size / (1 - test_size), seed + 1)
    else:
        train, val = rest, idx[:0]
    return {"train": train, "val": val, "test": test}


def spatial_block_split(coords, *, block, buffer=0, y=None, n_splits=5, fold=0, seed=0,
                        window=None):
    """Fold ``fold`` of a grouped K-fold on square spatial blocks.

    Parameters
    ----------
    coords : array (n, 2)
        Row/column (or easting/northing) of each sample's top-left corner.
    block : float
        Side of the blocks used as groups.
    buffer : float
        Test samples closer than ``buffer`` (Chebyshev distance) to a training
        sample are dropped from the test set.
    window : float, optional
        Window side; with ``buffer=0`` it is used as buffer so that test
        windows never overlap training windows.
    """
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    coords = np.asarray(coords, dtype=float)
    groups = np.floor(coords[:, 0] / block).astype(np.int64) * 1_000_003 + np.floor(
        coords[:, 1] / block).astype(np.int64)
    if y is None:
        splitter = GroupKFold(n_splits=n_splits)
        folds = list(splitter.split(coords, groups=groups))
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(coords, y, groups))
    train, test = folds[fold]
    gap = buffer or (window or 0)
    if gap and len(train) and len(test):
        from scipy.spatial import cKDTree

        tree = cKDTree(coords[train])
        d, _ = tree.query(coords[test], p=np.inf)
        test = test[d >= gap]
    return {"train": train, "test": test, "groups": groups}


def assert_no_leakage(groups_a, groups_b, what="group"):
    """Raise ``AssertionError`` if a group appears on both sides."""
    common = set(np.asarray(groups_a).tolist()) & set(np.asarray(groups_b).tolist())
    if common:
        sample = sorted(common, key=str)[:5]
        raise AssertionError(f"{len(common)} {what}(s) shared between splits, e.g. {sample}")


def assert_no_window_overlap(boxes_a, boxes_b):
    """Raise if a box of ``boxes_a`` intersects a box of ``boxes_b``.

    Boxes are ``(r0, c0, r1, c1)`` with exclusive ends, in the same image frame
    (add an image identifier offset beforehand if needed).
    """
    a, b = np.asarray(boxes_a, float), np.asarray(boxes_b, float)
    if not len(a) or not len(b):
        return
    for box in a:
        hit = (b[:, 0] < box[2]) & (box[0] < b[:, 2]) & (b[:, 1] < box[3]) & (box[1] < b[:, 3])
        if hit.any():
            raise AssertionError(f"window {box.tolist()} overlaps {b[hit][0].tolist()}")


def assert_no_duplicates(hash_a, hash_b):
    """Raise if the same content hash appears on both sides."""
    assert_no_leakage(hash_a, hash_b, what="duplicate sample")
