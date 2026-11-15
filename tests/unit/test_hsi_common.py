"""Shared hyperspectral helpers: window extraction, spatial splits, dataset wrapper."""

import numpy as np
import pytest

from spdnet_datasets.core import assert_no_window_overlap
from spdnet_datasets.hyperspectral._common import (
    block_kfold_split,
    purity_grid_windows,
    valid_pixels,
    window_boxes,
)


def _naive_best_offset(gt, w, threshold, c):
    best, best_off = 0, (0, 0)
    for oy in range(w):
        for ox in range(w):
            n = 0
            for r in range(oy, gt.shape[0] - w + 1, w):
                for col in range(ox, gt.shape[1] - w + 1, w):
                    win = gt[r:r + w, col:col + w]
                    lab = win[win > 0]
                    if len(lab) and (lab == c).sum() >= threshold * len(lab):
                        n += 1
            if n > best:
                best, best_off = n, (oy, ox)
    return best, best_off


def test_purity_grid_matches_naive_scan():
    rng = np.random.default_rng(0)
    gt = np.zeros((40, 37), np.uint8)
    gt[2:20, 3:30] = 1
    gt[18:39, 10:36] = 2
    gt[rng.random(gt.shape) < 0.05] = 3
    pos, lab, pur = purity_grid_windows(gt, 5, 0.8)
    for c in (1, 2, 3):
        n, (oy, ox) = _naive_best_offset(gt, 5, 0.8, c)
        sel = pos[lab == c]
        assert len(sel) == n
        if n:
            assert set(sel[:, 0] % 5) == {oy} and set(sel[:, 1] % 5) == {ox}
    assert np.all(pur >= 0.8)
    # windows of one class never overlap each other
    for c in (1, 2):
        b = window_boxes(pos[lab == c], 5)
        for i in range(len(b)):
            others = np.delete(b, i, axis=0)
            assert_no_window_overlap(b[i:i + 1], others)


def test_valid_pixels_drops_zero_and_nan_rows():
    X = np.ones((4, 3))
    X[1] = 0
    X[2, 1] = np.nan
    assert len(valid_pixels(X)) == 2


@pytest.mark.parametrize("fold", [0, 3])
def test_block_kfold_split_has_no_overlap(fold):
    rng = np.random.default_rng(1)
    pos = rng.integers(0, 300, size=(400, 2))
    y = rng.integers(0, 3, size=400)
    sp = block_kfold_split(pos, y, window=15, block=50, fold=fold)
    parts = [sp["train"], sp["val"], sp["test"]]
    assert sum(map(len, parts)) + sp["dropped"] == 400
    assert not set(sp["train"]) & set(sp["test"])
    boxes = window_boxes(pos, 15)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert_no_window_overlap(boxes[parts[a]], boxes[parts[b]])


def test_block_kfold_split_separates_scenes():
    pos = np.array([[0, 0], [0, 0], [100, 100], [100, 100]] * 10)
    scene = np.repeat([0, 1], 20)
    y = np.zeros(40, int)
    sp = block_kfold_split(pos, y, window=15, block=50, n_splits=2, fold=0, scene=scene)
    # identical positions in two scenes are not an overlap
    assert sp["dropped"] == 0
