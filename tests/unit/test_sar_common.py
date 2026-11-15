"""sar._common.finalize_labels: class weights fitted on the training part."""

from types import SimpleNamespace

import numpy as np
import pytest

from spdnet_datasets.sar._common import finalize_labels


def _ds(**kw):
    return SimpleNamespace(task="fine", drop_classes=(), merge_classes=None, balance=None,
                           seed=0, max_samples_per_class=None,
                           _limit_classes=lambda names: names, **kw)


def test_class_weights_use_the_training_part_of_the_split():
    labels = np.array(["a"] * 2 + ["b"] * 6 + ["c"] * 4)
    # train: 2 a + 2 b (balanced, c absent); the whole set is not balanced
    splits = {"recommended": {"train": np.arange(4), "val": np.arange(4, 8),
                              "test": np.arange(8, 12)},
              "other": {"train": np.arange(12), "val": np.zeros(0, int),
                        "test": np.zeros(0, int)}}
    ds = _ds()
    finalize_labels(ds, labels, np.arange(12), splits)
    assert ds.class_weights.tolist() == pytest.approx([1.0, 1.0, 0.0])
    ds = _ds(split="other")
    finalize_labels(ds, labels, np.arange(12), splits)
    assert ds.class_weights[0] / ds.class_weights[1] == pytest.approx(3.0)
    ds = _ds()
    finalize_labels(ds, labels, np.arange(12), {})
    assert ds.class_weights is None
