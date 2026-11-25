"""core.splits.random_split and the random_split_options of BaseDataset."""

import numpy as np
import pytest

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.core import random_split, shares_groups
from spdnet_datasets.core.splits import _random_split


def _check_partition(parts, n):
    cat = np.concatenate([parts["train"], parts["val"], parts["test"]])
    assert sorted(cat.tolist()) == list(range(n))
    for k in ("train", "val", "test"):
        assert parts[k].dtype == np.int64 and np.all(np.diff(parts[k]) > 0)


def test_sizes_disjointness_and_determinism():
    y = np.repeat(np.arange(4), 25)
    parts = random_split(y, seed=3)
    _check_partition(parts, 100)
    assert [len(parts[k]) for k in ("train", "val", "test")] == [70, 10, 20]
    again = random_split(y, seed=3)
    other = random_split(y, seed=4)
    for k in parts:
        np.testing.assert_array_equal(parts[k], again[k])
    assert not np.array_equal(parts["test"], other["test"])
    parts = random_split(y, val_ratio=0.0, test_ratio=0.5)
    assert len(parts["val"]) == 0 and len(parts["test"]) == 50


def test_stratification_keeps_the_class_proportions():
    y = np.array([0] * 80 + [1] * 20)
    parts, stratified = _random_split(y, seed=0)
    assert stratified
    assert np.bincount(y[parts["test"]]).tolist() == [16, 4]
    assert np.bincount(y[parts["val"]]).tolist() == [8, 2]
    names = np.array(["a"] * 80 + ["b"] * 20)  # class names work too
    np.testing.assert_array_equal(random_split(names, seed=0)["test"], parts["test"])


def test_infeasible_stratification_falls_back_to_a_permutation():
    y = np.array([0] * 30 + [1])  # class 1 has a single sample
    parts, stratified = _random_split(y, seed=0)
    assert not stratified
    _check_partition(parts, 31)
    assert [len(parts[k]) for k in ("val", "test")] == [3, 6]
    parts, stratified = _random_split(np.repeat(np.arange(10), 2), test_ratio=0.2, seed=0)
    assert not stratified  # 4 test samples for 10 classes
    _, stratified = _random_split(np.repeat(np.arange(2), 10), stratify=False)
    assert not stratified


def test_regression_targets_are_shuffled_not_stratified():
    t = np.linspace(0.0, 1.0, 50)
    parts, stratified = _random_split(t, seed=1)
    assert not stratified
    _check_partition(parts, 50)
    two_d = np.random.default_rng(0).random((50, 3))  # several regression targets
    _check_partition(random_split(two_d), 50)


def test_invalid_ratios_are_rejected():
    with pytest.raises(ValueError, match="sum < 1"):
        random_split(np.zeros(10, int), val_ratio=0.5, test_ratio=0.5)
    with pytest.raises(ValueError, match="no training sample"):
        random_split(np.zeros(2, int), val_ratio=0.4, test_ratio=0.4)


def test_shares_groups():
    assert not shares_groups(None)
    assert not shares_groups(np.arange(5))  # one sample per group: not leaky
    assert shares_groups(np.array([0, 0, 1]))


class _Toy(BaseDataset):
    def _load_data(self):
        pass

    def __getitem__(self, i):
        return i


def test_options_default_to_the_dataset_seed(tmp_path):
    ds = _Toy(tmp_path, seed=7)
    assert ds.random_split_options == {"val_ratio": 0.1, "test_ratio": 0.2, "seed": 7,
                                       "stratify": True}
    ds = _Toy(tmp_path, random_split_options={"test_ratio": 0.5, "seed": 1})
    assert ds.random_split_options["test_ratio"] == 0.5 and ds.random_split_options["seed"] == 1
    parts = ds._make_random_split(np.repeat([0, 1], 10), groups=np.repeat(np.arange(10), 2))
    assert len(parts["test"]) == 10 and ds.random_split_stratified and ds.random_split_leaky
    with pytest.raises(ValueError, match="Unknown random_split_options"):
        _Toy(tmp_path, random_split_options={"test_size": 0.3})


def test_selected_random_split_warns_only_when_groups_are_shared(tmp_path, recwarn):
    ds = _Toy(tmp_path)
    ds._check_selected_split("random", np.arange(4))
    ds._check_selected_split("recommended", np.array([0, 0, 1]))
    assert not ds.split_leaky and len(recwarn) == 0
    with pytest.warns(UserWarning, match="split='random'"):
        ds._check_selected_split("random", np.array([0, 0, 1]))
    assert ds.split_leaky
