"""BaseDataset output contract and named splits in create_dataloaders."""

import numpy as np
import pytest

from spdnet_datasets.base import BaseDataset, create_dataloaders


class _Toy(BaseDataset):
    def __init__(self, data_dir, **kw):
        super().__init__(data_dir, verbose=False, **kw)
        self._load_data()

    def _load_data(self):
        rng = np.random.default_rng(0)
        self.samples = [rng.standard_normal((20, 3)) for _ in range(20)]
        self.labels = np.repeat([0, 1], 10)
        self.classes = ["a", "b"]
        self.groups = np.repeat(np.arange(10), 2)
        self.splits = {"official": {"train": np.arange(12), "val": np.arange(12, 16),
                                    "test": np.arange(16, 20)}}

    def __getitem__(self, i):
        X = self.samples[i]
        x = self._pack(np.cov(X.T), X.mean(0), X)
        return x, int(self.labels[i])


class _RawOnly(_Toy):
    OUTPUTS = ("raw", "mean")


def test_default_output_is_covariance(tmp_path):
    x, y = _Toy(tmp_path)[0]
    assert x.shape == (3, 3) and y == 0


def test_optional_outputs_are_tuples(tmp_path):
    cov, mean, raw = _Toy(tmp_path, return_mean=True, return_raw=True)[0][0]
    assert cov.shape == (3, 3) and mean.shape == (3,) and raw.shape == (20, 3)


def test_raw_only_dataset_requires_a_choice(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        _RawOnly(tmp_path)
    assert _RawOnly(tmp_path, return_raw=True)[0][0].shape == (20, 3)


def test_named_split_is_used(tmp_path):
    ds = _Toy(tmp_path)
    tr, va, te = create_dataloaders(ds, batch_size=4, num_workers=0, split="official",
                                    verbose=False)
    assert (len(tr.dataset), len(va.dataset), len(te.dataset)) == (12, 4, 4)
    with pytest.raises(ValueError, match="Unknown split"):
        create_dataloaders(ds, num_workers=0, split="nope", verbose=False)


def test_sample_level_split_warns_when_groups_exist(tmp_path):
    with pytest.warns(UserWarning, match="groups"):
        create_dataloaders(_Toy(tmp_path), num_workers=0, verbose=False)


def test_max_classes_uses_dataset_seed(tmp_path):
    classes = [f"c{i}" for i in range(10)]
    picks = {s: _Toy(tmp_path, max_classes=3, seed=s)._limit_classes(classes) for s in range(6)}
    assert picks[1] == _Toy(tmp_path, max_classes=3, seed=1)._limit_classes(classes)
    assert len({tuple(v) for v in picks.values()}) > 1  # no longer fixed to seed 42
