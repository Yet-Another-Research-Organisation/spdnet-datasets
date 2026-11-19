"""CovarianceSeries around a fake loader: tokens, normalisation, splits, cache, collate."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from spdnet_datasets.core.normalize import geometric_scale
from spdnet_datasets.manager import DatasetManager
from spdnet_datasets.series import CovarianceSeries, collate_series
from spdnet_datasets.series import dataset as series_dataset
from spdnet_datasets.series.modes import SERIES_MODES, _spectrogram_modes

N, T, P = 12, 60, 4


class FakeLoader:
    """Minimal loader: 12 samples (T, 4), 3 groups of 4, split by group."""

    split, subset, task = "recommended", None, "fine"
    classes = ["a", "b"]

    def __init__(self, data_dir="fake", lengths=None, test_shift=0.0, constant=()):
        self.data_dir = data_dir
        self.lengths = lengths or [T] * N
        self.test_shift = test_shift
        self.constant = set(constant)
        self.labels = np.arange(N) % 2
        self.groups = np.repeat([0, 1, 2], 4)
        idx = np.arange(N)
        self.splits = {"recommended": {"train": idx[:4], "val": idx[4:8], "test": idx[8:]}}
        self._index = idx
        self.calls = 0

    def __len__(self):
        return N

    def structured_raw(self, i):
        self.calls += 1
        rng = np.random.default_rng(i)
        x = rng.standard_normal((self.lengths[i], P)) * (1 + i)
        if i >= 8:
            x = x * (1 + self.test_shift)
        if i in self.constant:
            x[:20] = 1.0  # first token constant: not SPD
        return x


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setitem(DatasetManager.DATASETS, "fake_series", FakeLoader)
    monkeypatch.setitem(SERIES_MODES, "fake_series",
                        _spectrogram_modes((T, P), window=20, step=10, n_channels=None))

    def make(loader=None, **kw):
        kw.setdefault("cache_dir", tmp_path)
        kw.setdefault("verbose", False)
        return CovarianceSeries(loader if loader is not None else FakeLoader(), **kw)
    return make


def test_shapes_and_tokens_equal_covariance_of_observations(fake):
    s = fake(normalize=False)
    (tokens, mask), y = s[3]
    assert tokens.shape == (5, P, P) and tokens.dtype == torch.float64
    assert mask.all() and y == 1
    obs, pos = s.tokenize(3)
    assert [p[0] for p in pos] == [(0, 20), (10, 30), (20, 40), (30, 50), (40, 60)]
    for k, o in enumerate(obs):
        np.testing.assert_allclose(tokens[k].numpy(), np.cov(o.T, bias=True), atol=1e-12)
    d = s.describe()
    assert (d["S"], d["p"], d["n_t"], d["n_t_over_p"]) == (5, P, 20, 5.0)
    assert d["spd_fraction"] == 1.0 and d["mode"] == "temporal"


def test_scale_fitted_on_train_tokens_only(fake, tmp_path):
    a = fake()
    b = fake(FakeLoader(test_shift=10.0), cache_dir=tmp_path / "b")
    assert a.scale == pytest.approx(b.scale)
    train = a.splits["recommended"]["train"]
    raw = fake(normalize=False)
    ref = geometric_scale(np.concatenate([raw[i][0][0].numpy() for i in train]))
    assert a.scale == pytest.approx(ref)
    np.testing.assert_allclose(a[9][0][0].numpy(), raw[9][0][0].numpy() * a.scale)


def test_means_scaled_by_sqrt_s(fake):
    s = fake(return_mean=True)
    (tokens, mask, mean), _ = s[2]
    obs, _ = s.tokenize(2)
    np.testing.assert_allclose(mean.numpy(), np.stack([o.mean(0) for o in obs])
                               * np.sqrt(s.scale))


def test_splits_and_groups_inherited_no_leakage(fake):
    loader = FakeLoader()
    s = fake(loader)
    for part in ("train", "val", "test"):
        np.testing.assert_array_equal(s.splits["recommended"][part],
                                      loader.splits["recommended"][part])
    sp = s.splits["recommended"]
    assert not set(s.groups[sp["train"]]) & set(s.groups[sp["test"]])
    test = fake(subset="test")
    assert len(test) == 4 and test.splits == {}
    np.testing.assert_array_equal(test.labels, loader.labels[8:])
    assert test.scale == pytest.approx(s.scale)


def test_cache_is_replayed(fake, tmp_path, monkeypatch):
    first = fake()
    calls = []
    original = CovarianceSeries._compute
    monkeypatch.setattr(CovarianceSeries, "_compute",
                        lambda self, out: calls.append(1) or original(self, out))
    again = fake()
    assert calls == [] and again.cache_path == first.cache_path
    np.testing.assert_array_equal(again[5][0][0].numpy(), first[5][0][0].numpy())
    assert "series" in first.cache_path.parts and "fake_series" in first.cache_path.parts
    other = fake(window=30, step=15)  # another parameter: another cache entry
    assert calls == [1] and other.cache_path != first.cache_path
    fake(FakeLoader(data_dir="other"))  # another source: another cache entry
    assert calls == [1, 1]


def test_overridden_parameters_and_modes(fake):
    s = fake(window=30, step=30)
    assert s.describe()["S"] == 2
    s = fake(mode="spectral", band_window=2, band_step=1)
    (tokens, _), _ = s[0]
    assert tokens.shape == (3, 2, 2)
    with pytest.raises(ValueError, match="no parameter"):
        fake(windw=3)


def test_small_n_raises_unless_allowed(fake):
    with pytest.raises(ValueError, match="fewer than 2p"):
        fake(window=6, step=6)
    s = fake(window=6, step=6, allow_small_n=True)
    assert s.estimator == "ledoit_wolf" and s.describe()["spd_fraction"] == 1.0


def test_degenerate_token_is_masked_identity_and_left_out_of_the_scale(fake):
    s = fake(FakeLoader(constant=(1, 9)), normalize=True)
    (tokens, mask), _ = s[1]
    assert not mask[0] and mask[1:].all()
    np.testing.assert_array_equal(tokens[0].numpy(), np.eye(P))
    assert s.describe()["spd_fraction"] == pytest.approx(1 - 2 / (N * 5))
    ref = fake(FakeLoader(constant=(1, 9)), normalize=False)
    train = [ref[i][0] for i in (0, 1, 2, 3)]
    ok = np.concatenate([t.numpy()[m.numpy()] for t, m in train])
    assert s.scale == pytest.approx(geometric_scale(ok))


def test_variable_lengths_and_collate(fake):
    lengths = [60, 40, 60, 30, 60, 60, 60, 60, 60, 60, 60, 60]
    s = fake(FakeLoader(lengths=lengths), return_mean=True)
    assert s[1][0][0].shape[0] == 3 and s[3][0][0].shape[0] == 2
    loader = DataLoader(s, batch_size=4, collate_fn=collate_series)
    (tokens, mask, mean), y = next(iter(loader))
    assert tokens.shape == (4, 5, P, P) and mask.shape == (4, 5) and mean.shape == (4, 5, P)
    assert mask[1].tolist() == [True, True, True, False, False]
    np.testing.assert_array_equal(tokens[3, 4].numpy(), np.eye(P))
    assert y.tolist() == [0, 1, 0, 1]


def test_loader_checks(fake, monkeypatch):
    loader = FakeLoader()
    loader.subset = "train"
    with pytest.raises(ValueError, match="subset=None"):
        fake(loader)
    with pytest.raises(ValueError, match="loader_kwargs only apply"):
        fake(FakeLoader(), loader_kwargs={"a": 1})
    loader = FakeLoader()
    loader.feature_kwargs = {"features": "mfcc"}
    monkeypatch.setitem(SERIES_MODES, "fake_series", _spectrogram_modes(
        (T, P), window=20, step=10, n_channels=None, loader_spectral={"features": "logmel"}))
    with pytest.raises(ValueError, match="features='logmel'"):
        fake(loader, mode="spectral")


def test_build_from_name(fake, monkeypatch):
    monkeypatch.setattr(series_dataset, "_build_loader",
                        lambda name, mode, kw: FakeLoader(**(kw or {})))
    s = fake("fake_series", loader_kwargs={"data_dir": "x"})
    assert s.name == "fake_series" and s.loader.data_dir == "x"


def test_standardisation_fitted_on_train_and_prepare(fake, monkeypatch, tmp_path):
    from dataclasses import replace

    base = _spectrogram_modes((T, P), window=20, step=10, n_channels=None)["temporal"]
    mode = replace(base, standardize="last", prepare=lambda x: 2 * x)
    monkeypatch.setitem(SERIES_MODES, "fake_series", {"temporal": mode})
    s = fake(normalize=False)
    b = fake(FakeLoader(test_shift=5.0), normalize=False, cache_dir=tmp_path / "b")
    train = np.concatenate([2 * FakeLoader().structured_raw(i) for i in range(4)])
    mu, sd = train.mean(0), train.std(0)
    np.testing.assert_allclose(s._standardization[0], mu, atol=1e-10)
    np.testing.assert_allclose(s._standardization[1], sd, rtol=1e-6)
    np.testing.assert_allclose(b._standardization[1], sd, rtol=1e-6)  # test data not used
    x = (2 * FakeLoader().structured_raw(9) - mu) / sd
    (tokens, _), _ = s[9]
    np.testing.assert_allclose(tokens[0].numpy(), np.cov(x[:20].T, bias=True), rtol=1e-5)


def test_fractional_windows_follow_the_object_size():
    from spdnet_datasets.series import tokenizers as tok

    assert tok.axis_windows(21, 0.5, 0.5) == [(0, 11), (10, 21)]
    obs, _ = tok.spatial(np.zeros((20, 16, 3)), (0.5, 0.5), (0.5, 0.5))
    assert len(obs) == 4 and obs[0].shape == (80, 3)
