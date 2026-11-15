"""Covariance series of financial returns: tokens, labels, purge between time blocks, next-covariance target."""

import numpy as np
import pytest

from spdnet_datasets.estimator.covariance import EstimateCovariance
from spdnet_datasets.real.series_finance import log_coordinates, volatility_series


def returns(n=3000, p=6, seed=0):
    rng = np.random.default_rng(seed)
    volatility = np.exp(np.convolve(rng.normal(0, 0.3, n), np.ones(50) / 50, mode="same"))  # slow regimes
    return rng.normal(size=(n, p)) * volatility[:, None], volatility


def blocks(n):
    return np.repeat([0, 1, 2], [n // 2, n // 4, n - n // 2 - n // 4])


def test_tokens_labels_and_purge():
    x, _ = returns()
    block = blocks(len(x))
    data, labels, n_obs, extra = volatility_series(x, x.mean(1), block, window=40, stride=20, every=5, horizon=10)
    assert data.shape[1:] == (8, 6, 6) and (np.linalg.eigvalsh(data) > 0).all()
    split = extra["split"]
    assert abs(np.bincount(labels[split == 0]) - (split == 0).sum() / 3).max() <= 1  # terciles of the train block
    # the last token is the window that ends at the sample date
    ends = np.arange(40 + 7 * 20 - 1, len(x) - 10, 5)
    ends = ends[block[ends] == block[ends + 10]]
    d = ends[3]
    window = x[d - 39:d + 1]
    assert np.allclose(np.trace(data[3, -1]), np.trace(np.cov(window.T)), rtol=0.2)
    assert (block[ends + 10] == split).all()  # purge: the horizon stays in the block


def test_correlation_tokens_have_unit_diagonal():
    x, _ = returns()
    data, *_ = volatility_series(x, x.mean(1), blocks(len(x)), 40, 20, 5, 10, correlation=True)
    assert np.allclose(np.diagonal(data, axis1=-2, axis2=-1), 1, atol=1e-4)


def test_log_coordinates_isometric():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5, 4, 4))
    c = a @ a.transpose(0, 2, 1) + np.eye(4)
    w, v = np.linalg.eigh(c)
    log = (v * np.log(w)[:, None, :]) @ v.transpose(0, 2, 1)
    assert np.allclose(np.linalg.norm(log_coordinates(c), axis=-1), np.linalg.norm(log, axis=(-2, -1)))


def test_next_covariance_target_follows_the_tokens():
    x, _ = returns()
    block = blocks(len(x))
    data, targets, _, extra = volatility_series(x, x.mean(1), block, 40, 20, 5, 40, next_covariance=True)
    assert targets.shape == (len(data), 21) and targets.dtype == np.float64
    ends = np.arange(40 + 7 * 20 - 1, len(x) - 40, 5)
    ends = ends[block[ends] == block[ends + 40]]
    # target of sample 0 = coordinates of the window right after its date (no overlap with the last token)
    future = EstimateCovariance("ledoit_wolf", remove_mean=True, exclude_zero=False)(x[ends[0] + 1:ends[0] + 41])
    future = future + 1e-6 * np.trace(future) / 6 * np.eye(6)
    assert np.allclose(log_coordinates(future), targets[0])


def test_relative_series_are_centred():
    x, _ = returns()
    data, targets, _, _ = volatility_series(x, x.mean(1), blocks(len(x)), 40, 20, 5, 40, next_covariance=True,
                                            relative=True)
    # the log-Euclidean mean of the whitened series is the identity up to the non-commutativity of the tokens
    coordinates = log_coordinates(data)
    assert np.linalg.norm(coordinates.mean(1), axis=-1).max() < 0.1 * np.linalg.norm(coordinates, axis=-1).mean()
    assert (np.linalg.eigvalsh(data) > 0).all() and np.isfinite(targets).all()


def test_missing_market_values_are_dropped_not_labelled_high():  # finding 5
    x, _ = returns()
    block = blocks(len(x))
    market = x.mean(1)
    market[1000:1010] = np.nan
    with pytest.warns(UserWarning, match="missing market values"):
        data, labels, _, extra = volatility_series(x, market, block, 40, 20, 5, 10)
    assert np.isfinite(extra["volatility"]).all() and set(labels.tolist()) <= {0, 1, 2}
    full, *_ = volatility_series(x, x.mean(1), block, 40, 20, 5, 10)
    assert 0 < len(full) - len(data) <= 45  # samples whose span or horizon covers the 10 missing days


def test_finance_root_from_environment(monkeypatch, tmp_path):
    from spdnet_datasets.real.series_finance import french_series

    monkeypatch.delenv("SPDNET_FRENCH_ROOT", raising=False)
    with pytest.raises(ValueError, match="SPDNET_FRENCH_ROOT"):
        french_series()
    monkeypatch.setenv("SPDNET_FRENCH_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        french_series()
