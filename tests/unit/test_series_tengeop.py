"""Tests for the TenGeoP-SARwv covariance-series loader (spdnet_datasets.real.series_tengeop)."""

import os

import numpy as np
import pytest

from spdnet_datasets.real.series_tengeop import tengeop_series

ROOT = "/DATA/TenGeoP-SARwv"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="TenGeoP-SARwv not available")


@pytest.fixture(scope="module")
def series():
    return tengeop_series(ROOT, max_per_class=2)


def test_shapes(series):
    data, labels, n_obs, extra = series
    assert data.shape == (20, 16, 16, 16)  # 10 classes x 2, 4 x 4 grid of 96 x 96 windows, p = 16
    assert data.dtype == np.float64
    assert labels.shape == n_obs.shape == extra["groups"].shape == (20,)


def test_spd(series):
    data = series[0]
    np.testing.assert_allclose(data, data.swapaxes(-1, -2))
    assert np.linalg.eigvalsh(data).min() > 0


def test_observations_and_labels(series):
    data, labels, n_obs, _ = series
    assert (n_obs == 93 ** 2).all() and (n_obs >= 2 * data.shape[-1]).all()
    assert sorted(set(labels.tolist())) == list(range(10))


def test_reproducible(series):
    again = tengeop_series(ROOT, max_per_class=2)
    np.testing.assert_array_equal(series[0], again[0])
