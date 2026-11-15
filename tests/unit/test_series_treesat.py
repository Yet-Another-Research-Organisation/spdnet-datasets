"""Tests for the TreeSatAI-TS monthly covariance series (spdnet_datasets.real.series_treesat), on a few real files."""

from pathlib import Path

import numpy as np
import pytest

from spdnet_datasets.real.series_treesat import MIN_PIXELS, treesat_series

ROOT = "/DATA/TreeSatAI-TS"
pytestmark = pytest.mark.skipif(not Path(ROOT).is_dir(), reason="TreeSatAI-TS not available")


@pytest.fixture(scope="module")
def series():
    return treesat_series(ROOT, max_per_class=3, n_classes=2, seed=0)  # 12 series, a few dozen files read


def test_shapes(series):
    data, labels, n_obs, extra = series
    assert data.shape == (12, 12, 10, 10) and data.dtype == np.float64
    assert labels.shape == n_obs.shape == extra["split"].shape == (12,)
    assert set(labels) == {0, 1}
    assert set(extra["split"]) <= {0, 1, 2} and (extra["split"] == 2).sum() == 6


def test_tokens(series):
    data, _, n_obs, _ = series
    missing = (data == np.eye(10)).all((-2, -1))
    assert ((~missing).sum(1) >= 6).all()  # at least 6 valid months per series
    assert np.allclose(data, data.swapaxes(-1, -2))
    assert np.linalg.eigvalsh(data[~missing]).min() > 0
    assert (n_obs >= MIN_PIXELS).all()  # n >= 2p for every valid token (the median is >= the minimum)


def test_patch_missing_months_are_identity():
    from spdnet_datasets.estimator.covariance import EstimateCovariance
    from spdnet_datasets.real.series_treesat import treesat_patch

    estimator = EstimateCovariance("ledoit_wolf", remove_mean=True, exclude_zero=False)
    for path in sorted(Path(ROOT, "sentinel-ts").glob("*.h5"))[:20]:
        covariances, pixels = treesat_patch(str(path), estimator)
        identity = (covariances == np.eye(10)).all((-2, -1))
        assert np.array_equal(identity, pixels < MIN_PIXELS)  # valid token <=> n >= 2p clear pixels
