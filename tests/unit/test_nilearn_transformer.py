"""nilearn adapter: the scikit-learn ``Covariances`` transformer (no nilearn needed)."""

import numpy as np
import pytest
from sklearn.base import clone

from spdnet_datasets.core.estimators import InsufficientSamplesError, check_spd
from spdnet_datasets.nilearn import Covariances, series_covariance

P = 5


def _series(n_subjects=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((T, P)) for T in rng.integers(2 * P, 6 * P, n_subjects)]


def test_variable_lengths_and_numpy_scm():
    X = _series()
    C = Covariances().fit_transform(X)
    assert C.shape == (4, P, P) and C.dtype == np.float64
    check_spd(C)
    np.testing.assert_allclose(C, np.stack([np.cov(x.T, bias=True) for x in X]), atol=1e-12)


def test_two_p_rule_and_allow_small_n():
    short = [np.random.default_rng(1).standard_normal((2 * P - 1, P))]
    with pytest.raises(InsufficientSamplesError):
        Covariances().fit(short)
    C = Covariances(estimator="scm", allow_small_n=True).fit_transform(short)
    from sklearn.covariance import LedoitWolf

    np.testing.assert_allclose(C[0], LedoitWolf().fit(short[0]).covariance_, atol=1e-12)
    np.testing.assert_allclose(series_covariance(short[0], allow_small_n=True), C[0])


def test_normalize_fitted_on_train_and_clonable():
    train, test = _series(seed=2), [5 * x for x in _series(seed=3)]
    t = Covariances(normalize=True).fit(train)
    assert abs(np.mean(np.log(np.linalg.eigvalsh(t.transform(train))))) < 1e-10
    np.testing.assert_allclose(t.transform(test), t.scale_ * Covariances().fit_transform(test))
    assert clone(t).get_params()["normalize"] is True
