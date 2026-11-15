"""core.estimators: SCM, Ledoit-Wolf, Tyler, Student, n >= 2p and SPD checks."""

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from spdnet_datasets.core import (
    ESTIMATORS,
    InsufficientSamplesError,
    NotSPDError,
    channel_mean,
    check_sample_size,
    check_spd,
    estimate_covariance,
    estimate_covariances,
)


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    return rng.standard_normal((200, 6)) @ A + 3.0


def _naive_fixed_point(X, weight, n_iter=2000, tol=1e-12, trace_norm=True):
    """Textbook M-estimator fixed point (reference), optionally trace-normalised to p."""
    n, p = X.shape
    S = np.eye(p)
    for _ in range(n_iter):
        q = np.einsum("ij,jk,ik->i", X, np.linalg.inv(S), X)
        S_new = (X * weight(q, p)[:, None]).T @ X / n
        if trace_norm:
            S_new *= p / np.trace(S_new)
        if np.linalg.norm(S_new - S) / np.linalg.norm(S) < tol:
            return S_new
        S = S_new
    return S


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_every_estimator_returns_spd_float64(X, estimator):
    C = estimate_covariance(X, estimator)
    assert C.shape == (6, 6) and C.dtype == np.float64
    check_spd(C)


def test_scm_matches_numpy_biased_covariance(X):
    np.testing.assert_allclose(estimate_covariance(X, "scm"), np.cov(X.T, bias=True), atol=1e-12)


def test_scm_assume_centered_uses_raw_second_moment(X):
    np.testing.assert_allclose(
        estimate_covariance(X, "scm", assume_centered=True), X.T @ X / len(X), atol=1e-10)


@pytest.mark.parametrize("assume_centered", [False, True])
def test_ledoit_wolf_matches_sklearn(X, assume_centered):
    ref = LedoitWolf(assume_centered=assume_centered).fit(X).covariance_
    np.testing.assert_allclose(
        estimate_covariance(X, "ledoit_wolf", assume_centered=assume_centered), ref, atol=1e-10)


def test_tyler_matches_naive_fixed_point(X):
    Xc = X - X.mean(0)
    ref = _naive_fixed_point(Xc, lambda q, p: p / q)
    np.testing.assert_allclose(estimate_covariance(X, "tyler"), ref, rtol=1e-4, atol=1e-5)


def test_student_matches_naive_fixed_point(X):
    nu = 5.0
    Xc = X - X.mean(0)
    ref = _naive_fixed_point(Xc, lambda q, p: (p + nu) / (nu + q), trace_norm=False)
    np.testing.assert_allclose(estimate_covariance(X, "student", nu=nu), ref, rtol=1e-4, atol=1e-5)


def test_tyler_is_trace_normalised(X):
    assert np.trace(estimate_covariance(X, "tyler")) == pytest.approx(6.0)


def test_tyler_is_robust_to_scale_outliers():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 4))
    X[:20] *= 1e4  # heavy outliers along random directions
    T = estimate_covariance(X, "tyler", assume_centered=True)
    eig = np.linalg.eigvalsh(T)
    assert eig.max() / eig.min() < 3  # close to identity up to scale


def test_sample_size_rule_raises_before_estimating():
    X = np.random.default_rng(2).standard_normal((11, 6))  # 11 < 2 * 6
    with pytest.raises(InsufficientSamplesError, match="2 \\* p = 12"):
        estimate_covariance(X)
    check_sample_size(12, 6)  # exactly 2p is accepted


def test_check_can_be_disabled_for_exploration():
    X = np.random.default_rng(3).standard_normal((8, 6))
    assert estimate_covariance(X, check=False).shape == (6, 6)


def test_check_spd_rejects_singular_and_asymmetric():
    with pytest.raises(NotSPDError, match="positive definite"):
        check_spd(np.diag([1.0, 0.0]))
    with pytest.raises(NotSPDError, match="symmetric"):
        check_spd(np.array([[1.0, 0.5], [0.0, 1.0]]))
    with pytest.raises(NotSPDError, match="NaN"):
        check_spd(np.array([[np.nan, 0.0], [0.0, 1.0]]))


def test_check_spd_accepts_batches():
    check_spd(np.stack([np.eye(3), 2 * np.eye(3)]))
    with pytest.raises(NotSPDError):
        check_spd(np.stack([np.eye(3), -np.eye(3)]))


def test_unknown_estimator_and_bad_shape():
    with pytest.raises(ValueError, match="Unknown estimator"):
        estimate_covariance(np.ones((10, 2)), "oas")
    with pytest.raises(ValueError, match="shape"):
        estimate_covariance(np.ones(10))


def test_batch_and_channel_mean(X):
    covs = estimate_covariances([X, X[:100]], "scm")
    assert covs.shape == (2, 6, 6)
    np.testing.assert_allclose(channel_mean(X), X.mean(0))
