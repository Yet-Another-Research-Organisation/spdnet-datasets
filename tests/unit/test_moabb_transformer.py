"""MOABB adapter: the scikit-learn ``Covariances`` transformer (no moabb needed)."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from spdnet_datasets.core.estimators import (
    ESTIMATORS,
    InsufficientSamplesError,
    check_spd,
)
from spdnet_datasets.moabb import Covariances, epochs_to_covariances

N_CH, N_T = 4, 40  # n_times = 10 * n_channels


def _two_class_epochs(n_per_class=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((2 * n_per_class, N_CH, N_T))
    X[n_per_class:] *= 3.0
    return X, np.repeat(["left_hand", "right_hand"], n_per_class)


# ---------------------------------------------------------------- transformer
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_transformer_outputs_spd(estimator):
    if estimator in ("tyler", "student"):
        pytest.importorskip("yetanotherspdnet")
    X = np.random.default_rng(1).standard_normal((5, N_CH, N_T))
    C = Covariances(estimator).fit_transform(X)
    assert C.shape == (5, N_CH, N_CH) and C.dtype == np.float64
    check_spd(C)


def test_transformer_matches_numpy_scm():
    X = np.random.default_rng(2).standard_normal((3, N_CH, N_T))
    expected = np.stack([np.cov(x, bias=True) for x in X])
    np.testing.assert_allclose(Covariances().fit(X).transform(X), expected, atol=1e-12)


def test_transformer_enforces_two_p_samples():
    X = np.random.default_rng(3).standard_normal((2, N_CH, 2 * N_CH - 1))
    with pytest.raises(InsufficientSamplesError):
        Covariances().fit(X)
    epochs_to_covariances(np.random.default_rng(3).standard_normal((2, N_CH, 2 * N_CH)))


def test_transformer_normalize_is_fitted_on_train():
    rng = np.random.default_rng(4)
    train, test = rng.standard_normal((6, N_CH, N_T)), 5 * rng.standard_normal((6, N_CH, N_T))
    t = Covariances(normalize=True).fit(train)
    eig = np.linalg.eigvalsh(t.transform(train))
    assert abs(np.mean(np.log(eig))) < 1e-10
    np.testing.assert_allclose(t.transform(test), t.scale_ * epochs_to_covariances(test))
    np.testing.assert_allclose(Covariances(normalize=True).fit_transform(train),
                               t.transform(train))


def test_transformer_is_clonable_and_cross_validates():
    X, y = _two_class_epochs()
    flat = FunctionTransformer(lambda c: np.log(np.diagonal(c, axis1=1, axis2=2)))
    pipe = make_pipeline(Covariances("ledoit_wolf"), flat, LinearDiscriminantAnalysis())
    assert clone(pipe).get_params()["covariances__estimator"] == "ledoit_wolf"
    assert cross_val_score(pipe, X, y, cv=3).mean() > 0.9
