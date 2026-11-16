"""Scikit-learn transformer from EEG epochs to SPD covariance matrices."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..core.estimators import estimate_covariances
from ..core.normalize import apply_scale, geometric_scale


def epochs_to_covariances(X, estimator="scm", assume_centered=False):
    """Covariances of epochs ``X`` (n_trials, n_channels, n_times).

    Each epoch is transposed to ``(n_times, n_channels)`` observations and passed
    to :func:`spdnet_datasets.core.estimators.estimate_covariance`, which raises
    :class:`~spdnet_datasets.core.estimators.InsufficientSamplesError` when
    ``n_times < 2 * n_channels`` and checks that every result is SPD.

    Returns
    -------
    ndarray of shape (n_trials, n_channels, n_channels), float64
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected epochs of shape (n_trials, n_channels, n_times), got {X.shape}")
    return estimate_covariances(np.swapaxes(X, 1, 2), estimator,
                                assume_centered=assume_centered)


class Covariances(TransformerMixin, BaseEstimator):
    """Estimate one SPD covariance per epoch, for MOABB pipelines.

    Drop-in first step of a pipeline given to MOABB evaluations
    (``WithinSessionEvaluation``, ``CrossSessionEvaluation``,
    ``CrossSubjectEvaluation``). Unlike :class:`pyriemann.estimation.Covariances`,
    it uses the estimators of :mod:`spdnet_datasets.core.estimators` with their
    ``n >= 2p`` and SPD checks.

    Parameters
    ----------
    estimator : {"scm", "ledoit_wolf", "tyler", "student"}
        Covariance estimator.
    assume_centered : bool
        If False, the per-channel mean of each epoch is removed first.
    normalize : bool
        If True, :meth:`fit` learns the geometric scale ``s`` of the training
        covariances (:func:`spdnet_datasets.core.normalize.geometric_scale`) and
        :meth:`transform` returns ``s * C``. The scale is fitted on the training
        fold only, so it does not leak test statistics.

    Attributes
    ----------
    scale_ : float
        Fitted scale (1.0 when ``normalize=False``).
    """

    def __init__(self, estimator="scm", assume_centered=False, normalize=False):
        self.estimator = estimator
        self.assume_centered = assume_centered
        self.normalize = normalize

    def _covs(self, X):
        return epochs_to_covariances(X, self.estimator, self.assume_centered)

    def fit(self, X, y=None):
        """Fit the geometric scale on ``X`` (n_trials, n_channels, n_times) if ``normalize``."""
        self._fit_scale(self._covs(X))
        return self

    def _fit_scale(self, covs):
        self.scale_ = geometric_scale(covs) if self.normalize else 1.0
        return covs

    def transform(self, X):
        """Return the covariances (n_trials, n_channels, n_channels), float64."""
        check_is_fitted(self, "scale_")
        return apply_scale(self._covs(X), self.scale_)

    def fit_transform(self, X, y=None):
        """Fit and transform, estimating the covariances only once."""
        return apply_scale(self._fit_scale(self._covs(X)), self.scale_)
