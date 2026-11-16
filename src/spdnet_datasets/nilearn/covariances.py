"""Scikit-learn transformer from fMRI region time series to SPD covariances."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from spdnet_datasets.core.estimators import check_spd, estimate_covariance
from spdnet_datasets.core.normalize import apply_scale, geometric_scale


def series_covariance(ts, estimator="scm", allow_small_n=False):
    """Covariance (p, p) of one region time series ``ts`` (T, p).

    With ``allow_small_n=False`` this is
    :func:`spdnet_datasets.core.estimators.estimate_covariance`, which raises
    :class:`~spdnet_datasets.core.estimators.InsufficientSamplesError` when
    ``T < 2p``. With ``allow_small_n=True`` the Ledoit-Wolf shrinkage estimator
    is used whatever ``estimator`` says, the ``T >= 2p`` rule is not enforced,
    and the result is still checked to be SPD.
    """
    if allow_small_n:
        C = estimate_covariance(ts, "ledoit_wolf", check=False)
        check_spd(C)
        return C
    return estimate_covariance(ts, estimator)


class Covariances(TransformerMixin, BaseEstimator):
    """Estimate one SPD covariance per subject from region time series.

    The input is a list of arrays ``(T_i, p)``, one per subject, as returned by
    nilearn maskers or ``fetch_abide_pcp`` (``T_i`` may differ between
    subjects). It plays the role of
    ``nilearn.connectome.ConnectivityMeasure(kind="covariance")`` with the
    estimators and checks of :mod:`spdnet_datasets.core.estimators`.

    Parameters
    ----------
    estimator : {"scm", "ledoit_wolf", "tyler", "student"}
        Covariance estimator.
    allow_small_n : bool
        Use Ledoit-Wolf and skip the ``T >= 2p`` rule (see
        :func:`series_covariance`).
    normalize : bool
        If True, :meth:`fit` learns the geometric scale ``s`` of the training
        covariances and :meth:`transform` returns ``s * C``.

    Attributes
    ----------
    scale_ : float
        Fitted scale (1.0 when ``normalize=False``).
    """

    def __init__(self, estimator="scm", allow_small_n=False, normalize=False):
        self.estimator = estimator
        self.allow_small_n = allow_small_n
        self.normalize = normalize

    def _covs(self, X):
        return np.stack([series_covariance(np.asarray(ts), self.estimator, self.allow_small_n)
                         for ts in X])

    def _fit_scale(self, covs):
        self.scale_ = geometric_scale(covs) if self.normalize else 1.0
        return covs

    def fit(self, X, y=None):
        """Fit the geometric scale on the time series ``X`` if ``normalize``."""
        self._fit_scale(self._covs(X))
        return self

    def transform(self, X):
        """Return the covariances (n_subjects, p, p), float64."""
        check_is_fitted(self, "scale_")
        return apply_scale(self._covs(X), self.scale_)

    def fit_transform(self, X, y=None):
        """Fit and transform, estimating the covariances only once."""
        return apply_scale(self._fit_scale(self._covs(X)), self.scale_)
