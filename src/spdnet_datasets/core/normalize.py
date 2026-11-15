"""Geometric normalisation of covariance matrices.

A single positive scalar ``s`` is fitted on the training covariances so that the
geometric mean of their eigenvalues equals 1::

    s = exp(-mean(log(eigenvalues of the training covariances)))

and applied to every split (``s * C``). Fitting on the training set only avoids
leaking test statistics; caches therefore store unscaled covariances.
"""

import numpy as np


def geometric_scale(covs_train: np.ndarray) -> float:
    """Scalar ``s`` such that the eigenvalues of ``s * C`` have geometric mean 1."""
    eig = np.linalg.eigvalsh(np.asarray(covs_train, dtype=np.float64))
    if np.any(eig <= 0):
        raise ValueError("geometric_scale needs SPD matrices (non-positive eigenvalue found)")
    return float(np.exp(-np.mean(np.log(eig))))


def apply_scale(covs: np.ndarray, s: float) -> np.ndarray:
    """Return ``s * covs``."""
    return np.asarray(covs) * s
