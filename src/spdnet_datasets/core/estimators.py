"""Covariance estimators with explicit sample-size and SPD checks.

Observations are given as ``X`` of shape ``(n, p)``: ``n`` observations of ``p``
channels (pixels x bands, time steps x frequencies, ...).

Estimators
----------
- ``scm``: sample covariance matrix, normalised by ``n``.
- ``ledoit_wolf``: :class:`sklearn.covariance.LedoitWolf` shrinkage.
- ``tyler``: Tyler's M-estimator (fixed point), trace normalised to ``p``.
- ``student``: Student-t M-estimator (fixed point, ``nu`` degrees of freedom).
  Unlike Tyler's, its scale is identifiable, so it is not normalised.

The two M-estimators are those of the group's library,
:mod:`yetanotherspdnet.functions.m_estimators` (``tyler_estimator``,
``student_estimator``), run in float64 on the CPU. With ``assume_centered=False`` the data are centred by the (non-robust) sample
mean; with heavy outliers, centre the data robustly beforehand and pass
``assume_centered=True``.
"""

import numpy as np

ESTIMATORS = ("scm", "ledoit_wolf", "tyler", "student")


class InsufficientSamplesError(ValueError):
    """Raised when there are fewer than ``ratio * p`` observations."""


class NotSPDError(ValueError):
    """Raised when an estimated matrix is not symmetric positive definite."""


def check_sample_size(n: int, p: int, ratio: float = 2.0) -> None:
    """Raise :class:`InsufficientSamplesError` if ``n < ratio * p``."""
    if n < ratio * p:
        raise InsufficientSamplesError(
            f"{n} observations for a {p}x{p} covariance: at least {ratio:g} * p = "
            f"{int(np.ceil(ratio * p))} are required."
        )


def check_spd(C: np.ndarray, rtol: float = 1e-8) -> None:
    """Raise :class:`NotSPDError` unless every matrix of ``C`` (..., p, p) is SPD.

    Symmetry: ``|C - C^T|_max <= rtol * |C|_max``. Positivity: the smallest
    eigenvalue is strictly positive.
    """
    C = np.asarray(C)
    if C.ndim < 2 or C.shape[-1] != C.shape[-2]:
        raise NotSPDError(f"Expected square matrices, got shape {C.shape}")
    if not np.all(np.isfinite(C)):
        raise NotSPDError("Matrix contains NaN or infinite values")
    scale = np.max(np.abs(C), axis=(-2, -1))
    asym = np.max(np.abs(C - np.swapaxes(C, -1, -2)), axis=(-2, -1))
    if np.any(asym > rtol * np.maximum(scale, np.finfo(float).tiny)):
        raise NotSPDError(f"Matrix is not symmetric (max asymmetry {asym.max():.3e})")
    lam_min = np.linalg.eigvalsh(C)[..., 0]
    if np.any(lam_min <= 0):
        raise NotSPDError(f"Matrix is not positive definite (min eigenvalue {lam_min.min():.3e})")


def channel_mean(X: np.ndarray) -> np.ndarray:
    """Per-channel mean of ``X`` (n, p) -> (p,) float64."""
    return np.asarray(X, dtype=np.float64).mean(axis=0)


def _m_estimator(X, kind, nu, tol, n_iter_max):
    try:
        import torch
        from yetanotherspdnet.functions.m_estimators import (
            student_estimator,
            tyler_estimator,
        )
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The 'tyler' and 'student' estimators need yetanotherspdnet "
            "(pip install 'spdnet-datasets[robust]')."
        ) from exc

    data = torch.from_numpy(np.ascontiguousarray(X))
    with torch.no_grad():
        if kind == "tyler":
            C = tyler_estimator(data, n_iterations=n_iter_max, tol=tol,
                                assume_centered=True, normalize="trace")
        else:
            C = student_estimator(data, nu=nu, n_iterations=n_iter_max, tol=tol,
                                  assume_centered=True)
    return C.numpy()


def estimate_covariance(
    X: np.ndarray,
    estimator: str = "scm",
    assume_centered: bool = False,
    check: bool = True,
    nu: float = 5.0,
    tol: float = 1e-8,
    n_iter_max: int = 500,
) -> np.ndarray:
    """Estimate one ``(p, p)`` float64 covariance from observations ``X`` (n, p).

    Parameters
    ----------
    X : array of shape (n, p)
    estimator : {"scm", "ledoit_wolf", "tyler", "student"}
    assume_centered : bool
        If False, the per-channel mean is removed first.
    check : bool
        If True, enforce ``n >= 2p`` before estimating and check the result is SPD.
    nu : float
        Degrees of freedom of the Student-t estimator.
    tol, n_iter_max
        Convergence settings of the M-estimators.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected X of shape (n, p), got {X.shape}")
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator {estimator!r}; choose among {ESTIMATORS}")
    n, p = X.shape
    if check:
        check_sample_size(n, p)

    if estimator == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        C = LedoitWolf(assume_centered=assume_centered).fit(X).covariance_
    else:
        Xc = X if assume_centered else X - X.mean(axis=0, keepdims=True)
        if estimator == "scm":
            C = Xc.T @ Xc / n
        else:
            C = _m_estimator(Xc, estimator, nu, tol, n_iter_max)
    C = (C + C.T) / 2
    if check:
        check_spd(C)
    return C


def estimate_covariances(Xs, estimator: str = "scm", **kwargs) -> np.ndarray:
    """Estimate a batch of covariances from a sequence of ``(n_i, p)`` arrays."""
    return np.stack([estimate_covariance(X, estimator, **kwargs) for X in Xs])
