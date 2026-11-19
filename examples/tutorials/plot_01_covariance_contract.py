"""
The covariance contract: n >= 2p, four estimators, SPD checks
=============================================================

Every sample of a dataset is a ``(p, p)`` covariance estimated from ``n``
observations of ``p`` channels. :mod:`spdnet_datasets.core.estimators` enforces
two rules (see :doc:`/guide/principles` and :doc:`/datasets/covariance`):

- at least ``n >= 2p`` observations per covariance;
- the result is symmetric positive definite (SPD) with a condition number
  ``kappa = lambda_max / lambda_min`` below ``1e12``.

This tutorial checks both rules on synthetic data whose true covariance is
known, and compares the four estimators ``scm``, ``ledoit_wolf``, ``tyler`` and
``student`` on Gaussian and heavy-tailed observations.
"""

# %%
# A known covariance
# ------------------
# We draw a random rotation ``U`` and eigenvalues spread over two decades, so
# that the true covariance ``Sigma = U diag(lambda) U^T`` has ``kappa = 100``.

import matplotlib.pyplot as plt
import numpy as np

from spdnet_datasets.core.estimators import (
    ESTIMATORS,
    InsufficientSamplesError,
    NotSPDError,
    check_spd,
    estimate_covariance,
)

rng = np.random.default_rng(0)
p = 8
U, _ = np.linalg.qr(rng.standard_normal((p, p)))
eigenvalues = np.logspace(0, -2, p)
sigma = U @ np.diag(eigenvalues) @ U.T
sqrt_sigma = U @ np.diag(np.sqrt(eigenvalues)) @ U.T
print(f"p = {p}, true condition number = {np.linalg.cond(sigma):.0f}")


def draw(n, heavy_tailed=False):
    """``n`` observations of covariance ``sigma`` (Student-t with 3 dof if heavy-tailed)."""
    z = rng.standard_normal((n, p))
    if heavy_tailed:
        z *= np.sqrt(1 / rng.gamma(1.5, 1 / 1.5, size=(n, 1)))  # t_3, up to a scale
    return z @ sqrt_sigma


# %%
# The ``n >= 2p`` rule
# --------------------
# With ``check=True`` (the default), too few observations raise
# :class:`~spdnet_datasets.core.estimators.InsufficientSamplesError` before any
# estimation.

try:
    estimate_covariance(draw(2 * p - 1))
except InsufficientSamplesError as err:
    print("InsufficientSamplesError:", err)

C = estimate_covariance(draw(2 * p))
print("n = 2p accepted, shape", C.shape, C.dtype)

# %%
# The SPD check
# -------------
# :func:`~spdnet_datasets.core.estimators.check_spd` rejects a matrix that is
# not symmetric, has a non-positive eigenvalue, or is numerically singular.
# With ``n < p`` the sample covariance has rank ``n - 1`` and is rejected.

for name, M in [
    ("rank-deficient SCM (n = p / 2)", estimate_covariance(draw(p // 2), check=False)),
    ("non-symmetric matrix", sigma + 1e-3 * np.triu(np.ones((p, p)), 1)),
]:
    try:
        check_spd(M)
    except NotSPDError as err:
        print(f"{name}: NotSPDError: {err}")
check_spd(sigma)
print("true covariance: SPD")

# %%
# Four estimators
# ---------------
# For each sample size we estimate 20 covariances with each estimator and record
# their condition number and their error on the *shape* of ``Sigma`` (Tyler's
# estimator is only defined up to a scale, so every matrix is normalised to
# trace ``p`` before comparing). The two M-estimators come from
# ``yetanotherspdnet`` (``pip install 'spdnet-datasets[robust]'``).

sizes = [2 * p, 4 * p, 8 * p, 32 * p]
n_repeats = 20


def shape(M):
    return p * M / np.trace(M)


def run(heavy_tailed):
    kappa = {e: np.zeros(len(sizes)) for e in ESTIMATORS}
    error = {e: np.zeros(len(sizes)) for e in ESTIMATORS}
    for j, n in enumerate(sizes):
        for _ in range(n_repeats):
            X = draw(n, heavy_tailed)
            for e in ESTIMATORS:
                C = estimate_covariance(X, e)  # raises if C is not SPD
                kappa[e][j] += np.linalg.cond(C) / n_repeats
                err = np.linalg.norm(shape(C) - shape(sigma)) / np.linalg.norm(shape(sigma))
                error[e][j] += err / n_repeats
    return kappa, error


results = {"Gaussian": run(False), "Student-t, 3 dof": run(True)}
for law, (kappa, error) in results.items():
    print(f"\n{law}: mean condition number / relative shape error at n = {sizes[-1]}")
    for e in ESTIMATORS:
        print(f"  {e:12s} kappa = {kappa[e][-1]:7.1f}   error = {error[e][-1]:.3f}")

# %%
# Condition number and error against ``n``
# ----------------------------------------
# Small samples inflate the condition number of the SCM; Ledoit-Wolf shrinkage
# keeps it low at the cost of a bias. On heavy-tailed data the M-estimators
# (Tyler, Student-t) recover the shape much better than the SCM.

COLORS = dict(zip(ESTIMATORS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], strict=True))
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, layout="constrained")
for col, (law, (kappa, error)) in enumerate(results.items()):
    for e in ESTIMATORS:
        style = {"color": COLORS[e], "lw": 2, "marker": "o", "ms": 5, "label": e}
        axes[0, col].plot(sizes, kappa[e], **style)
        axes[1, col].plot(sizes, error[e], **style)
    axes[0, col].axhline(np.linalg.cond(sigma), color="#52514e", lw=1, ls="--",
                         label="true kappa")
    axes[0, col].set_title(law, loc="left", fontsize=11)
    axes[1, col].set_xlabel("observations n")
for ax in axes.flat:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(color="#e7e6e2", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].set_ylabel("condition number kappa")
axes[1, 0].set_ylabel("relative shape error")
axes[0, 0].legend(frameon=False, fontsize=9)
plt.show()

# %%
# Takeaways
# ---------
# - The ``n >= 2p`` floor is a hard check, not a warning: a dataset whose
#   windows are too small fails at construction time (and counts the rejected
#   samples, see the dataset pages).
# - ``ledoit_wolf`` is the safe choice when ``n`` is close to ``2p``;
#   ``tyler`` / ``student`` when the observations are heavy-tailed (SAR,
#   impulsive audio).
# - Datasets expose the choice as ``estimator=...``; the next tutorial shows
#   what a sample then contains.
