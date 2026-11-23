"""Adapter from nilearn fMRI datasets to SPD connectomes.

nilearn keeps the datasets and their download; this sub-package only adds

- :class:`Covariances`: a scikit-learn transformer from region time series
  ``(T_i, p)`` to SPD covariances, with the ``n >= 2p`` and SPD checks of
  :mod:`spdnet_datasets.core.estimators`;
- :class:`NilearnConnectomes`: a PyTorch dataset of one covariance per subject
  (ABIDE PCP, ADHD-200 sample, development fMRI, NKI enhanced surface, OpenNeuro
  ds000030), with sites as groups, site-grouped splits and fold splits;
- :func:`cross_validate_folds`: scores over all folds, with a bootstrap
  interval and a within-site permutation test.

``nilearn`` is an optional dependency (``pip install "spdnet-datasets[nilearn]"``),
imported only when data are loaded. Inside this sub-package, ``import nilearn``
is an absolute import and resolves to the PyPI package, not to this module.
"""

from __future__ import annotations

from spdnet_datasets.nilearn.covariances import Covariances, series_covariance
from spdnet_datasets.nilearn.dataset import (
    ABIDE_ATLASES,
    ATLASES,
    CLASSES,
    DATASETS,
    DEFAULT_ATLAS,
    NilearnConnectomes,
    site_fold_split,
    site_grouped_split,
    stratified_fold_split,
    stratified_split,
)
from spdnet_datasets.nilearn.evaluation import cross_validate_folds

__all__ = ["ABIDE_ATLASES", "ATLASES", "CLASSES", "DATASETS", "DEFAULT_ATLAS", "Covariances",
           "NilearnConnectomes", "cross_validate_folds", "series_covariance",
           "site_fold_split", "site_grouped_split", "stratified_fold_split",
           "stratified_split"]
