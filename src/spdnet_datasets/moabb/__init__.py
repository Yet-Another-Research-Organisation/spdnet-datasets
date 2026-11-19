"""Adapter from MOABB (Mother of All BCI Benchmarks) to SPD covariances.

MOABB keeps the datasets, paradigms and evaluations; this sub-package only adds

- :class:`Covariances`: a scikit-learn transformer from epochs
  (n_trials, n_channels, n_times) to SPD covariances, for the pipelines given to
  MOABB evaluations;
- :class:`MOABBCovariances`: a PyTorch dataset of the same covariances, with
  groups and the splits of the MOABB evaluation protocols.

``moabb`` is an optional dependency (``pip install "spdnet-datasets[moabb]"``),
imported only when data are loaded or splits are built.
"""

from .covariances import Covariances, epochs_to_covariances
from .dataset import SPLIT_KINDS, MOABBCovariances, moabb_splits

__all__ = ["SPLIT_KINDS", "Covariances", "MOABBCovariances", "epochs_to_covariances",
           "moabb_splits"]
