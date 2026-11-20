"""Adapter from MOABB (Mother of All BCI Benchmarks) to SPD covariances.

MOABB keeps the datasets, paradigms and evaluations; this sub-package only adds

- :class:`Covariances`: a scikit-learn transformer from epochs
  (n_trials, n_channels, n_times) to SPD covariances, for the pipelines given to
  MOABB evaluations;
- :class:`MOABBCovariances`: a PyTorch dataset of the same covariances, with
  groups and the splits of the MOABB evaluation protocols;
- :class:`MOABBDataset`: the same dataset built from class names, registered as
  ``"moabb"`` in the :class:`~spdnet_datasets.manager.DatasetManager`.

``moabb`` is an optional dependency (``pip install "spdnet-datasets[moabb]"``),
imported only when data are loaded or splits are built.
"""

from .covariances import Covariances, epochs_to_covariances
from .dataset import (
    SPLIT_KINDS,
    MOABBCovariances,
    MOABBDataset,
    moabb_splits,
    stratified_split,
)

__all__ = ["SPLIT_KINDS", "Covariances", "MOABBCovariances", "MOABBDataset",
           "epochs_to_covariances", "moabb_splits", "stratified_split"]
