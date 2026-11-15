"""Deprecated alias of :mod:`spdnet_datasets.hyperspectral.deephsfruit`."""

import warnings

from ..hyperspectral.deephsfruit import DeepHSFruitDataset, fetch_deephsfruit

warnings.warn(
    "spdnet_datasets.real.deephsfruit is deprecated; use spdnet_datasets.hyperspectral.deephsfruit",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DeepHSFruitDataset", "fetch_deephsfruit"]
