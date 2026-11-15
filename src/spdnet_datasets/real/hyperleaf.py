"""Deprecated alias of :mod:`spdnet_datasets.hyperspectral.hyperleaf`."""

import warnings

from ..hyperspectral.hyperleaf import HyperLeafDataset, fetch_hyperleaf

warnings.warn(
    "spdnet_datasets.real.hyperleaf is deprecated; use spdnet_datasets.hyperspectral.hyperleaf",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HyperLeafDataset", "fetch_hyperleaf"]
