"""Deprecated alias of :mod:`spdnet_datasets.hyperspectral.rices90`."""

import warnings

from ..hyperspectral.rices90 import Rices90Dataset, fetch_rices90

warnings.warn(
    "spdnet_datasets.real.rices90 is deprecated; use spdnet_datasets.hyperspectral.rices90",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Rices90Dataset", "fetch_rices90"]
