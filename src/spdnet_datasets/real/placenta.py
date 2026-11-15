"""Deprecated alias of :mod:`spdnet_datasets.hyperspectral.placenta`."""

import warnings

from ..hyperspectral.placenta import PlacentaDataset, fetch_placenta

warnings.warn(
    "spdnet_datasets.real.placenta is deprecated; use spdnet_datasets.hyperspectral.placenta",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PlacentaDataset", "fetch_placenta"]
