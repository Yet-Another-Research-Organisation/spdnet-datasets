"""Deprecated alias of :mod:`spdnet_datasets.hyperspectral.gsoff`."""

import warnings

from ..hyperspectral.gsoff import GSOFFDataset, GSOffDataset, fetch_gsoff

warnings.warn(
    "spdnet_datasets.real.gsoff is deprecated; use spdnet_datasets.hyperspectral.gsoff",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["GSOFFDataset", "GSOffDataset", "fetch_gsoff"]
