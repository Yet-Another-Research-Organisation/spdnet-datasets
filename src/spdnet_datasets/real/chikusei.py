"""Deprecated alias of :mod:`spdnet_datasets.hyperspectral.chikusei`."""

import warnings

from ..hyperspectral.chikusei import ChikuseiDataset, fetch_chikusei

warnings.warn(
    "spdnet_datasets.real.chikusei is deprecated; use spdnet_datasets.hyperspectral.chikusei",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ChikuseiDataset", "fetch_chikusei"]
