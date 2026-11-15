"""Deprecated alias of :mod:`spdnet_datasets.hyperspectral.kaggle_wheat`."""

import warnings

from ..hyperspectral.kaggle_wheat import KaggleWheatDataset, fetch_kaggle_wheat

warnings.warn(
    "spdnet_datasets.real.kaggle_wheat is deprecated; use spdnet_datasets.hyperspectral.kaggle_wheat",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["KaggleWheatDataset", "fetch_kaggle_wheat"]
