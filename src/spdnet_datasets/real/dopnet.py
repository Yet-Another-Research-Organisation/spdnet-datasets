"""Deprecated alias of :mod:`spdnet_datasets.radar_microdoppler.dopnet`.

The loader now reads the raw data (no ``cov/`` directory under the
read-only dataset copy) and caches its covariances in ``SPDNET_CACHE_DIR``.
"""

import warnings

from ..radar_microdoppler.dopnet import DopNetDataset, fetch_dopnet

warnings.warn(
    "spdnet_datasets.real.dopnet is deprecated; use spdnet_datasets.radar_microdoppler.dopnet",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DopNetDataset", "fetch_dopnet"]
