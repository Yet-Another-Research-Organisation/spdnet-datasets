"""Deprecated alias of :mod:`spdnet_datasets.radar_microdoppler.mvdoppler`.

The loader now reads the raw data (no ``cov/`` directory under the
read-only dataset copy) and caches its covariances in ``SPDNET_CACHE_DIR``.
"""

import warnings

from ..radar_microdoppler.mvdoppler import MVDopplerDataset, fetch_mvdoppler

warnings.warn(
    "spdnet_datasets.real.mvdoppler is deprecated; use spdnet_datasets.radar_microdoppler.mvdoppler",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MVDopplerDataset", "fetch_mvdoppler"]
