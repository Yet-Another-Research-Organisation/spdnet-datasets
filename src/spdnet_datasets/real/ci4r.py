"""Deprecated alias of :mod:`spdnet_datasets.radar_microdoppler.ci4r`.

The loader now reads the raw data (no ``cov/`` directory under the
read-only dataset copy) and caches its covariances in ``SPDNET_CACHE_DIR``.
"""

import warnings

from ..radar_microdoppler.ci4r import CI4RDataset, fetch_ci4r

warnings.warn(
    "spdnet_datasets.real.ci4r is deprecated; use spdnet_datasets.radar_microdoppler.ci4r",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["CI4RDataset", "fetch_ci4r"]
