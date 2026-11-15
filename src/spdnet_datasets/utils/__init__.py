"""Utility functions and preprocessing scripts for dataset preparation.

Preprocessing scripts (run as standalone or via CLI):
    - placenta_preprocess: Extract windows from hyperspectral Placenta TIFF files
      (the windows of ``imagettes/``, read by ``hyperspectral.placenta``)

The Chikusei, GSOFF, UAV-HSI-Crop and Kaggle wheat scripts were replaced by the
``fetch_<dataset>()`` functions of :mod:`spdnet_datasets.hyperspectral`.

The covariance precomputation scripts (``covariance_precompute``,
``base_covariance_precompute``, ``ci4r`` / ``dopnet`` / ``mvdoppler`` /
``ntu120_covariance_precompute``) wrote ``cov/`` inside the dataset directories;
they were removed: every ``fetch_<dataset>()`` now builds its covariances in
the cache (``SPDNET_CACHE_DIR``).

Placenta utilities (placenta_utils sub-package):
    - spectral_tiffs: Read/write spectral TIFF image cubes and mask bitmaps
    - annotations: Parse and render annotation CSV files as bitmap masks
"""

__all__ = []
