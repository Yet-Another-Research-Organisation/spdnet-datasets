"""Utility functions and preprocessing scripts for dataset preparation.

Preprocessing scripts (run as standalone or via CLI):
    - placenta_preprocess: Extract windows from hyperspectral Placenta TIFF files
      (the windows of ``imagettes/``, read by ``hyperspectral.placenta``)

The Chikusei, GSOFF, UAV-HSI-Crop and Kaggle wheat scripts were replaced by the
``fetch_<dataset>()`` functions of :mod:`spdnet_datasets.hyperspectral`.

Covariance precomputation:
    - base_covariance_precompute: Base class for covariance matrix precomputation
    - covariance_precompute: Unified precomputation for HyperLeaf, Rices90, UAV,
      GSOFF, Chikusei, Placenta, and DeepHS-Fruit datasets

Placenta utilities (placenta_utils sub-package):
    - spectral_tiffs: Read/write spectral TIFF image cubes and mask bitmaps
    - annotations: Parse and render annotation CSV files as bitmap masks
"""

__all__ = []
