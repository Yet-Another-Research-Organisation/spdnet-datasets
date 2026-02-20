"""Utility functions and preprocessing scripts for dataset preparation.

Preprocessing scripts (run as standalone or via CLI):
    - chikusei_preprocess: Extract windows from Chikusei hyperspectral .mat files
    - gsoff_preprocess: Extract windows from GSOFF GeoTIFF imagery
    - uav_preprocess: Extract windows from UAV-HSI-Crop patches
    - placenta_preprocess: Extract windows from hyperspectral Placenta TIFF files

Covariance precomputation:
    - base_covariance_precompute: Base class for covariance matrix precomputation
    - covariance_precompute: Unified precomputation for HyperLeaf, Rices90, UAV,
      GSOFF, Chikusei, Placenta, and DeepHS-Fruit datasets
    - kaggle_wheat_covariance_precompute: Specialized precomputation for Kaggle
      Wheat Disease HS dataset (107x107 covariance matrices)

Placenta utilities (placenta_utils sub-package):
    - spectral_tiffs: Read/write spectral TIFF image cubes and mask bitmaps
    - annotations: Parse and render annotation CSV files as bitmap masks
"""

__all__ = []
