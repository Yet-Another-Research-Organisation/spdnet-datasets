#!/usr/bin/env python3
"""
Kaggle Wheat Disease (HS) covariance precomputation.

Computes 107x107 covariance matrices from hyperspectral TIFF images.
- Truncates all images to 125 channels (min across dataset)
- Removes the first 8 and last 10 noisy spectral bands -> 107 bands
- Filters out samples that are entirely 0 or 65535
- Scales covariance eigenvalues around 1.0
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import tifffile

from .base_covariance_precompute import CovariancePrecompute


class KaggleWheatCovariancePrecompute(CovariancePrecompute):
    """
    Covariance precomputation for Kaggle Wheat Disease HS dataset.

    Images are TIFF files with shape (H, W, C) where C is 125 or 126.
    Processing pipeline:
      1. Truncate to 125 channels (min across dataset)
      2. Remove the first 8 noisy bands and the last 10 noisy bands
      -> 125 - 8 - 10 = 107 usable bands -> 107x107 covariance matrices.
    """

    def __init__(
        self,
        data_dir: str,
        cov_dir: str | None = None,
        scaling_factor: float | None = None,
        remove_mean: bool = True,
        exclude_zero: bool = True,
        num_workers: int | None = None,
        min_channels: int = 125,
        skip_first: int = 8,
        skip_last: int = 10,
    ):
        """
        Initialize Kaggle Wheat covariance precomputation.

        Args:
            data_dir: Directory containing .tif HS images
            cov_dir: Output directory for covariances (default: data_dir/../cov)
            scaling_factor: Scaling factor (auto-estimated if None)
            remove_mean: Center data before covariance
            exclude_zero: Exclude zero pixels
            num_workers: Parallel workers
            min_channels: Minimum spectral channels to keep (before trimming)
            skip_first: Number of leading noisy bands to remove
            skip_last: Number of trailing noisy bands to remove
        """
        self.min_channels = min_channels
        self.skip_first = skip_first
        self.skip_last = skip_last
        self._filtered_paths: list[Path] | None = None

        super().__init__(
            data_dir=data_dir,
            cov_dir=cov_dir,
            scaling_factor=scaling_factor,
            remove_mean=remove_mean,
            exclude_zero=exclude_zero,
            num_workers=num_workers,
        )

    def _is_valid_image(self, image_path: Path) -> bool:
        """Check whether an image should be kept (not all-0 / all-65535)."""
        try:
            img = tifffile.imread(str(image_path))
            if img.min() == 0 and img.max() == 0:
                return False
            if img.min() == 65535 and img.max() == 65535:
                return False
            return True
        except Exception:
            return False

    def load_image(self, image_path: Path) -> torch.Tensor:
        """
        Load a Kaggle Wheat HS TIFF image.

        The image is:
        1. Loaded as float32
        2. Truncated to ``min_channels`` spectral bands (125)
        3. First ``skip_first`` (8) and last ``skip_last`` (10) noisy
           bands removed -> 107 usable bands

        Args:
            image_path: Path to TIFF file

        Returns:
            Image tensor of shape (H, W, 107)
        """
        image_array = tifffile.imread(str(image_path)).astype(np.float32)

        # image_array shape is (H, W, C)
        if image_array.ndim != 3:
            raise ValueError(
                f"Expected 3D image (H, W, C), got shape {image_array.shape}"
            )

        # Truncate to min_channels (125)
        image_array = image_array[:, :, : self.min_channels]

        # Remove noisy spectral end-bands
        if self.skip_last > 0:
            image_array = image_array[:, :, : -self.skip_last]
        if self.skip_first > 0:
            image_array = image_array[:, :, self.skip_first :]

        return torch.tensor(image_array)

    def get_image_paths(self) -> list[Path]:
        """
        Get all valid TIFF image paths (filters out all-0 / all-65535).

        Returns:
            Sorted list of valid image paths
        """
        if self._filtered_paths is not None:
            return self._filtered_paths

        all_tifs = sorted(list(self.data_dir.glob("*.tif")))

        if len(all_tifs) == 0:
            raise FileNotFoundError(
                f"No .tif files found in {self.data_dir}"
            )

        valid = []
        skipped = 0
        skip_unvalid = False
        for p in all_tifs:
            if self._is_valid_image(p):
                valid.append(p)
            else:
                if skip_unvalid:
                    skipped += 1
                    print(f"  Filtered out (invalid): {p.name}")
                else:
                    valid.append(p)

        print(
            f"Found {len(all_tifs)} images, {skipped} filtered, "
            f"{len(valid)} valid"
        )
        self._filtered_paths = valid
        return valid


def main():
    """CLI for Kaggle Wheat Disease HS covariance precomputation."""
    parser = argparse.ArgumentParser(
        description="Precompute 107x107 covariance matrices for Kaggle Wheat HS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing .tif HS images (e.g. .../Kaggle_Prepared/train/HS)",
    )

    parser.add_argument(
        "--cov_dir",
        type=str,
        default=None,
        help="Output directory for covariances (default: data_dir/../../cov_<split>)",
    )

    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=None,
        help="Scaling factor (auto-estimated if not provided)",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of existing covariances",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify covariances after computation",
    )

    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing covariances",
    )

    args = parser.parse_args()

    precompute = KaggleWheatCovariancePrecompute(
        data_dir=args.data_dir,
        cov_dir=args.cov_dir,
        scaling_factor=args.scaling_factor,
        num_workers=args.num_workers,
        remove_mean=False,
        exclude_zero=False,
    )

    if args.verify_only:
        precompute.verify()
    else:
        precompute.precompute(force=args.force)
        if args.verify:
            precompute.verify()


if __name__ == "__main__":
    main()
