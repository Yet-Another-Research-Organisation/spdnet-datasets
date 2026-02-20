#!/usr/bin/env python3
"""
Unified covariance precomputation for all hyperspectral datasets.

This module provides covariance precomputation classes for:
- HyperLeaf: TIFF images
- Rices90: ENVI files with imagette extraction
- UAV: Pre-extracted windows (.npy format)
- GSOFF: Pre-extracted windows (.npy format)
- Chikusei: Pre-extracted windows (.npy format)
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tifffile
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, rectangle
from spectral.io import envi
from tqdm import tqdm

from .base_covariance_precompute import CovariancePrecompute


# =============================================================================
# HyperLeaf Dataset
# =============================================================================

class HyperLeafCovariancePrecompute(CovariancePrecompute):
    """
    Covariance precomputation for HyperLeaf dataset.

    HyperLeaf images are stored as TIFF files with shape (C, H, W).
    """

    def load_image(self, image_path: Path) -> torch.Tensor:
        """
        Load a HyperLeaf TIFF image.

        Args:
            image_path: Path to TIFF file

        Returns:
            Image tensor of shape (H, W, C)
        """
        # Load TIFF
        image_array = tifffile.imread(str(image_path))

        # Convert to float32
        image_array = image_array.astype(np.float32)

        # Rearrange from (C, H, W) to (H, W, C)
        image_array = np.moveaxis(image_array, 0, -1)

        # Convert to tensor
        return torch.tensor(image_array)

    def get_image_paths(self) -> List[Path]:
        """
        Get all TIFF image paths in the images directory.

        Returns:
            List of image paths
        """
        images_dir = self.data_dir / "images"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        return sorted(list(images_dir.glob("*.tif*")))


# =============================================================================
# Rices90 Dataset
# =============================================================================

class RicesCovariancePrecompute(CovariancePrecompute):
    """
    Covariance precomputation for Rices dataset.

    Rices dataset consists of ENVI hyperspectral files (.hdr).
    This class first extracts imagettes (patches) from the full images,
    then computes covariances on these patches.
    """

    def __init__(
        self,
        data_dir: str,
        imagettes_dir: Optional[str] = None,
        cov_dir: Optional[str] = None,
        scaling_factor: Optional[float] = None,
        remove_mean: bool = True,
        exclude_zero: bool = True,
        num_workers: Optional[int] = None,
        min_area: int = 50,
        margin_h: float = 0.35,
        margin_w: float = 0.15
    ):
        """
        Initialize Rices covariance precomputation.

        Args:
            data_dir: Root directory containing ENVI files
            imagettes_dir: Directory for extracted imagettes (default: data_dir_imagettes)
            cov_dir: Output directory for covariances (default: data_dir_cov)
            scaling_factor: Scaling factor for normalization
            remove_mean: Whether to center data
            exclude_zero: Exclude zero pixels
            num_workers: Number of parallel workers
            min_area: Minimum area for imagette extraction (pixels)
            margin_h: Vertical margin for bbox enlargement
            margin_w: Horizontal margin for bbox enlargement
        """
        # Set directories
        data_path = Path(data_dir).expanduser()
        if imagettes_dir is None:
            imagettes_dir = data_path.parent / f"{data_path.name}_imagettes"
        if cov_dir is None:
            cov_dir = data_path.parent / f"{data_path.name}_cov"

        self.imagettes_dir = Path(imagettes_dir)
        self.imagettes_dir.mkdir(parents=True, exist_ok=True)

        self.min_area = min_area
        self.margin_h = margin_h
        self.margin_w = margin_w

        super().__init__(
            data_dir=data_dir,
            cov_dir=str(cov_dir),
            scaling_factor=scaling_factor,
            remove_mean=remove_mean,
            exclude_zero=exclude_zero,
            num_workers=num_workers
        )

    def extract_imagettes_from_envi(self, hdr_path: Path) -> List[Path]:
        """
        Extract imagettes from an ENVI file.

        Args:
            hdr_path: Path to .hdr file

        Returns:
            List of extracted imagette paths
        """
        try:
            # Load ENVI file
            data = envi.open(str(hdr_path))
            img_full = data.asarray()  # Shape: (H, W, C)

            # Use last band for segmentation
            last_band = img_full[:, :, -1]

            # Normalize
            normalized = last_band.astype(np.float32)

            # Threshold and segment
            thresh = threshold_otsu(normalized)
            bw = closing(normalized > thresh, rectangle(3, 3))
            cleared = clear_border(bw)
            label_image = label(cleared)
            regions = regionprops(label_image)

            # Filter and enlarge bboxes
            valid_bboxes = []
            for region in regions:
                if region.area >= self.min_area:
                    minr, minc, maxr, maxc = region.bbox

                    # Enlarge bbox
                    height = maxr - minr
                    width = maxc - minc

                    minr = max(0, int(minr - self.margin_h * height))
                    maxr = min(img_full.shape[0], int(maxr + self.margin_h * height))
                    minc = max(0, int(minc - self.margin_w * width))
                    maxc = min(img_full.shape[1], int(maxc + self.margin_w * width))

                    valid_bboxes.append((minr, minc, maxr, maxc))

            # Save imagettes
            base_name = hdr_path.stem
            extracted_paths = []

            for i, (minr, minc, maxr, maxc) in enumerate(valid_bboxes):
                imagette = img_full[minr:maxr, minc:maxc, :]
                output_filename = f"{base_name}_{i:03d}.tif"
                output_path = self.imagettes_dir / output_filename

                tifffile.imwrite(str(output_path), imagette.astype(np.uint16))
                extracted_paths.append(output_path)

            return extracted_paths

        except Exception as e:
            print(f"Warning: Failed to process {hdr_path.name}: {e}")
            return []

    def extract_all_imagettes(self) -> List[Path]:
        """
        Extract all imagettes from ENVI files.

        Returns:
            List of all extracted imagette paths
        """
        # Find all .hdr files
        hdr_files = list(self.data_dir.rglob("*.hdr"))

        if len(hdr_files) == 0:
            raise ValueError(f"No .hdr files found in {self.data_dir}")

        print(f"Found {len(hdr_files)} ENVI files")
        print(f"Extracting imagettes to {self.imagettes_dir}...")

        all_imagettes = []

        for hdr_file in tqdm(hdr_files, desc="Extracting imagettes"):
            imagettes = self.extract_imagettes_from_envi(hdr_file)
            all_imagettes.extend(imagettes)

        print(f"Extracted {len(all_imagettes)} imagettes")

        return all_imagettes

    def load_image(self, image_path: Path) -> torch.Tensor:
        """
        Load an imagette TIFF file.

        Args:
            image_path: Path to TIFF file

        Returns:
            Image tensor of shape (H, W, C)
        """
        image_array = tifffile.imread(str(image_path)).astype(np.float32)
        return torch.tensor(image_array)

    def get_image_paths(self) -> List[Path]:
        """
        Get all imagette paths. Extract if not already done.

        Returns:
            List of imagette paths
        """
        # Check if imagettes already exist
        existing_imagettes = sorted(list(self.imagettes_dir.glob("*.tif")))

        if len(existing_imagettes) > 0:
            print(f"Using {len(existing_imagettes)} existing imagettes")
            return existing_imagettes

        # Extract imagettes
        return self.extract_all_imagettes()


# =============================================================================
# Window-based datasets (UAV, GSOFF, Chikusei)
# =============================================================================

class WindowBasedCovariancePrecompute(CovariancePrecompute):
    """
    Base class for window-based covariance precomputation.

    For datasets with pre-extracted windows stored in .npy format.
    """

    def __init__(
        self,
        data_dir: str,
        windows_data_filename: str,
        windows_labels_filename: str,
        cov_dir: Optional[str] = None,
        scaling_factor: Optional[float] = None,
        remove_mean: bool = True,
        exclude_zero: bool = True,
        num_workers: Optional[int] = None
    ):
        """
        Initialize window-based covariance precomputation.

        Args:
            data_dir: Root directory containing window files
            windows_data_filename: Name of .npy file with window data
            windows_labels_filename: Name of .npy file with labels
            cov_dir: Output directory for covariances
            scaling_factor: Scaling factor for normalization
            remove_mean: Whether to center data
            exclude_zero: Exclude zero pixels
            num_workers: Number of parallel workers
        """
        self.windows_data_path = Path(data_dir) / windows_data_filename
        self.windows_labels_path = Path(data_dir) / windows_labels_filename

        if not self.windows_data_path.exists():
            raise FileNotFoundError(f"Windows data not found: {self.windows_data_path}")
        if not self.windows_labels_path.exists():
            raise FileNotFoundError(f"Windows labels not found: {self.windows_labels_path}")

        # Load data
        print(f"Loading windows from {self.windows_data_path}...")
        self.windows_data = np.load(str(self.windows_data_path), mmap_mode='r')
        self.windows_labels = np.load(str(self.windows_labels_path))

        print(f"Loaded {len(self.windows_data)} windows")
        print(f"Window shape: {self.windows_data.shape[1:]}")

        # Set cov_dir
        if cov_dir is None:
            cov_dir = Path(data_dir) / "cov"

        super().__init__(
            data_dir=data_dir,
            cov_dir=str(cov_dir),
            scaling_factor=scaling_factor,
            remove_mean=remove_mean,
            exclude_zero=exclude_zero,
            num_workers=num_workers
        )

    def load_image(self, image_path: Path) -> torch.Tensor:
        """
        Load a window by index (stored in stem as integer).

        Args:
            image_path: Path with index in stem

        Returns:
            Window tensor of shape (H, W, C)
        """
        idx = int(image_path.stem)
        window = self.windows_data[idx].astype(np.float32)
        return torch.tensor(window)

    def get_image_paths(self) -> List[Path]:
        """
        Generate pseudo-paths for each window.

        Returns:
            List of pseudo-paths with indices
        """
        # Create pseudo-paths with indices
        return [Path(f"{i:06d}") for i in range(len(self.windows_data))]


class UAVCovariancePrecompute(WindowBasedCovariancePrecompute):
    """Covariance precomputation for UAV dataset."""

    def __init__(
        self,
        data_dir: str,
        cov_dir: Optional[str] = None,
        scaling_factor: Optional[float] = None,
        remove_mean: bool = True,
        exclude_zero: bool = True,
        num_workers: Optional[int] = None
    ):
        super().__init__(
            data_dir=data_dir,
            windows_data_filename="uav_windows_data.npy",
            windows_labels_filename="uav_windows_labels.npy",
            cov_dir=cov_dir,
            scaling_factor=scaling_factor,
            remove_mean=remove_mean,
            exclude_zero=exclude_zero,
            num_workers=num_workers
        )


class GSOFFCovariancePrecompute(WindowBasedCovariancePrecompute):
    """Covariance precomputation for GSOFF dataset."""

    def __init__(
        self,
        data_dir: str,
        cov_dir: Optional[str] = None,
        scaling_factor: Optional[float] = None,
        remove_mean: bool = True,
        exclude_zero: bool = True,
        num_workers: Optional[int] = None
    ):
        super().__init__(
            data_dir=data_dir,
            windows_data_filename="gaofeng_windows_data.npy",
            windows_labels_filename="gaofeng_windows_labels.npy",
            cov_dir=cov_dir,
            scaling_factor=scaling_factor,
            remove_mean=remove_mean,
            exclude_zero=exclude_zero,
            num_workers=num_workers
        )


class ChikuseiCovariancePrecompute(WindowBasedCovariancePrecompute):
    """Covariance precomputation for Chikusei dataset."""

    def __init__(
        self,
        data_dir: str,
        cov_dir: Optional[str] = None,
        scaling_factor: Optional[float] = None,
        remove_mean: bool = True,
        exclude_zero: bool = True,
        num_workers: Optional[int] = None
    ):
        super().__init__(
            data_dir=data_dir,
            windows_data_filename="chikusei_windows_data.npy",
            windows_labels_filename="chikusei_windows_labels.npy",
            cov_dir=cov_dir,
            scaling_factor=scaling_factor,
            remove_mean=remove_mean,
            exclude_zero=exclude_zero,
            num_workers=num_workers
        )


class PlacentaCovariancePrecompute(WindowBasedCovariancePrecompute):
    """Covariance precomputation for Placenta dataset."""

    def __init__(
        self,
        data_dir: str,
        cov_dir: Optional[str] = None,
        scaling_factor: Optional[float] = None,
        remove_mean: bool = True,
        exclude_zero: bool = True,
        num_workers: Optional[int] = None
    ):
        super().__init__(
            data_dir=data_dir,
            windows_data_filename="placenta_windows_data.npy",
            windows_labels_filename="placenta_windows_labels.npy",
            cov_dir=cov_dir,
            scaling_factor=scaling_factor,
            remove_mean=remove_mean,
            exclude_zero=exclude_zero,
            num_workers=num_workers
        )


# =============================================================================
# DeepHS-Fruit Dataset
# =============================================================================

class DeepHSFruitCovariancePrecompute(CovariancePrecompute):
    """
    Covariance precomputation for DeepHS-Fruit dataset.

    DeepHS-Fruit images are stored as ENVI files (.hdr/.bin) organized by:
    - Fruit type (Avocado, Kaki, Kiwi, Mango, Papaya)
    - Modality (VIS, NIR)
    - Day (day_01 to day_06)
    - Side (front, back)
    """

    FRUIT_TYPES = ['Avocado', 'Kaki', 'Kiwi', 'Mango', 'Papaya']

    def __init__(
        self,
        data_dir: str,
        modality: str = 'VIS',
        cov_dir: Optional[str] = None,
        scaling_factor: Optional[float] = None,
        remove_mean: bool = True,
        exclude_zero: bool = True,
        num_workers: Optional[int] = None
    ):
        """
        Initialize DeepHS-Fruit covariance precomputation.

        Args:
            data_dir: Root directory of DeepHS-Fruit dataset
            modality: 'VIS' or 'NIR'
            cov_dir: Output directory for covariances (default: data_dir/cov_{modality})
            scaling_factor: Scaling factor for normalization
            remove_mean: Whether to center data
            exclude_zero: Exclude zero pixels
            num_workers: Number of parallel workers
        """
        self.modality = modality

        # Set cov_dir
        if cov_dir is None:
            cov_dir = Path(data_dir) / f"cov_{modality}"

        super().__init__(
            data_dir=data_dir,
            cov_dir=str(cov_dir),
            scaling_factor=scaling_factor,
            remove_mean=remove_mean,
            exclude_zero=exclude_zero,
            num_workers=num_workers
        )

    def load_image(self, image_path: Path) -> torch.Tensor:
        """
        Load an ENVI image (.hdr file).

        Args:
            image_path: Path to .hdr file

        Returns:
            Image tensor of shape (H, W, C)
        """
        try:
            # Construct .bin path
            bin_path = str(image_path).replace('.hdr', '.bin')

            if not Path(bin_path).exists():
                raise FileNotFoundError(f"Binary file not found: {bin_path}")

            # Load ENVI image
            img = envi.open(str(image_path), image=bin_path)
            data = img.load()
            image_array = np.array(data, dtype=np.float32)

            # Convert to tensor
            return torch.tensor(image_array)

        except Exception as e:
            raise RuntimeError(f"Error loading {image_path}: {e}")

    def get_image_paths(self) -> List[Path]:
        """
        Get all .hdr image paths organized by fruit/modality/day.

        Returns:
            List of .hdr file paths
        """
        image_paths = []

        for fruit in self.FRUIT_TYPES:
            fruit_dir = self.data_dir / fruit / self.modality

            if not fruit_dir.exists():
                continue

            # Find all .hdr files recursively
            hdr_files = list(fruit_dir.rglob("*.hdr"))
            image_paths.extend(hdr_files)

        return sorted(image_paths)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Unified CLI for covariance precomputation."""
    parser = argparse.ArgumentParser(
        description="Precompute covariance matrices for hyperspectral datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hyperleaf", "rices90", "uav", "gsoff", "chikusei", "deephsfruit", "placenta"],
        help="Dataset name"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of dataset"
    )

    parser.add_argument(
        "--cov_dir",
        type=str,
        default=None,
        help="Output directory for covariances (default: data_dir/cov)"
    )

    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=None,
        help="Scaling factor for normalization (auto-estimated if not provided)"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of existing covariances"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify covariances after computation"
    )

    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing covariances"
    )

    parser.add_argument(
        "--no_exclude_zero",
        action="store_true",
        help="Include zero pixels in computation"
    )

    # Rices-specific arguments
    parser.add_argument(
        "--imagettes_dir",
        type=str,
        default=None,
        help="[Rices only] Directory for extracted imagettes"
    )

    parser.add_argument(
        "--min_area",
        type=int,
        default=50,
        help="[Rices only] Minimum area for imagette extraction (pixels)"
    )

    # DeepHSFruit-specific arguments
    parser.add_argument(
        "--modality",
        type=str,
        default='VIS',
        choices=['VIS', 'NIR'],
        help="[DeepHSFruit only] Modality: VIS or NIR"
    )

    args = parser.parse_args()

    # Create appropriate precompute instance
    if args.dataset == "hyperleaf":
        precompute = HyperLeafCovariancePrecompute(
            data_dir=args.data_dir,
            cov_dir=args.cov_dir,
            scaling_factor=args.scaling_factor,
            exclude_zero=not args.no_exclude_zero,
            num_workers=args.num_workers
        )

    elif args.dataset == "rices90":
        precompute = RicesCovariancePrecompute(
            data_dir=args.data_dir,
            imagettes_dir=args.imagettes_dir,
            cov_dir=args.cov_dir,
            scaling_factor=args.scaling_factor,
            exclude_zero=not args.no_exclude_zero,
            num_workers=args.num_workers,
            min_area=args.min_area
        )

    elif args.dataset == "uav":
        precompute = UAVCovariancePrecompute(
            data_dir=args.data_dir,
            cov_dir=args.cov_dir,
            scaling_factor=args.scaling_factor,
            exclude_zero=not args.no_exclude_zero,
            num_workers=args.num_workers
        )

    elif args.dataset == "gsoff":
        precompute = GSOFFCovariancePrecompute(
            data_dir=args.data_dir,
            cov_dir=args.cov_dir,
            scaling_factor=args.scaling_factor,
            exclude_zero=not args.no_exclude_zero,
            num_workers=args.num_workers
        )

    elif args.dataset == "chikusei":
        precompute = ChikuseiCovariancePrecompute(
            data_dir=args.data_dir,
            cov_dir=args.cov_dir,
            scaling_factor=args.scaling_factor,
            exclude_zero=not args.no_exclude_zero,
            num_workers=args.num_workers
        )

    elif args.dataset == "deephsfruit":
        precompute = DeepHSFruitCovariancePrecompute(
            data_dir=args.data_dir,
            modality=args.modality,
            cov_dir=args.cov_dir,
            scaling_factor=args.scaling_factor,
            exclude_zero=not args.no_exclude_zero,
            num_workers=args.num_workers
        )

    elif args.dataset == "placenta":
        precompute = PlacentaCovariancePrecompute(
            data_dir=args.data_dir,
            cov_dir=args.cov_dir,
            scaling_factor=args.scaling_factor,
            exclude_zero=not args.no_exclude_zero,
            num_workers=args.num_workers
        )

    # Run precomputation or verification
    if args.verify_only:
        precompute.verify()
    else:
        precompute.precompute(force=args.force)
        if args.verify:
            precompute.verify()


if __name__ == "__main__":
    main()
