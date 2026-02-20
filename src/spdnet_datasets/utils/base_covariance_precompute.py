#!/usr/bin/env python3
"""
Base class for covariance matrix precomputation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import torch
from tqdm import tqdm

from ..estimator.covariance import EstimateCovarianceTorch


def process_single_image(image_path: Path, output_path: Path, scaling_factor: float, load_func, cov_estimator) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single image: load, compute covariance, save.

    Args:
        image_path: Path to image file
        output_path: Path to save covariance
        scaling_factor: Scaling factor
        load_func: Function to load image
        cov_estimator: Covariance estimator object

    Returns:
        Tuple of (image_id, success, error_message)
    """
    image_id = image_path.stem

    try:
        # Load and scale image (multiply instead of divide)
        image = load_func(image_path)
        image_scaled = image * scaling_factor

        # Compute covariance
        cov_matrix = cov_estimator.from_image(image_scaled)

        # Save
        torch.save(cov_matrix, output_path)

        return (image_id, True, None)

    except Exception as e:
        return (image_id, False, str(e))


class CovariancePrecompute(ABC):
    """
    Base class for precomputing covariance matrices from hyperspectral images.

    This class handles the general workflow:
    1. Load images with scaling factor
    2. Compute covariance matrices
    3. Save them in a 'cov' directory

    If scaling factor is not provided, it is estimated to normalize
    eigenvalues around 1.0 (geometric mean of min/max across samples).
    """

    def __init__(
        self,
        data_dir: str,
        cov_dir: Optional[str] = None,
        scaling_factor: Optional[float] = None,
        remove_mean: bool = True,
        exclude_zero: bool = True,
        num_workers: Optional[int] = None
    ):
        """
        Initialize covariance precomputation.

        Args:
            data_dir: Root directory containing images
            cov_dir: Output directory for covariances (default: data_dir/cov)
            scaling_factor: Factor to divide images before computing covariance
            remove_mean: Whether to center data before computing covariance
            exclude_zero: Exclude zero pixels from computation
            num_workers: Number of parallel workers (default: auto-detect)
        """
        self.data_dir = Path(data_dir).expanduser()
        self.cov_dir = Path(cov_dir).expanduser() if cov_dir else self.data_dir / "cov"
        self.scaling_factor = scaling_factor
        self.remove_mean = remove_mean
        self.exclude_zero = exclude_zero
        self.num_workers = num_workers or min(cpu_count(), 8)

        # Create output directory
        self.cov_dir.mkdir(parents=True, exist_ok=True)

        # Covariance estimator
        self.cov_estimator = EstimateCovarianceTorch(
            remove_mean=remove_mean,
            exclude_zero=exclude_zero
        )

    @abstractmethod
    def load_image(self, image_path: Path) -> torch.Tensor:
        """
        Load and preprocess an image.

        Args:
            image_path: Path to image file

        Returns:
            Image tensor of shape (H, W, C)
        """
        pass

    @abstractmethod
    def get_image_paths(self) -> List[Path]:
        """
        Get list of all image paths to process.

        Returns:
            List of image paths
        """
        pass

    def estimate_scaling_factor(self) -> float:
        """
        Estimate optimal scaling factor for eigenvalue stability.

        The factor is computed so that the geometric mean of eigenvalues
        across all samples is around 1.0. Uses linalg.eigvalsh on covariances.

        Returns:
            Estimated scaling factor (to multiply with images)
        """
        image_paths = self.get_image_paths()
        if len(image_paths) == 0:
            raise ValueError("No images found for scaling factor estimation")

        print(f"Estimating scaling factor from {len(image_paths)} samples...")

        geometric_means = []

        # Temporary covariance estimator without scaling
        temp_estimator = EstimateCovarianceTorch(
            remove_mean=self.remove_mean,
            exclude_zero=self.exclude_zero
        )

        for img_path in tqdm(image_paths, desc="Computing eigenvalues"):
            try:
                image = self.load_image(img_path)

                # Compute covariance without scaling
                cov = temp_estimator.from_image(image)

                # Compute eigenvalues
                eigenvalues = torch.linalg.eigvalsh(cov)

                # Geometric mean of eigenvalues
                eig_min = eigenvalues.min().item()
                eig_max = eigenvalues.max().item()

                if eig_min > 0 and eig_max > 0:
                    geo_mean = np.sqrt(eig_min * eig_max)
                    geometric_means.append(geo_mean)

            except Exception as e:
                print(f"Warning: Failed to process {img_path.name}: {e}")
                continue

        if len(geometric_means) == 0:
            raise ValueError("No valid samples for scaling factor estimation")

        # Average geometric mean across samples
        avg_geo_mean = np.mean(geometric_means)

        # Target geometric mean of 1.0
        target_geo_mean = 1.0
        scaling_factor = np.sqrt(target_geo_mean / avg_geo_mean)

        print(f"Estimated scaling factor: {scaling_factor:.6e}")
        print(f"Based on {len(geometric_means)} samples")

        return scaling_factor

    def _process_single_image(self, args: Tuple[Path, Path, float]) -> Tuple[str, bool, Optional[str]]:
        """
        Process a single image: load, compute covariance, save.

        Args:
            args: Tuple of (image_path, output_path, scaling_factor)

        Returns:
            Tuple of (image_id, success, error_message)
        """
        image_path, output_path, scaling_factor = args
        image_id = image_path.stem

        try:
            # Load and scale image (multiply instead of divide)
            image = self.load_image(image_path)
            image_scaled = image * scaling_factor

            # Compute covariance
            cov_matrix = self.cov_estimator.from_image(image_scaled)

            # Save
            torch.save(cov_matrix, output_path)

            return (image_id, True, None)

        except Exception as e:
            return (image_id, False, str(e))

    def precompute(self, force: bool = False) -> Dict[str, int]:
        """
        Precompute all covariance matrices.

        Args:
            force: If True, recompute even if covariance already exists

        Returns:
            Dictionary with statistics
        """
        # Get image paths
        image_paths = self.get_image_paths()

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {self.data_dir}")

        print(f"Found {len(image_paths)} images")

        # Estimate scaling factor if not provided
        if self.scaling_factor is None:
            self.scaling_factor = self.estimate_scaling_factor()
        else:
            print(f"Using provided scaling factor: {self.scaling_factor:.6e}")

        # Prepare tasks
        tasks = []
        skipped = 0

        for img_path in image_paths:
            output_path = self.cov_dir / f"{img_path.stem}.pt"

            if output_path.exists() and not force:
                skipped += 1
                continue

            tasks.append((img_path, output_path, self.scaling_factor))

        if skipped > 0:
            print(f"Skipping {skipped} existing covariances (use --force to recompute)")

        if len(tasks) == 0:
            print("All covariances already computed")
            return {
                "total": len(image_paths),
                "skipped": skipped,
                "processed": 0,
                "failed": 0
            }

        print(f"Processing {len(tasks)} images with {self.num_workers} workers...")

        # Process in parallel
        successful = 0
        failed = []

        if self.num_workers == 1:
            for task in tqdm(tasks, desc="Computing covariances"):
                image_id, success, error = self._process_single_image(task)
                if success:
                    successful += 1
                else:
                    failed.append((image_id, error))
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(self._process_single_image, task): task for task in tasks}

                with tqdm(total=len(tasks), desc="Computing covariances") as pbar:
                    for future in as_completed(futures):
                        image_id, success, error = future.result()
                        if success:
                            successful += 1
                        else:
                            failed.append((image_id, error))
                        pbar.update(1)

        # Statistics
        stats = {
            "total": len(image_paths),
            "skipped": skipped,
            "processed": successful,
            "failed": len(failed)
        }

        # Summary
        print("\n" + "="*60)
        print("COVARIANCE PRECOMPUTATION COMPLETE")
        print("="*60)
        print(f"Total images: {stats['total']}")
        print(f"Already computed: {stats['skipped']}")
        print(f"Successfully processed: {stats['processed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Output directory: {self.cov_dir}")

        if failed:
            print("\nFirst 5 failures:")
            for image_id, error in failed[:5]:
                print(f"  - {image_id}: {error}")

        return stats

    def verify(self, num_samples: int = 5) -> None:
        """
        Verify precomputed covariances.

        Args:
            num_samples: Number of samples to verify
        """
        cov_files = sorted(list(self.cov_dir.glob("*.pt")))

        if len(cov_files) == 0:
            print(f"No covariance files found in {self.cov_dir}")
            return

        print(f"\nVerifying {min(num_samples, len(cov_files))} covariances...")
        print("-" * 60)

        for i, cov_file in enumerate(cov_files[:num_samples]):
            try:
                cov = torch.load(cov_file)

                # Check properties
                is_square = cov.shape[0] == cov.shape[1]
                is_symmetric = torch.allclose(cov, cov.T, rtol=1e-5, atol=1e-7)

                eigenvalues = torch.linalg.eigvalsh(cov)
                min_eig = eigenvalues.min().item()
                is_psd = min_eig >= -1e-6

                status = "OK" if (is_square and is_symmetric and is_psd) else "FAIL"

                print(f"{i+1}. {cov_file.name} {status}")
                print(f"   Shape: {cov.shape}, Trace: {torch.trace(cov).item():.4f}")
                print(f"   Eigenvalues: [{eigenvalues.min().item():.2e}, {eigenvalues.max().item():.2e}]")

            except Exception as e:
                print(f"{i+1}. {cov_file.name} FAIL ERROR: {e}")

        print("-" * 60)
        print(f"Total covariance files: {len(cov_files)}")
