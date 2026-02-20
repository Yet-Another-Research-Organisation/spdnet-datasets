"""
Covariance estimation methods for SPD matrices.
"""

import numpy as np
import torch
from sklearn.covariance import ledoit_wolf
from typing import Literal


class EstimateCovariance:
    """
    Estimate covariance matrix from data.

    Supports multiple methods:
    - 'scm': Sample Covariance Matrix (empirical)
    - 'ledoit_wolf': Ledoit-Wolf shrinkage estimator
    """

    def __init__(
        self,
        method: Literal['scm', 'ledoit_wolf'] = 'scm',
        remove_mean: bool = True,
        exclude_zero: bool = True
    ):
        """
        Initialize covariance estimator.

        Args:
            method: Estimation method ('scm' or 'ledoit_wolf')
            remove_mean: Whether to center data before computing covariance
            exclude_zero: Exclude samples where all features are zero
        """
        if method not in ['scm', 'ledoit_wolf']:
            raise ValueError(f"Unknown method '{method}'. Use 'scm' or 'ledoit_wolf'")

        self.method = method
        self.remove_mean = remove_mean
        self.exclude_zero = exclude_zero

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Compute covariance matrix.

        Args:
            X: Data array of shape (n_samples, n_features)
               For images: reshape to (H*W, C) before calling

        Returns:
            Covariance matrix of shape (n_features, n_features)
        """
        # Ensure 2D array
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        # Exclude zero samples if requested
        if self.exclude_zero:
            mask = ~np.all(X == 0, axis=1)
            data = X[mask]
        else:
            data = X

        if len(data) == 0:
            raise ValueError("No samples remaining after filtering zeros")

        # Compute covariance
        if self.method == 'ledoit_wolf':
            cov, _ = ledoit_wolf(data)
            return cov

        else:  # scm
            if self.remove_mean:
                # Center data
                data_centered = data - data.mean(axis=0, keepdims=True)
            else:
                data_centered = data

            # Compute empirical covariance: (1/n) * X^T @ X
            n_samples = data_centered.shape[0]
            cov = (data_centered.T @ data_centered) / n_samples

            return cov

    def from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Compute covariance from image.

        Args:
            image: Image array of shape (H, W, C)

        Returns:
            Covariance matrix of shape (C, C)
        """
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image (H, W, C), got shape {image.shape}")

        # Reshape to (H*W, C)
        data = image.reshape(-1, image.shape[-1])

        return self(data)

    def from_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Compute covariance for a batch of images.

        Args:
            images: Batch of images of shape (B, H, W, C)

        Returns:
            Batch of covariance matrices of shape (B, C, C)
        """
        if images.ndim != 4:
            raise ValueError(f"Expected 4D batch (B, H, W, C), got shape {images.shape}")

        batch_size = images.shape[0]
        cov_matrices = []

        for i in range(batch_size):
            cov = self.from_image(images[i])
            cov_matrices.append(cov)

        return np.stack(cov_matrices, axis=0)


class EstimateCovarianceTorch:
    """
    PyTorch version of covariance estimator (GPU-compatible).
    Only supports SCM method (empirical covariance).
    """

    def __init__(
        self,
        remove_mean: bool = True,
        exclude_zero: bool = True
    ):
        """
        Initialize PyTorch covariance estimator.

        Args:
            remove_mean: Whether to center data before computing covariance
            exclude_zero: Exclude samples where all features are zero
        """
        self.remove_mean = remove_mean
        self.exclude_zero = exclude_zero

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance matrix.

        Args:
            X: Data tensor of shape (n_samples, n_features)

        Returns:
            Covariance matrix of shape (n_features, n_features)
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {X.shape}")

        # Exclude zero samples if requested
        if self.exclude_zero:
            mask = ~torch.all(X == 0, dim=1)
            data = X[mask]
        else:
            data = X

        if len(data) == 0:
            raise ValueError("No samples remaining after filtering zeros")

        # Center data if requested
        if self.remove_mean:
            data_centered = data - data.mean(dim=0, keepdim=True)
        else:
            data_centered = data

        # Compute empirical covariance
        n_samples = data_centered.shape[0]
        cov = (data_centered.T @ data_centered) / n_samples

        return cov

    def from_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance from image.

        Args:
            image: Image tensor of shape (H, W, C)

        Returns:
            Covariance matrix of shape (C, C)
        """
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image (H, W, C), got shape {image.shape}")

        # Reshape to (H*W, C)
        data = image.reshape(-1, image.shape[-1])

        return self(data)

    def from_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance for a batch of images.

        Args:
            images: Batch of images of shape (B, H, W, C)

        Returns:
            Batch of covariance matrices of shape (B, C, C)
        """
        if images.ndim != 4:
            raise ValueError(f"Expected 4D batch (B, H, W, C), got shape {images.shape}")

        batch_size = images.shape[0]
        cov_matrices = []

        for i in range(batch_size):
            cov = self.from_image(images[i])
            cov_matrices.append(cov)

        return torch.stack(cov_matrices, dim=0)
