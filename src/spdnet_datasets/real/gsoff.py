"""
GSOFF (Gaofeng State Owned Forest Farm) Dataset - Hyperspectral imagery for tree species classification.
Dataset from https://github.com/Bin-Zh/Three-dimensional-convolutional-neural-network-model-for-tree-species-classification-using-airborne-
"""

import numpy as np
import torch
from typing import Tuple
from pathlib import Path

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.manager import DatasetManager
from spdnet_datasets.estimator import EstimateCovariance


@DatasetManager.register_dataset('gsoff')
class GSOffDataset(BaseDataset):
    """
    Gaofeng State Owned Forest Farm hyperspectral dataset for tree species classification.

    Dataset contains hyperspectral windows extracted from airborne imagery
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = 'raw',  # 'cov' for covariance, 'raw' for windows
        cov_method: str = 'scm',  # 'scm' or 'ledoit_wolf'
        exclude_zero: bool = True,
        preload: bool = False,
        **kwargs
    ):
        """
        Initialize GSOFF dataset.

        Args:
            data_dir: Path to GSOFF folder containing extracted windows
            mode: 'cov' for pre-computed covariances, 'raw' for computing from windows
            cov_method: 'scm' (empirical) or 'ledoit_wolf'
            exclude_zero: Exclude zero pixels when computing covariance
            preload: If True, load all data to RAM at init (faster training, more RAM)
            **kwargs: Additional parameters from config
        """
        self.mode = mode
        self.cov_method = cov_method
        self.exclude_zero = exclude_zero
        self.preload = preload
        self._cache = {}  # Cache for preloaded data

        # Setup covariance estimator if needed
        if self.mode == 'raw':
            self.cov_estimator = EstimateCovariance(
                method=cov_method,
                remove_mean=True,
                exclude_zero=exclude_zero
            )

        # Initialize base
        super().__init__(data_dir=data_dir, **kwargs)

        # Load data
        self._load_data()

    def _load_data(self):
        """Load GSOFF dataset from pre-extracted windows."""
        if self.verbose:
            print(f"Loading GSOFF dataset")

        # Load data files
        windows_file = self.data_dir / 'gaofeng_windows_data.npy'
        labels_file = self.data_dir / 'gaofeng_windows_labels.npy'

        if not windows_file.exists() or not labels_file.exists():
            raise FileNotFoundError(
                f"Data files not found in {self.data_dir}\n"
                f"Expected files: gaofeng_windows_data.npy, gaofeng_windows_labels.npy"
            )

        # Load arrays
        all_windows = np.load(str(windows_file))
        all_labels = np.load(str(labels_file)).astype(int)

        if self.verbose:
            print(f"  Loaded {len(all_windows)} samples")
            print(f"  Window shape: {all_windows.shape[1:]}")

        # Get unique labels and create mapping
        unique_labels = sorted(np.unique(all_labels).tolist())
        self.label_mapping = {old_label: new_idx for new_idx, old_label in enumerate(unique_labels)}
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

        remapped_labels = np.array([self.label_mapping[label] for label in all_labels])

        # Setup cov_dir for covariance mode
        self.cov_dir = self.data_dir / "cov"
        if self.mode == 'cov' and not self.cov_dir.exists():
            raise FileNotFoundError(f"Covariance directory not found: {self.cov_dir}")

        # Store windows data for raw mode
        self.all_windows = all_windows

        # Setup classes
        self.classes = [f"TreeSpecies_{label}" for label in unique_labels]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Create samples grouped by class (with indices)
        samples_by_class = {name: [] for name in self.classes}

        for idx in range(len(all_windows)):
            class_idx = remapped_labels[idx]
            class_name = self.classes[class_idx]

            samples_by_class[class_name].append({
                'index': idx,  # Store index instead of window data
                'class_idx': class_idx,
                'class_name': class_name
            })

        # Limit samples per class if requested
        self.samples = self._limit_samples_per_class(samples_by_class)

        if self.verbose:
            print(f"Final dataset: {len(self.samples)} samples, {len(self.classes)} classes")
            self._print_class_distribution()

        # Preload data if requested
        if self.preload and self.mode == 'cov':
            self._preload_all()

    def _print_class_distribution(self):
        """Print class distribution."""
        if not self.verbose:
            return

        class_counts = {}
        for sample in self.samples:
            class_name = sample['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print("\nClass distribution:")
        for class_name in self.classes:
            count = class_counts.get(class_name, 0)
            percentage = count / len(self.samples) * 100 if self.samples else 0
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

    @property
    def num_classes(self):
        """Return number of classes."""
        return len(self.classes)

    def _preload_all(self):
        """Preload all covariance matrices into memory."""
        if not self.preload or self.mode != 'cov':
            return

        if self.verbose:
            print(f"Preloading {len(self.samples)} covariance matrices to RAM...")

        for sample in self.samples:
            idx = sample['index']
            cov_path = self.cov_dir / f"{idx:06d}.pt"
            if cov_path.exists():
                cov_matrix = torch.load(cov_path, weights_only=False)
                if isinstance(cov_matrix, np.ndarray):
                    cov_matrix = torch.from_numpy(cov_matrix).double()
                elif cov_matrix.dtype != torch.float64:
                    cov_matrix = cov_matrix.double()
                self._cache[idx] = cov_matrix

        if self.verbose:
            print(f"Preloaded {len(self._cache)} matrices to RAM")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Returns:
            Tuple of (data_tensor, class_index)
            data_tensor is covariance matrix (mode='cov') or window (mode='raw')
        """
        sample = self.samples[idx]
        sample_idx = sample['index']

        # Check cache first (for preloaded covariances)
        if sample_idx in self._cache:
            data_tensor = self._cache[sample_idx]

        elif self.mode == 'cov':
            # Load pre-computed covariance
            cov_path = self.cov_dir / f"{sample_idx:06d}.pt"

            if not cov_path.exists():
                raise FileNotFoundError(f"Covariance file not found: {cov_path}")

            data_tensor = torch.load(cov_path, weights_only=False)

            if isinstance(data_tensor, np.ndarray):
                data_tensor = torch.from_numpy(data_tensor).double()
            elif data_tensor.dtype != torch.float64:
                data_tensor = data_tensor.double()

        else:  # mode == 'raw'
            # Get window data
            window = self.all_windows[sample_idx]

            # Convert to tensor (H, W, C)
            window_tensor = torch.from_numpy(window).float()

            # Compute covariance if estimator is configured
            if hasattr(self, 'cov_estimator'):
                data_tensor = self.cov_estimator.from_image(window)
                data_tensor = torch.from_numpy(data_tensor).float()
            else:
                data_tensor = window_tensor

        # Apply transform if specified
        if self.transform is not None:
            data_tensor = self.transform(data_tensor)

        class_idx = sample['class_idx']

        return data_tensor, class_idx

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)
