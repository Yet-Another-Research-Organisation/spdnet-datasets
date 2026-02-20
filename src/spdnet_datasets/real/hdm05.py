"""
HDM05 Human Motion Dataset Loader.

Dataset: HDM05 (Hochschule der Medien Motion Capture Database)
Format: Covariance matrices from motion capture sequences
Pattern: {id}_{frames}_{class}.npy
Size: Variable dimension covariance matrices
"""

import numpy as np
import torch
from pathlib import Path
import re

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.manager import DatasetManager


@DatasetManager.register_dataset("hdm05")
class HDM05Dataset(BaseDataset):
    """
    HDM05 Human Motion Dataset.

    Expected structure:
        data_dir/
            {id}_{frames}_{class}.npy  # Covariance matrices

    Filename format: {sequence_id}_{num_frames}_{motion_class}.npy
    Example: 001_120_walk.npy

    Args:
        data_dir: Root directory of the dataset
        max_classes: Maximum number of classes to use (None = all)
        max_samples_per_class: Maximum samples per class (None = all)
        scaling_factor: Scaling factor to apply to covariance matrices (default: 1.0)
        verbose: Print dataset information
    """

    def __init__(self, scaling_factor: float = 1.0, **kwargs):
        """Initialize HDM05 dataset."""
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor
        self._load_data()

    @property
    def num_classes(self):
        """Return number of classes."""
        return len(self.classes)

    def _load_data(self):
        """Load HDM05 motion dataset."""
        data_dir = Path(self.data_dir)

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        if self.verbose:
            print(f"Loading HDM05 dataset from {data_dir}")

        # Find all .npy files
        npy_files = list(data_dir.glob("*.npy"))

        if len(npy_files) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")

        if self.verbose:
            print(f"Found {len(npy_files)} .npy files")

        # Parse filenames to extract classes
        # Pattern: {id}_{frames}_{class}.npy
        pattern = re.compile(r'^(\d+)_(\d+)_(.+)\.npy$')

        class_names = set()
        parsed_files = []

        for file_path in npy_files:
            match = pattern.match(file_path.name)
            if match:
                seq_id, num_frames, class_name = match.groups()
                class_names.add(class_name)
                parsed_files.append({
                    'file_path': str(file_path),
                    'seq_id': seq_id,
                    'num_frames': int(num_frames),
                    'class_name': class_name
                })
            elif self.verbose:
                print(f"Warning: Skipping file with unexpected format: {file_path.name}")

        if len(parsed_files) == 0:
            raise ValueError(f"No valid files found matching pattern {pattern.pattern}")

        # Set up classes
        self.classes = sorted(list(class_names))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        if self.verbose:
            print(f"Found {len(self.classes)} motion classes: {self.classes}")

        self.samples = parsed_files

        if self.verbose:
            print(f"Loaded {len(self.samples)} samples")
            self._print_class_distribution()

    def _print_class_distribution(self):
        """Print distribution of samples across classes."""
        if not self.verbose:
            return

        class_counts = {}
        for sample in self.samples:
            cls = sample['class_name']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        print("\nClass distribution:")
        for cls in self.classes:
            count = class_counts.get(cls, 0)
            print(f"  {cls}: {count} samples")

    def __getitem__(self, idx: int):
        """
        Get sample at index.

        Args:
            idx: Sample index

        Returns:
            tuple: (covariance_matrix, class_idx)
        """
        sample = self.samples[idx]

        # Load covariance matrix
        cov_matrix = np.load(sample['file_path'])
        cov_matrix = cov_matrix * self.scaling_factor
        cov_matrix = torch.from_numpy(cov_matrix).float()

        # Get class index
        class_idx = self.class_to_idx[sample['class_name']]

        return cov_matrix, class_idx

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def get_num_classes(self) -> int:
        """Return number of classes."""
        return len(self.classes)
