"""
Dop-NET Doppler Gesture Dataset Loader.

Dataset: Dop-NET (Doppler radar gesture recognition)
Format: Pre-computed covariance matrices from complex spectrograms
Size: ~50×50 covariance matrices
Classes: Click, Swipe, Pinch, Wave
"""

from pathlib import Path

import torch

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.manager import DatasetManager


@DatasetManager.register_dataset("dopnet")
class DopNetDataset(BaseDataset):
    """
    Dop-NET Doppler Gesture Dataset.

    Loads pre-computed covariance matrices from Doppler radar spectrograms.
    Supports train/test split as defined by the original dataset.

    Expected structure:
        data_dir/
            cov/
                train/
                    Click/
                        sample.pt
                    Swipe/
                        ...
                test/
                    Click/
                        ...

    Args:
        data_dir: Root directory of the dataset
        split: Which split to load ('train', 'test', or 'all')
        max_classes: Maximum number of classes to use (None = all)
        max_samples_per_class: Maximum samples per class (None = all)
        scaling_factor: Scaling factor to apply to covariance matrices
        verbose: Print dataset information
    """

    def __init__(
        self,
        split: str = 'all',
        scaling_factor: float = 1.0,
        **kwargs,
    ):
        """Initialize DopNet dataset."""
        super().__init__(**kwargs)
        self.split = split
        self.scaling_factor = scaling_factor
        self._load_data()

    @property
    def num_classes(self):
        """Return number of classes."""
        return len(self.classes)

    def _load_data(self):
        """Load DopNet covariance matrices."""
        data_dir = Path(self.data_dir)
        cov_dir = data_dir / "cov"

        if not cov_dir.exists():
            raise FileNotFoundError(
                f"Covariance directory not found: {cov_dir}. "
                "Run dopnet_covariance_precompute.py first."
            )

        if self.verbose:
            print(f"Loading DopNet dataset from {cov_dir} (split={self.split})")

        # Determine which split dirs to load
        if self.split == 'all':
            split_dirs = [d for d in cov_dir.iterdir()
                          if d.is_dir() and d.name in ('train', 'test')]
        elif self.split in ('train', 'test'):
            split_dir = cov_dir / self.split
            if not split_dir.exists():
                raise FileNotFoundError(f"Split directory not found: {split_dir}")
            split_dirs = [split_dir]
        else:
            raise ValueError(f"Invalid split: {self.split}. "
                             "Use 'train', 'test', or 'all'.")

        # Collect all class names first
        all_class_names = set()
        for split_dir in split_dirs:
            for d in split_dir.iterdir():
                if d.is_dir():
                    all_class_names.add(d.name)

        self.classes = sorted(all_class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Collect samples
        self.samples = []
        for split_dir in split_dirs:
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                if class_name not in self.class_to_idx:
                    continue
                for pt_file in sorted(class_dir.glob("*.pt")):
                    self.samples.append({
                        'file_path': str(pt_file),
                        'class_name': class_name,
                        'split': split_dir.name,
                    })

        if not self.samples:
            raise ValueError(f"No .pt files found in {cov_dir}")

        if self.verbose:
            print(f"Found {len(self.classes)} classes, "
                  f"{len(self.samples)} samples")
            self._print_class_distribution()

    def _print_class_distribution(self):
        """Print distribution of samples across classes."""
        class_counts = {}
        for sample in self.samples:
            cls = sample['class_name']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        print("\nClass distribution:")
        for cls in self.classes:
            count = class_counts.get(cls, 0)
            print(f"  {cls}: {count} samples")

    def __getitem__(self, idx: int):
        """Get sample at index. Returns (covariance_matrix, class_idx)."""
        sample = self.samples[idx]
        cov_matrix = torch.load(sample['file_path'], weights_only=True)
        cov_matrix = cov_matrix * self.scaling_factor
        class_idx = self.class_to_idx[sample['class_name']]
        return cov_matrix, class_idx

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def get_num_classes(self) -> int:
        """Return number of classes."""
        return len(self.classes)
