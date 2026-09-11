"""
MVDoppler Radar Activity Dataset Loader.

Dataset: MVDoppler (multi-view Doppler radar activity recognition)
Format: Pre-computed covariance matrices from radar spectrograms
Size: 96×96 covariance matrices (from center-cropped frequency axis)
Classes: normal, phone_call, pockets, texting
"""

from pathlib import Path

import torch

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.manager import DatasetManager


@DatasetManager.register_dataset("mvdoppler")
class MVDopplerDataset(BaseDataset):
    """
    MVDoppler Radar Activity Dataset.

    Loads pre-computed covariance matrices from radar I/Q data.
    Original data: (128, 256, 2) → magnitude → crop center 75% → (96, 256)
    → covariance over time axis: (96, 96).

    Expected structure:
        data_dir/
            cov/
                normal/
                    sample.pt
                phone_call/
                    ...

    Args:
        data_dir: Root directory of the dataset
        max_classes: Maximum number of classes to use (None = all)
        max_samples_per_class: Maximum samples per class (None = all)
        scaling_factor: Scaling factor to apply to covariance matrices
        verbose: Print dataset information
    """

    def __init__(self, scaling_factor: float = 1.0, **kwargs):
        """Initialize MVDoppler dataset."""
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor
        self._load_data()

    @property
    def num_classes(self):
        """Return number of classes."""
        return len(self.classes)

    def _load_data(self):
        """Load MVDoppler covariance matrices."""
        data_dir = Path(self.data_dir)
        cov_dir = data_dir / "cov"

        if not cov_dir.exists():
            raise FileNotFoundError(
                f"Covariance directory not found: {cov_dir}. "
                "Run mvdoppler_covariance_precompute.py first."
            )

        if self.verbose:
            print(f"Loading MVDoppler dataset from {cov_dir}")

        class_dirs = sorted([d for d in cov_dir.iterdir()
                             if d.is_dir() and d.name != "__pycache__"])

        if not class_dirs:
            raise ValueError(f"No class directories found in {cov_dir}")

        self.classes = sorted([d.name for d in class_dirs])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        self.samples = []
        for class_dir in class_dirs:
            class_name = class_dir.name
            pt_files = sorted(class_dir.glob("*.pt"))
            for pt_file in pt_files:
                self.samples.append({
                    'file_path': str(pt_file),
                    'class_name': class_name,
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
