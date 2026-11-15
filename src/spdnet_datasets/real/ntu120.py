"""
NTU RGB+D 120 Skeleton Dataset Loader (deprecated).

Use :class:`spdnet_datasets.skeleton.ntu120.NTURGBD120Dataset`, which computes
the covariances from the skeletons in a cache, with the official X-Sub /
X-Setup splits; the name ``"ntu120"`` of the DatasetManager points to it.

This legacy class only reads covariance matrices precomputed under
``<data_dir>/cov/`` (former ``utils/ntu120_covariance_precompute.py``).
"""

import warnings
from pathlib import Path

import torch

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.skeleton.ntu120 import NTURGBD120Dataset, fetch_ntu120

__all__ = ["NTU120Dataset", "NTURGBD120Dataset", "fetch_ntu120"]


class NTU120Dataset(BaseDataset):
    """
    NTU RGB+D 120 Skeleton Dataset (deprecated, precomputed ``cov/`` only).

    Loads pre-computed covariance matrices from skeleton motion sequences.
    Each skeleton has 25 joints × 3 coordinates = 75 features.
    Covariance is computed over time steps.

    Expected structure::

        data_dir/
            cov/
                A001/
                    sample_name.pt
                A002/
                    ...
                ...

    Args:
        data_dir: Root directory of the dataset
        max_classes: Maximum number of classes to use (None = all)
        max_samples_per_class: Maximum samples per class (None = all)
        scaling_factor: Scaling factor to apply to covariance matrices
        verbose: Print dataset information
    """

    def __init__(self, scaling_factor: float = 1.0, **kwargs):
        """Initialize NTU120 dataset."""
        warnings.warn("spdnet_datasets.real.NTU120Dataset is deprecated; use "
                      "spdnet_datasets.skeleton.NTURGBD120Dataset", DeprecationWarning,
                      stacklevel=2)
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor
        self._load_data()

    @property
    def num_classes(self):
        """Return number of classes."""
        return len(self.classes)

    def _load_data(self):
        """Load NTU120 covariance matrices."""
        data_dir = Path(self.data_dir)
        cov_dir = data_dir / "cov"

        if not cov_dir.exists():
            raise FileNotFoundError(
                f"Covariance directory not found: {cov_dir}. "
                "Run ntu120_covariance_precompute.py first."
            )

        if self.verbose:
            print(f"Loading NTU120 dataset from {cov_dir}")

        class_dirs = sorted([d for d in cov_dir.iterdir()
                             if d.is_dir() and d.name != "__pycache__"])

        if not class_dirs:
            raise ValueError(f"No class directories found in {cov_dir}")

        # Class selection (explicit `classes`, else `max_classes`) and the
        # per-class sample limit go through the BaseDataset helpers.
        self.classes = self._limit_classes(sorted(d.name for d in class_dirs))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        samples_by_class = {}
        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                continue
            samples_by_class[class_name] = [
                {'file_path': str(pt_file), 'class_name': class_name}
                for pt_file in sorted(class_dir.glob("*.pt"))
            ]
        self.samples = self._limit_samples_per_class(samples_by_class)

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
        if cov_matrix.dtype != torch.float64:
            cov_matrix = cov_matrix.double()
        cov_matrix = cov_matrix * self.scaling_factor
        class_idx = self.class_to_idx[sample['class_name']]
        return cov_matrix, class_idx

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def get_num_classes(self) -> int:
        """Return number of classes."""
        return len(self.classes)
