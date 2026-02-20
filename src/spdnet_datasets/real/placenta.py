"""
Hyperspectral Placenta Dataset - Tissue classification from hyperspectral imagery.

Two loading modes:
1. 'mixed': Shuffle samples from both dye types (ICG and red_blue) together
2. 'domain_adaptation': Use one dye type for train, other for test
"""

import numpy as np
import torch
from typing import Tuple, Dict, List
from pathlib import Path
import json

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.manager import DatasetManager
from spdnet_datasets.estimator import EstimateCovariance


@DatasetManager.register_dataset('placenta')
class PlacentaDataset(BaseDataset):
    """
    Hyperspectral Placenta dataset for tissue classification.

    Classes: Artery, Stroma, Umbilical, Suture, Vein
    Two dye types: ICG and red_blue
    """

    # Class names
    CLASS_NAMES = ['Artery', 'Stroma', 'Umbilical', 'Suture', 'Vein']
    DYE_TYPES = ['ICG', 'red_blue']

    def __init__(
        self,
        data_dir: str,
        mode: str = 'cov',  # 'cov' for covariances, 'raw' for windows
        split_mode: str = 'mixed',  # 'mixed' or 'domain_adaptation'
        train_dye: str = None,  # For domain_adaptation: 'ICG', 'red_blue', or None (auto-select)
        cov_method: str = 'scm',  # 'scm' or 'ledoit_wolf'
        exclude_zero: bool = True,
        preload: bool = False,
        **kwargs
    ):
        """
        Initialize Placenta dataset.

        Args:
            data_dir: Path to imagettes folder
            mode: 'cov' for pre-computed covariances, 'raw' for computing from windows
            split_mode: 'mixed' or 'domain_adaptation'
            train_dye: For domain_adaptation mode, specify train dye or None for auto
            cov_method: 'scm' (empirical) or 'ledoit_wolf'
            exclude_zero: Exclude zero pixels when computing covariance
            preload: If True, load all data to RAM at init
            **kwargs: Additional parameters from config
        """
        self.mode = mode
        self.split_mode = split_mode
        self.train_dye = train_dye
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
        """Load Placenta dataset from pre-extracted windows."""
        if self.verbose:
            print(f"Loading Placenta dataset (mode: {self.split_mode})")

        # Check if we're in imagettes or cov directory
        data_dir = Path(self.data_dir)

        # If data_dir is 'cov', go up one level to find imagettes
        if data_dir.name == 'cov':
            imagettes_dir = data_dir.parent / 'imagettes'
        else:
            imagettes_dir = data_dir

        # Load data files
        windows_file = imagettes_dir / 'placenta_windows_data.npy'
        labels_file = imagettes_dir / 'placenta_windows_labels.npy'
        metadata_file = imagettes_dir / 'placenta_windows_metadata.json'

        if not windows_file.exists() or not labels_file.exists():
            raise FileNotFoundError(
                f"Data files not found in {imagettes_dir}\n"
                f"Expected: placenta_windows_data.npy, placenta_windows_labels.npy"
            )

        # Load arrays
        all_windows = np.load(str(windows_file))
        all_labels = np.load(str(labels_file))

        with open(metadata_file, 'r') as f:
            all_metadata = json.load(f)

        if self.verbose:
            print(f"  Loaded {len(all_windows)} samples")
            print(f"  Window shape: {all_windows.shape[1:]}")

        # Setup cov_dir
        if data_dir.name == 'cov':
            self.cov_dir = data_dir
        else:
            self.cov_dir = imagettes_dir.parent / "cov"

        if self.mode == 'cov' and not self.cov_dir.exists():
            raise FileNotFoundError(f"Covariance directory not found: {self.cov_dir}")

        # Store windows for raw mode
        self.all_windows = all_windows

        # Setup class mapping
        self.classes = self.CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Group samples by class and dye type
        if self.split_mode == 'mixed':
            samples = self._create_mixed_samples(all_labels, all_metadata)
        elif self.split_mode == 'domain_adaptation':
            samples = self._create_domain_adaptation_samples(all_labels, all_metadata)
        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")

        # Limit samples per class if requested
        self.samples = self._limit_samples_per_class(samples)

        if self.verbose:
            print(f"Final dataset: {len(self.samples)} samples, {len(self.classes)} classes")
            self._print_class_distribution()

        # Preload data if requested
        if self.preload and self.mode == 'cov':
            self._preload_all()

    def _create_mixed_samples(self, all_labels, all_metadata):
        """Create samples for mixed mode (shuffle both dye types together)."""
        samples_by_class = {name: [] for name in self.classes}

        for idx in range(len(all_labels)):
            label = all_labels[idx]
            meta = all_metadata[idx]

            if label not in self.class_to_idx:
                continue

            class_idx = self.class_to_idx[label]

            samples_by_class[label].append({
                'index': idx,
                'class_idx': class_idx,
                'class_name': label,
                'dye_type': meta['dye_type'],
                'sample_id': meta['sample_id'],
                'metadata': meta
            })

        return samples_by_class

    def _create_domain_adaptation_samples(self, all_labels, all_metadata):
        """
        Create samples for domain adaptation mode.
        One dye type for train, other for test.
        """
        # Count samples per class per dye type
        dye_class_counts = {dye: {cls: 0 for cls in self.classes} for dye in self.DYE_TYPES}

        for idx in range(len(all_labels)):
            label = all_labels[idx]
            meta = all_metadata[idx]
            dye = meta['dye_type']

            if label in self.class_to_idx:
                dye_class_counts[dye][label] += 1

        # Auto-select train dye if not specified (choose one with most samples)
        if self.train_dye is None:
            total_counts = {dye: sum(counts.values()) for dye, counts in dye_class_counts.items()}
            self.train_dye = max(total_counts, key=total_counts.get)

            if self.verbose:
                print(f"  Auto-selected train_dye: {self.train_dye}")
                print(f"    ICG: {total_counts['ICG']} samples")
                print(f"    red_blue: {total_counts['red_blue']} samples")

        # Determine test dye
        test_dye = 'red_blue' if self.train_dye == 'ICG' else 'ICG'

        # Store dye info for dataset splits
        self.domain_info = {
            'train_dye': self.train_dye,
            'test_dye': test_dye,
            'train_counts': dye_class_counts[self.train_dye],
            'test_counts': dye_class_counts[test_dye]
        }

        # Create samples grouped by class (will be split by dye type during train/test split)
        samples_by_class = {name: [] for name in self.classes}

        for idx in range(len(all_labels)):
            label = all_labels[idx]
            meta = all_metadata[idx]

            if label not in self.class_to_idx:
                continue

            class_idx = self.class_to_idx[label]
            dye = meta['dye_type']

            samples_by_class[label].append({
                'index': idx,
                'class_idx': class_idx,
                'class_name': label,
                'dye_type': dye,
                'sample_id': meta['sample_id'],
                'metadata': meta,
                'is_train_domain': (dye == self.train_dye)
            })

        return samples_by_class

    def _print_class_distribution(self):
        """Print class distribution."""
        if not self.verbose:
            return

        class_counts = {}
        dye_counts = {dye: {} for dye in self.DYE_TYPES}

        for sample in self.samples:
            class_name = sample['class_name']
            dye = sample['dye_type']

            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            dye_counts[dye][class_name] = dye_counts[dye].get(class_name, 0) + 1

        print("\nClass distribution:")
        for class_name in self.classes:
            count = class_counts.get(class_name, 0)
            percentage = count / len(self.samples) * 100 if self.samples else 0
            print(f"  {class_name:15s}: {count:5d} samples ({percentage:5.1f}%)")

        if self.split_mode == 'domain_adaptation':
            print(f"\nDomain adaptation info:")
            print(f"  Train dye: {self.train_dye}")
            print(f"  Test dye: {self.domain_info['test_dye']}")
            print(f"\n  Distribution by dye type:")
            for dye in self.DYE_TYPES:
                total = sum(dye_counts[dye].values())
                print(f"    {dye:10s}: {total:5d} samples")
                for cls in self.classes:
                    count = dye_counts[dye].get(cls, 0)
                    if count > 0:
                        print(f"      {cls:15s}: {count:5d}")

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
            meta = sample['metadata']

            # Build filename from metadata
            sample_id = meta['sample_id']
            dye_type = meta['dye_type']
            class_name = meta['class']
            y, x = meta['position']
            filename = f"{sample_id}_{dye_type}_{class_name}_y{y}_x{x}.pt"

            cov_path = self.cov_dir / filename
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
            meta = sample['metadata']
            sample_id = meta['sample_id']
            dye_type = meta['dye_type']
            class_name = meta['class']
            y, x = meta['position']
            filename = f"{sample_id}_{dye_type}_{class_name}_y{y}_x{x}.pt"

            cov_path = self.cov_dir / filename

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
                data_tensor = self.cov_estimator.from_image(window_tensor)
                if isinstance(data_tensor, np.ndarray):
                    data_tensor = torch.from_numpy(data_tensor).double()
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

    def get_domain_split_indices(self) -> Tuple[List[int], List[int]]:
        """
        Get indices for domain adaptation split.

        Returns:
            train_indices: Indices for train domain (train_dye)
            test_indices: Indices for test domain (other dye)
        """
        if self.split_mode != 'domain_adaptation':
            raise ValueError("get_domain_split_indices only available in domain_adaptation mode")

        train_indices = []
        test_indices = []

        for idx, sample in enumerate(self.samples):
            if sample['is_train_domain']:
                train_indices.append(idx)
            else:
                test_indices.append(idx)

        return train_indices, test_indices
