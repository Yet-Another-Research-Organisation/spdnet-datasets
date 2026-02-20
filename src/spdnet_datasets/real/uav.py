"""
UAV-HSI-Crop Dataset - Hyperspectral imagery from UAV for crop classification.
Supports classification task with multiple crop types.
Dataset from https://www.scidb.cn/en/detail?dataSetId=6de15e4ec9b74dacab12e29cb557f041
"""

import numpy as np
import torch
from typing import Tuple, Optional, List
from pathlib import Path

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.manager import DatasetManager
from spdnet_datasets.estimator import EstimateCovariance


@DatasetManager.register_dataset('uav')
class UAVDataset(BaseDataset):
    """
    UAV-HSI-Crop hyperspectral dataset for classification.

    Contains three geographical scenes: MJK_N, MJK_S, XJM
    Supports two split modes:
    - split_geographic=False: Standard random train/val/test split from all scenes
    - split_geographic=True: Geographic split with MJK_N (train), MJK_S (test)
      using only the 18 common labels between them
    """

    # Geographic split configuration (best combination found: MJK_N + MJK_S)
    TRAIN_SCENES = ['MJK_N']
    TEST_SCENES = ['MJK_S']
    # Common labels between MJK_N and MJK_S
    COMMON_LABELS = [1, 2, 3, 4, 7, 8, 10, 11, 14, 15, 16, 19, 20, 22, 23, 24, 25, 26]

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        split_geographic: bool = False,
        mode: str = 'raw',  # 'cov' for covariance, 'raw' for windows
        cov_method: str = 'scm',  # 'scm' or 'ledoit_wolf'
        exclude_zero: bool = True,
        preload: bool = False,
        **kwargs
    ):
        """
        Initialize UAV dataset.

        Args:
            data_dir: Path to UAV-HSI-Crop folder containing scene folders
            split: 'train', 'val', or 'test' (used only if split_geographic=False)
            split_geographic: If True, use geographic split (MJK_N train, MJK_S test)
                            If False, use standard random split from all scenes
            mode: 'cov' for pre-computed covariances, 'raw' for computing from windows
            cov_method: 'scm' (empirical) or 'ledoit_wolf'
            exclude_zero: Exclude zero pixels when computing covariance
            preload: If True, load all data to RAM at init (faster training, more RAM)
            **kwargs: Additional parameters from config
        """
        self.split = split
        self.split_geographic = split_geographic
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
        """Load UAV dataset from pre-extracted windows."""
        if self.split_geographic:
            self._load_geographic_split()
        else:
            self._load_standard_split()

    def _load_geographic_split(self):
        """Load with geographic split: MJK_N (train) vs MJK_S (test)."""
        if self.verbose:
            print(f"Loading UAV dataset with GEOGRAPHIC split")
            print(f"  Split: {self.split}")

        # Determine which scenes to load
        if self.split in ['train', 'val']:
            scenes_to_load = self.TRAIN_SCENES
        else:  # test
            scenes_to_load = self.TEST_SCENES

        # Load data from selected scenes
        all_windows = []
        all_labels = []

        for scene in scenes_to_load:
            scene_dir = self.data_dir / scene
            if not scene_dir.exists():
                raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

            windows_file = scene_dir / 'uav_windows_data.npy'
            labels_file = scene_dir / 'uav_windows_labels.npy'

            if not windows_file.exists() or not labels_file.exists():
                raise FileNotFoundError(f"Data files not found in {scene_dir}")

            windows = np.load(str(windows_file))
            labels = np.load(str(labels_file)).astype(int)

            # Filter to keep only common labels
            mask = np.isin(labels, self.COMMON_LABELS)
            windows = windows[mask]
            labels = labels[mask]

            all_windows.append(windows)
            all_labels.append(labels)

            if self.verbose:
                print(f"  Loaded scene {scene}: {len(windows)} samples (after filtering)")

        # Concatenate all scenes
        all_windows = np.concatenate(all_windows, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Setup cov_dir for covariance mode
        self.cov_dir = self.data_dir / "cov"
        if self.mode == 'cov' and not self.cov_dir.exists():
            raise FileNotFoundError(f"Covariance directory not found: {self.cov_dir}")

        # Store windows data for raw mode
        self.all_windows = all_windows

        # Remap labels to continuous indices [0, 17]
        self.label_mapping = {old_label: new_idx for new_idx, old_label in enumerate(sorted(self.COMMON_LABELS))}
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

        remapped_labels = np.array([self.label_mapping[label] for label in all_labels])

        # Setup classes
        self.classes = [f"Class_{label}" for label in sorted(self.COMMON_LABELS)]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Create samples with indices instead of windows
        self.samples = []
        for idx in range(len(all_windows)):
            self.samples.append({
                'index': idx,  # Store index instead of window data
                'class_idx': remapped_labels[idx],
                'class_name': self.classes[remapped_labels[idx]]
            })

        if self.verbose:
            print(f"Final dataset: {len(self.samples)} samples, {len(self.classes)} classes")
            self._print_class_distribution()

        # Preload data if requested
        if self.preload and self.mode == 'cov':
            self._preload_all()

    def _load_standard_split(self):
        """Load all scenes together for standard random split."""
        if self.verbose:
            print(f"Loading UAV dataset with STANDARD split")
            print(f"  Loading all scenes together")

        # Load all scenes
        scenes = ['MJK_N', 'MJK_S', 'XJM']
        all_windows = []
        all_labels = []

        for scene in scenes:
            scene_dir = self.data_dir / scene
            if not scene_dir.exists():
                if self.verbose:
                    print(f"  Warning: Scene {scene} not found, skipping")
                continue

            windows_file = scene_dir / 'uav_windows_data.npy'
            labels_file = scene_dir / 'uav_windows_labels.npy'

            if not windows_file.exists() or not labels_file.exists():
                if self.verbose:
                    print(f"  Warning: Data files not found for {scene}, skipping")
                continue

            windows = np.load(str(windows_file))
            labels = np.load(str(labels_file)).astype(int)

            all_windows.append(windows)
            all_labels.append(labels)

            if self.verbose:
                print(f"  Loaded scene {scene}: {len(windows)} samples")

        if not all_windows:
            raise FileNotFoundError(f"No scene data found in {self.data_dir}")

        # Concatenate all scenes
        all_windows = np.concatenate(all_windows, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Get all unique labels and remap to continuous indices
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
        self.classes = [f"Class_{label}" for label in unique_labels]
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
