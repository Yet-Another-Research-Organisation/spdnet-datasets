"""
HyperLeaf Dataset - Hyperspectral leaf imagery.
Supports classification task with 4 barley varieties.
"""

import pandas as pd
import numpy as np
import torch
import tifffile
from typing import Tuple

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.manager import DatasetManager
from spdnet_datasets.estimator import EstimateCovariance


@DatasetManager.register_dataset('hyperleaf')
class HyperLeafDataset(BaseDataset):
    """
    HyperLeaf hyperspectral dataset for classification.

    Supports three classification tasks:
    - 'cultivar': 4 barley varieties (Heerup, Kvium, Rembrandt, Sheriff)
    - 'fertilizer': 3 fertilizer levels (0.0, 0.5, 1.0)
    - 'combined': 12 classes (3 fertilizer × 4 cultivar combinations)

    Images are hyperspectral with 204 bands
    Can load either pre-computed covariance matrices or raw images
    """

    CULTIVAR_NAMES = ['Heerup', 'Kvium', 'Rembrandt', 'Sheriff']
    FERTILIZER_NAMES = ['0.0', '0.5', '1.0']

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        task: str = 'cultivar',  # 'cultivar', 'fertilizer', or 'combined'
        mode: str = 'cov',  # 'cov' for covariance, 'raw' for images
        cov_method: str = 'scm',  # 'scm' or 'ledoit_wolf'
        exclude_zero: bool = True,
        preload: bool = False,  # Disabled by default to avoid contention in multirun
        **kwargs
    ):
        """
        Initialize HyperLeaf dataset.

        Args:
            data_dir: Path to HyperLeaf folder
            split: 'train' or 'test'
            task: Classification task - 'cultivar' (4 classes), 'fertilizer' (3 classes),
                  or 'combined' (12 classes)
            mode: 'cov' for pre-computed covariances, 'raw' for computing from images
            cov_method: 'scm' (empirical) or 'ledoit_wolf'
            exclude_zero: Exclude zero pixels when computing covariance
            preload: If True, load all data to RAM at init (faster training, more RAM)
            **kwargs: Additional parameters from config
        """
        self.split = split
        self.task = task
        self.mode = mode
        self.cov_method = cov_method
        self.exclude_zero = exclude_zero
        self.preload = preload
        self.target_size = (204, 204)  # Covariance matrices are 204x204
        self._cache = {}  # Cache for preloaded data

        # Validate task
        if self.task not in ['cultivar', 'fertilizer', 'combined']:
            raise ValueError(f"Task must be 'cultivar', 'fertilizer', or 'combined', got {self.task}")

        # Set class names based on task
        if self.task == 'cultivar':
            self.CLASS_NAMES = self.CULTIVAR_NAMES
        elif self.task == 'fertilizer':
            self.CLASS_NAMES = self.FERTILIZER_NAMES
        else:  # combined
            # Create combined class names: "Fertilizer_Cultivar"
            self.CLASS_NAMES = [
                f"{fert}_{cult}"
                for fert in self.FERTILIZER_NAMES
                for cult in self.CULTIVAR_NAMES
            ]

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

        # Preload data to RAM if requested
        if self.preload and self.mode == 'cov':
            self._preload_all()

    def _load_data(self):
        """Load CSV and prepare samples."""
        # Validate split
        if self.split not in ['train', 'test']:
            raise ValueError(f"Split must be 'train' or 'test', got {self.split}")

        # Load CSV
        csv_path = self.data_dir / f"{self.split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # Check if cov directory exists
        self.cov_dir = self.data_dir / "cov"
        if self.mode == 'cov' and not self.cov_dir.exists():
            raise FileNotFoundError(f"Covariance directory not found: {self.cov_dir}")

        df = pd.read_csv(csv_path, dtype={'ImageId': str})

        # Check if labels exist (train has labels, test doesn't)
        self.has_labels = self.split == 'train'

        if self.verbose:
            print(f"Loaded {len(df)} samples for {self.split} split")
            if self.has_labels and len(df) > 0:
                print(f"Dataset has labels: {self.CLASS_NAMES}")

        if self.has_labels:
            # Setup classes
            self.classes = self.CLASS_NAMES
            self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

            # Extract labels based on task
            if self.task == 'cultivar':
                # Convert one-hot cultivar columns to class indices
                class_columns = self.CULTIVAR_NAMES
                class_labels = df[class_columns].values.argmax(axis=1)
                label_names = [self.CULTIVAR_NAMES[idx] for idx in class_labels]

            elif self.task == 'fertilizer':
                # Map fertilizer values to class indices
                fertilizer_to_idx = {0.0: 0, 0.5: 1, 1.0: 2}
                class_labels = df['Fertilizer'].map(fertilizer_to_idx).values
                label_names = [self.FERTILIZER_NAMES[idx] for idx in class_labels]

            else:  # combined
                # Create combined labels: fertilizer_idx * 4 + cultivar_idx
                fertilizer_to_idx = {0.0: 0, 0.5: 1, 1.0: 2}
                fertilizer_labels = df['Fertilizer'].map(fertilizer_to_idx).values
                cultivar_labels = df[self.CULTIVAR_NAMES].values.argmax(axis=1)
                class_labels = fertilizer_labels * 4 + cultivar_labels
                label_names = [self.CLASS_NAMES[idx] for idx in class_labels]

            # Create samples list
            samples_by_class = {name: [] for name in self.classes}

            for idx, row in df.iterrows():
                image_id = row['ImageId']
                class_idx = class_labels[idx]
                class_name = label_names[idx]

                samples_by_class[class_name].append({
                    'image_id': image_id,
                    'class_name': class_name,
                    'class_idx': class_idx
                })

            # Limit samples per class if requested
            self.samples = self._limit_samples_per_class(samples_by_class)

            # Print statistics
            if self.verbose:
                print(f"Final dataset: {len(self.samples)} samples, {len(self.classes)} classes")
                self._print_class_distribution()
        else:
            # Test set - no labels
            self.classes = []
            self.class_to_idx = {}
            self.samples = [
                {'image_id': row['ImageId']}
                for _, row in df.iterrows()
            ]
            if self.verbose:
                print(f"Test dataset: {len(self.samples)} samples (no labels)")

    def _print_class_distribution(self):
        """Print class distribution."""
        if not self.has_labels or not self.verbose:
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

        images_dir = self.data_dir / 'cov'
        for sample in self.samples:
            image_id = sample['image_id']
            image_path = images_dir / f"{image_id}.pt"
            if image_path.exists():
                cov_matrix = torch.load(image_path, weights_only=False)
                if isinstance(cov_matrix, np.ndarray):
                    cov_matrix = torch.from_numpy(cov_matrix).double()
                elif cov_matrix.dtype != torch.float64:
                    cov_matrix = cov_matrix.double()
                self._cache[image_id] = cov_matrix

        if self.verbose:
            print(f"Preloaded {len(self._cache)} matrices to RAM")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Returns:
            Tuple of (covariance_matrix, class_index) if has_labels
            Or just covariance_matrix if test set
        """
        sample = self.samples[idx]
        image_id = sample['image_id']

        # Check cache first (for preloaded data)
        if image_id in self._cache:
            cov_matrix = self._cache[image_id]
        elif self.mode == 'cov':
            images_dir = self.data_dir / 'cov'
            image_path = images_dir / f"{image_id}.pt"

            # Load pre-computed covariance
            if not image_path.exists():
                raise FileNotFoundError(f"Covariance file not found: {image_path}")

            cov_matrix = torch.load(image_path, weights_only=False)

            if isinstance(cov_matrix, np.ndarray):
                cov_matrix = torch.from_numpy(cov_matrix).double()
            elif cov_matrix.dtype != torch.float64:
                cov_matrix = cov_matrix.double()

        else:  # mode == 'raw'
            images_dir = self.data_dir / 'images'
            image_path = images_dir / f"{image_id}.tiff"

            # Load hyperspectral image
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            try:
                # Load with tifffile (better for scientific data)
                image_array = tifffile.imread(str(image_path))
            except Exception as e:
                raise RuntimeError(f"Error loading image {image_path}: {e}")

            # Convert to float and move bands to last dimension
            image_array = image_array.astype(np.float32)
            image_array = np.moveaxis(image_array, 0, -1)  # (H, W, C)

            # Compute covariance
            cov_matrix = self.cov_estimator.from_image(image_array)
            cov_matrix = torch.from_numpy(cov_matrix).float()

        # Apply transform if specified
        if self.transform is not None:
            cov_matrix = self.transform(cov_matrix)

        # Return with or without label
        if self.has_labels:
            class_idx = sample['class_idx']
            return cov_matrix, class_idx
        else:
            return cov_matrix
