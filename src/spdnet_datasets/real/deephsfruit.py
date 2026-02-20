"""
DeepHS-Fruit Dataset - Hyperspectral fruit classification.
Supports fruit classification (5 types) and ripeness classification (3 states).
"""

import json
import numpy as np
import torch
from typing import Tuple, Dict, Optional
from pathlib import Path
from spectral.io import envi

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.manager import DatasetManager
from spdnet_datasets.estimator import EstimateCovariance


@DatasetManager.register_dataset('deephsfruit')
class DeepHSFruitDataset(BaseDataset):
    """
    DeepHS-Fruit hyperspectral dataset for fruit and ripeness classification.

    Supports two classification modes:
    - 'fruit': 5 fruit types (Avocado, Kaki, Kiwi, Mango, Papaya)
    - 'ripeness': 3 ripeness states (unripe, perfect/ripe, overripe)

    Dataset structure:
    - Images organized as: Fruit/Modality/Day/fruit_day_XX_side.hdr
    - Total VIS samples: ~3405 (Avocado: 1030, Kaki: 395, Kiwi: 1144, Mango: 562, Papaya: 274)
    - Labeled for ripeness: 636 samples (from annotations)
    """

    FRUIT_CLASSES = ['Avocado', 'Kaki', 'Kiwi', 'Mango', 'Papaya']

    # Ripeness states (basic 3-class)
    RIPENESS_CLASSES = ['unripe', 'perfect', 'overripe']  # 'ripe' maps to 'perfect'

    # Distribution from analysis (636 labeled samples):
    # perfect: 348 (54.7%), overripe: 161 (25.3%), unripe: 127 (20.0%)

    def __init__(
        self,
        data_dir: str,
        modality: str = 'VIS',
        classification_mode: str = 'fruit',
        annotations_file: Optional[str] = None,
        mode: str = 'cov',  # 'cov' for covariance, 'raw' for images
        cov_method: str = 'scm',
        exclude_zero: bool = True,
        preload: bool = False,
        **kwargs
    ):
        """
        Initialize DeepHS-Fruit dataset.

        Args:
            data_dir: Path to DeepHS-Fruit folder
            modality: 'VIS' or 'NIR' (only VIS supported currently)
            classification_mode: 'fruit' or 'ripeness'
            annotations_file: Path to annotations JSON (for ripeness mode)
            mode: 'cov' for pre-computed covariances, 'raw' for ENVI images
            cov_method: 'scm' or 'ledoit_wolf'
            exclude_zero: Exclude zero pixels when computing covariance
            preload: If True, load all data to RAM at init
            **kwargs: Additional parameters from config
        """
        self.modality = modality
        self.classification_mode = classification_mode
        self.mode = mode
        self.cov_method = cov_method
        self.exclude_zero = exclude_zero
        self.preload = preload
        self._cache = {}

        # Validate parameters
        if modality != 'VIS':
            raise ValueError(f"Only VIS modality is supported, got {modality}")

        if classification_mode not in ['fruit', 'ripeness']:
            raise ValueError(f"classification_mode must be 'fruit' or 'ripeness', got {classification_mode}")

        # Setup covariance estimator if needed
        if self.mode == 'raw':
            self.cov_estimator = EstimateCovariance(
                method=cov_method,
                remove_mean=True,
                exclude_zero=exclude_zero
            )

        # Load annotations for ripeness mode
        if classification_mode == 'ripeness':
            if annotations_file is None:
                annotations_file = Path(data_dir) / 'annotations-upd-2024-01-09' / 'annotations' / 'train_only_labeled_v2.json'
            self.annotations = self._load_annotations(annotations_file)
        else:
            self.annotations = None

        # Initialize base
        super().__init__(data_dir=data_dir, **kwargs)

        # Load data
        self._load_data()

        # Preload if requested
        if self.preload and self.mode == 'cov':
            self._preload_all()

    def _load_annotations(self, annotations_file: Path) -> Dict:
        """Load annotations JSON file."""
        with open(annotations_file, 'r') as f:
            return json.load(f)

    def _load_data(self):
        """Load dataset samples."""
        if self.verbose:
            print(f"Loading DeepHS-Fruit dataset ({self.modality}, {self.classification_mode} mode)")

        # Setup cov_dir for covariance mode
        self.cov_dir = self.data_dir / f"cov_{self.modality}"
        if self.mode == 'cov' and not self.cov_dir.exists():
            raise FileNotFoundError(f"Covariance directory not found: {self.cov_dir}")

        if self.classification_mode == 'fruit':
            self._load_fruit_mode()
        else:  # ripeness
            self._load_ripeness_mode()

        if self.verbose:
            print(f"Final dataset: {len(self.samples)} samples, {len(self.classes)} classes")
            self._print_class_distribution()

    def _load_fruit_mode(self):
        """Load samples for fruit classification (all samples)."""
        # Setup classes
        self.classes = self.FRUIT_CLASSES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Find all images/covariances organized by fruit
        samples_by_class = {fruit: [] for fruit in self.FRUIT_CLASSES}

        for fruit in self.FRUIT_CLASSES:
            fruit_dir = self.data_dir / fruit / self.modality

            if not fruit_dir.exists():
                if self.verbose:
                    print(f"  Warning: {fruit} directory not found")
                continue

            if self.mode == 'cov':
                # Use covariance files from cov_VIS directory
                # Files are named: fruit_VIS_day_XXXX_side.pt
                cov_files = list(self.cov_dir.glob(f"{fruit.lower()}_*.pt"))

                for cov_file in cov_files:
                    samples_by_class[fruit].append({
                        'cov_file': cov_file.name,
                        'fruit': fruit,
                        'class_idx': self.class_to_idx[fruit],
                        'class_name': fruit
                    })
            else:
                # Use ENVI .hdr files
                hdr_files = list(fruit_dir.rglob("*.hdr"))

                for hdr_file in hdr_files:
                    samples_by_class[fruit].append({
                        'hdr_file': hdr_file,
                        'fruit': fruit,
                        'class_idx': self.class_to_idx[fruit],
                        'class_name': fruit
                    })

        # Limit samples per class if requested
        self.samples = self._limit_samples_per_class(samples_by_class)

        if self.verbose:
            print(f"  Total samples: {len(self.samples)}")

    def _load_ripeness_mode(self):
        """Load samples for ripeness classification (only labeled samples)."""
        if self.annotations is None:
            raise ValueError("Annotations required for ripeness mode")

        # Setup classes
        self.classes = self.RIPENESS_CLASSES
        self.class_to_idx = {'unripe': 0, 'perfect': 1, 'overripe': 2}

        # Map 'ripe' to 'perfect'
        ripeness_mapping = {
            'unripe': 'unripe',
            'perfect': 'perfect',
            'ripe': 'perfect',  # Map ripe to perfect
            'overripe': 'overripe'
        }

        # Build lookup dict from annotations section: record_id -> ripeness_state
        ripeness_lookup = {}
        if 'annotations' in self.annotations:
            for annotation in self.annotations['annotations']:
                record_id = annotation['record_id']
                ripeness_state = annotation.get('ripeness_state', 'unknown')
                if ripeness_state in ripeness_mapping:
                    ripeness_lookup[record_id] = ripeness_mapping[ripeness_state]

        if self.verbose:
            print(f"  Found {len(ripeness_lookup)} labeled samples for ripeness")

        # Create samples grouped by class
        samples_by_class = {cls: [] for cls in self.classes}

        if self.mode == 'cov':
            # Match covariance files with annotations
            # Covariance files format: fruit_VIS_day_XX_YYYY_side.pt or fruit_VIS_day_mX_YY_ZZZZ_side.pt
            # where the second to last part is the record_id
            # IMPORTANT: Filter by camera_type to match self.modality
            # NOTE: Fruit name in filename may differ from JSON, use JSON as source of truth
            for cov_file in self.cov_dir.glob("*.pt"):
                # Parse filename
                parts = cov_file.stem.split('_')
                if len(parts) >= 5:
                    # Record ID is always the second to last part (before side)
                    try:
                        record_id = int(parts[-2])  # Second to last is record_id

                        if record_id in ripeness_lookup:
                            # Check that this record has the correct camera type
                            record = self.annotations['records'][record_id]
                            if record['camera_type'] != self.modality:
                                continue

                            # Use fruit name from JSON, not from filename
                            fruit = record['fruit']
                            ripeness_state = ripeness_lookup[record_id]
                            class_idx = self.class_to_idx[ripeness_state]
                            samples_by_class[ripeness_state].append({
                                'cov_file': cov_file,
                                'ripeness': ripeness_state,
                                'fruit': fruit,
                                'class_idx': class_idx,
                                'class_name': ripeness_state
                            })
                    except ValueError:
                        # Skip files that don't match expected format
                        continue
        else:
            # Match ENVI files with annotations
            for record_id, ripeness_state in ripeness_lookup.items():
                # Find the corresponding ENVI file from annotations
                record = self.annotations['records'][record_id]
                if record['camera_type'] != self.modality:
                    continue

                hdr_path = self.data_dir / record['files']['header_file']
                if not hdr_path.exists():
                    continue

                class_idx = self.class_to_idx[ripeness_state]
                samples_by_class[ripeness_state].append({
                    'hdr_file': hdr_path,
                    'ripeness': ripeness_state,
                    'fruit': record['fruit'],
                    'class_idx': class_idx,
                    'class_name': ripeness_state
                })

        # Limit samples per class if requested
        self.samples = self._limit_samples_per_class(samples_by_class)

        if self.verbose:
            print(f"  Labeled samples: {len(self.samples)}")

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
            cov_file = sample['cov_file']
            cov_path = self.cov_dir / cov_file
            if cov_path.exists():
                cov_matrix = torch.load(cov_path, weights_only=False)
                if isinstance(cov_matrix, np.ndarray):
                    cov_matrix = torch.from_numpy(cov_matrix).double()
                elif cov_matrix.dtype != torch.float64:
                    cov_matrix = cov_matrix.double()
                self._cache[cov_file] = cov_matrix

        if self.verbose:
            print(f"Preloaded {len(self._cache)} matrices to RAM")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Returns:
            Tuple of (data_tensor, class_index)
            data_tensor is covariance matrix (mode='cov') or image (mode='raw')
        """
        sample = self.samples[idx]

        # Check cache first (for preloaded covariances)
        if self.mode == 'cov':
            cov_file = sample['cov_file']

            if cov_file in self._cache:
                data_tensor = self._cache[cov_file]
            else:
                # Load pre-computed covariance
                cov_path = self.cov_dir / cov_file

                if not cov_path.exists():
                    raise FileNotFoundError(f"Covariance file not found: {cov_path}")

                data_tensor = torch.load(cov_path, weights_only=False)

                if isinstance(data_tensor, np.ndarray):
                    data_tensor = torch.from_numpy(data_tensor).double()
                elif data_tensor.dtype != torch.float64:
                    data_tensor = data_tensor.double()

        else:  # mode == 'raw'
            # Load ENVI image
            hdr_file = sample['hdr_file']

            if not hdr_file.exists():
                raise FileNotFoundError(f"Image file not found: {hdr_file}")

            try:
                # Load ENVI image
                bin_file = str(hdr_file).replace('.hdr', '.bin')
                img = envi.open(str(hdr_file), image=bin_file)
                image_array = img.load()
                image_array = np.array(image_array, dtype=np.float32)

                # Convert to tensor (bands, height, width)
                # ENVI format is (height, width, bands), we need (bands, height, width)
                data_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
            except Exception as e:
                raise RuntimeError(f"Error loading {hdr_file}: {e}")

        # Apply transform if specified
        if self.transform is not None:
            data_tensor = self.transform(data_tensor)

        class_idx = sample['class_idx']

        return data_tensor, class_idx

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)
