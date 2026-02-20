"""
Rices90 Dataset - Vietnamese rice varieties classification.
90 classes of rice varieties with 256x256 covariance matrices.
"""

import torch
from typing import Tuple
import warnings

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.manager import DatasetManager


@DatasetManager.register_dataset('rices90')
class Rices90Dataset(BaseDataset):
    """
    Dataset for Rices_90 with pre-computed covariance matrices.

    Class names are extracted from filenames: "CLASS-VARIANT_INDEX.pt"
    Example: "91RH-02_000.pt" -> class "91RH"

    90 classes of Vietnamese rice varieties with 256x256 covariance matrices.
    """

    # All 90 rice variety class names
    CLASS_NAMES = [
        '91RH', '9d', 'A128', 'AH1000', 'BacThomSo7', 'BC15', 'BQ10', 'BT6', 'CH12',
        'CL61', 'CNC12', 'CS6', 'CT286', 'CTX30', 'DA1', 'DaiThom8', 'DMV58', 'DT52',
        'DT66', 'DTH155', 'DTL2', 'DV108', 'GiaLoc301', 'GS55R', 'H229', 'HaNa39',
        'HaPhat28', 'HoangLong', 'HongQuang15', 'HS1', 'HT18', 'HungDan1', 'KB16',
        'KB27', 'KB6', 'KhangDan18', 'KimCuong111', 'KL25', 'KN5', 'LDA8', 'LocTroi183',
        'LTH35', 'MT15', 'MyHuong88', 'N54', 'N97', 'N98', 'NBK', 'NBP', 'NBT1',
        'NC2', 'NC7', 'ND9', 'NDC1', 'NepCoTien', 'NepDacSanLienHoa', 'NepHongNgoc',
        'NepKB19', 'NepPhatQuy', 'NepThomBacHai', 'NepThomHungYen', 'NH92', 'NM14',
        'NN4B', 'NPT1', 'NPT3', 'NT16', 'NTP', 'NV1', 'PC10', 'PD211', 'R068',
        'R998KBL', 'SHPT1', 'SVN1', 'TB13', 'TB14', 'TC10', 'TC11', 'ThuanViet2',
        'TQ14', 'TQ36', 'TruongXuan1', 'TruongXuanHQ', 'VietHuong8', 'VietThom8',
        'VinhPhuc1', 'VS1', 'VS5', 'VS6'
    ]

    def __init__(self, **kwargs):
        """Initialize Rices90 dataset."""
        super().__init__(**kwargs)
        self.target_size = (256, 256)
        self._load_data()

    @property
    def num_classes(self):
        """Return number of classes."""
        return len(self.classes)

    def _load_data(self):
        """Load .pt files and extract class information from filenames."""
        # Find covariance directory
        cov_dir = self.data_dir / "Rices_90_cov"
        if not cov_dir.exists():
            cov_dir = self.data_dir  # Fallback to data_dir

        # Find all .pt files (exclude chessboard files)
        pt_files = [
            f for f in cov_dir.glob("*.pt")
            if not f.stem.startswith('chessboard')
        ]

        if len(pt_files) == 0:
            raise ValueError(f"No .pt files found in {cov_dir}")

        if self.verbose:
            print(f"Found {len(pt_files)} covariance files")

        # Extract class names from filenames
        # Format: "CLASS-VARIANT_INDEX.pt" -> class "CLASS"
        file_info = []
        for pt_file in pt_files:
            filename = pt_file.stem
            try:
                class_name = filename.split("-")[0]  # First part before dash
                file_info.append({
                    'path': pt_file,
                    'class_name': class_name,
                    'filename': filename
                })
            except IndexError:
                warnings.warn(f"Invalid filename (no dash): {filename}")
                continue

        # Get unique classes
        all_classes = sorted(list(set(info['class_name'] for info in file_info)))
        if self.verbose:
            print(f"Found classes: {len(all_classes)} classes")

        # Limit number of classes if requested
        selected_classes = self._limit_classes(all_classes)

        # Filter files by selected classes
        file_info = [info for info in file_info if info['class_name'] in selected_classes]

        # Create class to index mapping
        self.classes = selected_classes
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Group samples by class
        samples_by_class = {}
        for info in file_info:
            class_name = info['class_name']
            if class_name not in samples_by_class:
                samples_by_class[class_name] = []
            samples_by_class[class_name].append(info)

        # Limit samples per class if requested
        self.samples = self._limit_samples_per_class(samples_by_class)

        # Print final statistics
        if self.verbose:
            print("\nFinal dataset:")
            print(f"  - {len(self.samples)} samples")
            print(f"  - {len(self.classes)} classes")
            print("  - Matrix dimensions: 256x256")

            self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class distribution."""
        if not self.verbose:
            return

        class_counts = {}
        for sample in self.samples:
            class_name = sample['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print("\nClass distribution:")
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            percentage = count / len(self.samples) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample and its label.

        Args:
            idx: Sample index

        Returns:
            Tuple of (covariance_matrix, class_index)
        """
        sample = self.samples[idx]

        # Load covariance matrix
        try:
            cov_matrix = torch.load(sample['path'], weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Error loading {sample['path']}: {e}")

        # Verify dimensions
        if cov_matrix.shape != self.target_size:
            raise ValueError(
                f"Invalid covariance matrix: {cov_matrix.shape} != {self.target_size}"
            )

        # Apply transform if specified
        if self.transform is not None:
            cov_matrix = self.transform(cov_matrix)

        # Get class index
        class_idx = self.class_to_idx[sample['class_name']]

        return cov_matrix, class_idx
