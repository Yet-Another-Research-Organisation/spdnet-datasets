"""
Kaggle Wheat Disease Dataset – Hyperspectral classification.
3 classes: Health, Rust, Other.

Train folder is split into train/val/test (0.75 / 0.1 / 0.15) with stratification.
Val folder (unlabelled) is used for final Kaggle submission inference.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.manager import DatasetManager


@DatasetManager.register_dataset("kaggle_wheat")
class KaggleWheatDataset(BaseDataset):
    """
    Kaggle Wheat Disease hyperspectral dataset.

    Classes (derived from filename prefix in train/HS):
        - Health  (healthy wheat)
        - Rust    (rust-infected)
        - Other   (other conditions / background)

    Modes:
        - 'cov': load pre-computed 107×107 covariance matrices (.pt)
        - 'inference': load covariance matrices from val set (no labels)

    Directory layout expected:
        <data_dir>/
            cov_train/        ← precomputed covariances of train/HS
            cov_val/          ← precomputed covariances of val/HS
    """

    CLASS_NAMES = ["Health", "Other", "Rust"]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        mode: str = "cov",
        preload: bool = False,
        **kwargs,
    ):
        """
        Args:
            data_dir: Root of Kaggle_Prepared (contains cov_train/, cov_val/).
            split: 'train' loads labelled training covariances.
                   'val_kaggle' loads unlabelled val covariances (for submission).
            mode: 'cov' (only supported mode).
            preload: Pre-load all .pt files into RAM.
            **kwargs: Forwarded to BaseDataset.
        """
        self.split = split
        self.mode = mode
        self.preload = preload
        self.target_size = (107, 107)
        self._cache: dict[str, torch.Tensor] = {}

        super().__init__(data_dir=data_dir, **kwargs)

        self._load_data()

        if self.preload and self.mode == "cov":
            self._preload_all()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Populate self.samples, self.classes, etc."""
        if self.split == "train":
            self._load_train()
        elif self.split == "val_kaggle":
            self._load_val_kaggle()
        else:
            raise ValueError(
                f"split must be 'train' or 'val_kaggle', got '{self.split}'"
            )

    def _load_train(self):
        """Load labelled covariances from cov_train/."""
        cov_dir = self.data_dir / "cov_train"
        if not cov_dir.exists():
            raise FileNotFoundError(f"cov_train directory not found: {cov_dir}")

        self.cov_dir = cov_dir
        self.classes = list(self.CLASS_NAMES)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.has_labels = True

        # Derive label from filename prefix: Health_hyper_*.pt, Rust_hyper_*.pt, Other_hyper_*.pt
        all_pts = sorted(cov_dir.glob("*.pt"))
        if len(all_pts) == 0:
            raise FileNotFoundError(f"No .pt files found in {cov_dir}")

        samples_by_class: dict[str, list] = {c: [] for c in self.classes}

        for pt_path in all_pts:
            stem = pt_path.stem  # e.g. Health_hyper_42
            class_name = self._class_from_filename(stem)
            if class_name is None:
                if self.verbose:
                    print(f"  Skipping unknown class file: {pt_path.name}")
                continue

            class_idx = self.class_to_idx[class_name]
            samples_by_class[class_name].append(
                {
                    "image_id": stem,
                    "class_name": class_name,
                    "class_idx": class_idx,
                }
            )

        self.samples = self._limit_samples_per_class(samples_by_class)

        if self.verbose:
            print(f"Loaded {len(self.samples)} labelled samples from {cov_dir}")
            self._print_class_distribution()

    def _load_val_kaggle(self):
        """Load unlabelled covariances from cov_val/ (Kaggle submission)."""
        cov_dir = self.data_dir / "cov_val"
        if not cov_dir.exists():
            raise FileNotFoundError(f"cov_val directory not found: {cov_dir}")

        self.cov_dir = cov_dir
        self.classes = []
        self.class_to_idx = {}
        self.has_labels = False

        all_pts = sorted(cov_dir.glob("*.pt"))
        if len(all_pts) == 0:
            raise FileNotFoundError(f"No .pt files found in {cov_dir}")

        self.samples = [{"image_id": pt.stem} for pt in all_pts]

        if self.verbose:
            print(f"Loaded {len(self.samples)} unlabelled val samples from {cov_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _class_from_filename(stem: str) -> str | None:
        """Extract class name from file stem, e.g. 'Health_hyper_42' → 'Health'."""
        for cls in KaggleWheatDataset.CLASS_NAMES:
            if stem.startswith(cls):
                return cls
        return None

    def _print_class_distribution(self):
        if not self.has_labels or not self.verbose:
            return
        class_counts: dict[str, int] = {}
        for s in self.samples:
            class_counts[s["class_name"]] = class_counts.get(s["class_name"], 0) + 1
        print("\nClass distribution:")
        for cn in self.classes:
            count = class_counts.get(cn, 0)
            pct = 100 * count / len(self.samples) if self.samples else 0
            print(f"  {cn}: {count} samples ({pct:.1f}%)")

    def _preload_all(self):
        if self.verbose:
            print(f"Preloading {len(self.samples)} covariance matrices to RAM...")
        for sample in self.samples:
            image_id = sample["image_id"]
            pt_path = self.cov_dir / f"{image_id}.pt"
            if pt_path.exists():
                cov = torch.load(pt_path, weights_only=False)
                if isinstance(cov, np.ndarray):
                    cov = torch.from_numpy(cov).double()
                elif cov.dtype != torch.float64:
                    cov = cov.double()
                self._cache[image_id] = cov
        if self.verbose:
            print(f"Preloaded {len(self._cache)} matrices")

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image_id = sample["image_id"]

        if image_id in self._cache:
            cov_matrix = self._cache[image_id]
        else:
            pt_path = self.cov_dir / f"{image_id}.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"Covariance not found: {pt_path}")
            cov_matrix = torch.load(pt_path, weights_only=False)
            if isinstance(cov_matrix, np.ndarray):
                cov_matrix = torch.from_numpy(cov_matrix).double()
            elif cov_matrix.dtype != torch.float64:
                cov_matrix = cov_matrix.double()

        if self.transform is not None:
            cov_matrix = self.transform(cov_matrix)

        if self.has_labels:
            return cov_matrix, sample["class_idx"]
        else:
            return cov_matrix
