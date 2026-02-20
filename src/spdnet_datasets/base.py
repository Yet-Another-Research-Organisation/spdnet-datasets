"""
Base Dataset class for all SPDNet benchmark datasets.
Provides common functionality and interface for data loading and splitting.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets.

    Common attributes:
        data_dir: Path to dataset directory
        classes: List of class names
        class_to_idx: Mapping from class name to index
        transform: Optional transform to apply
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        data_dir: str,
        classes: Optional[List[str]] = None,
        max_samples_per_class: Optional[int] = None,
        max_classes: Optional[int] = None,
        transform: Optional[callable] = None,
        seed: int = 42,
        verbose: bool = True,
        **kwargs  # Accept any additional parameters
    ):
        """
        Initialize base dataset.

        Args:
            data_dir: Path to dataset directory
            classes: Optional list of classes to use (None = all)
            max_samples_per_class: Maximum samples per class
            max_classes: Maximum number of classes to use
            transform: Optional transform to apply
            seed: Random seed
            verbose: Whether to print information during loading
            **kwargs: Additional dataset-specific parameters
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.max_classes = max_classes
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)

        # Will be set by subclasses
        self.classes = []
        self.class_to_idx = {}
        self.samples = []

        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    @abstractmethod
    def _load_data(self):
        """Load dataset files and populate self.samples, self.classes, etc."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample."""
        pass

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)

    def _limit_classes(self, all_classes: List[str]) -> List[str]:
        """Limit number of classes if max_classes is set."""
        if self.max_classes is not None and self.max_classes < len(all_classes):
            rng_classes = np.random.RandomState(42)  # Fixed seed for class selection
            selected = rng_classes.choice(all_classes, size=self.max_classes, replace=False).tolist()
            if self.verbose:
                print(f"Using {self.max_classes}/{len(all_classes)} classes: {sorted(selected)}")
            return sorted(selected)
        return all_classes

    def _limit_samples_per_class(self, samples_by_class: Dict[str, List]) -> List:
        """Limit samples per class if max_samples_per_class is set."""
        limited_samples = []
        for class_name, class_samples in samples_by_class.items():
            if self.max_samples_per_class and len(class_samples) > self.max_samples_per_class:
                indices = self.rng.choice(
                    len(class_samples),
                    size=self.max_samples_per_class,
                    replace=False
                )
                selected = [class_samples[i] for i in indices]
                limited_samples.extend(selected)
                if self.verbose:
                    print(f"Class {class_name}: {len(class_samples)} -> {self.max_samples_per_class} samples")
            else:
                limited_samples.extend(class_samples)
                if self.verbose:
                    print(f"Class {class_name}: {len(class_samples)} samples (not limited)")
        return limited_samples

    def get_num_classes(self) -> int:
        """Return number of classes."""
        return len(self.classes)


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    shuffle: bool = True,
    seed: int = 42,
    stratified: bool = False,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split dataset and create train/val/test dataloaders.

    Args:
        dataset: The dataset to split
        batch_size: Batch size for dataloaders
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        persistent_workers: Keep workers alive between epochs
        shuffle: Whether to shuffle training data
        seed: Random seed for splitting
        stratified: Whether to use stratified splitting (ensures all classes in each split)
        verbose: Whether to print split information

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    total_size = len(dataset)

    if stratified:
        # Extract all labels from dataset
        all_labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            # Handle different dataset formats
            if isinstance(sample, dict):
                label = sample['label']
            elif isinstance(sample, tuple):
                label = sample[1]  # (data, label) format
            else:
                raise ValueError(f"Unsupported dataset format: {type(sample)}")

            if isinstance(label, torch.Tensor):
                label = label.item()
            all_labels.append(label)

        all_labels = np.array(all_labels)
        all_indices = np.arange(len(dataset))

        if verbose:
            print(f"\nStratified split - Original class distribution:")
            unique, counts = np.unique(all_labels, return_counts=True)
            for cls, count in zip(unique, counts):
                print(f"  Class {cls}: {count} samples ({100*count/len(all_labels):.1f}%)")

        # Step 1: Split off test set first
        indices_temp, indices_test, labels_temp, labels_test = train_test_split(
            all_indices,
            all_labels,
            test_size=test_ratio,
            stratify=all_labels,
            random_state=seed
        )

        # Step 2: Split remaining into train/val
        # Adjust val_ratio since we're working with remaining data after test split
        val_ratio_adjusted = val_ratio / (1 - test_ratio)

        indices_train, indices_val, labels_train, labels_val = train_test_split(
            indices_temp,
            labels_temp,
            test_size=val_ratio_adjusted,
            stratify=labels_temp,
            random_state=seed
        )

        # Create Subset datasets with the stratified indices
        train_ds = Subset(dataset, indices_train.tolist())
        val_ds = Subset(dataset, indices_val.tolist())
        test_ds = Subset(dataset, indices_test.tolist())

        if verbose:
            print("\nStratified dataset split:")
            print(f"  Train: {len(train_ds)} samples")
            print(f"  Val:   {len(val_ds)} samples")
            print(f"  Test:  {len(test_ds)} samples")

            # Print class distribution for each split
            print("\n  Class distribution per split:")
            print(f"  {'Class':<10} {'Train':<15} {'Val':<15} {'Test':<15}")
            print(f"  {'-'*55}")

            for cls in sorted(np.unique(all_labels)):
                train_count = np.sum(labels_train == cls)
                val_count = np.sum(labels_val == cls)
                test_count = np.sum(labels_test == cls)
                train_pct = 100 * train_count / len(train_ds)
                val_pct = 100 * val_count / len(val_ds)
                test_pct = 100 * test_count / len(test_ds)
                print(f"  {cls:<10} {train_count:>4} ({train_pct:>4.1f}%)   {val_count:>4} ({val_pct:>4.1f}%)   {test_count:>4} ({test_pct:>4.1f}%)")
    else:
        # Original non-stratified split
        test_size = int(total_size * test_ratio)
        val_size = int(total_size * val_ratio)
        train_size = total_size - test_size - val_size

        # Split dataset
        train_ds, val_ds, test_ds = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        if verbose:
            print("\nDataset split:")
            print(f"  Train: {len(train_ds)} samples")
            print(f"  Val:   {len(val_ds)} samples")
            print(f"  Test:  {len(test_ds)} samples")

            # Show class distribution in each split if dataset has classes
            if hasattr(dataset, 'classes') and len(dataset.classes) > 0:
                _print_split_distribution(dataset, train_ds, val_ds, test_ds)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    return train_loader, val_loader, test_loader


def _print_split_distribution(dataset, train_ds, val_ds, test_ds):
    """Print class distribution for each split."""
    # Get labels for each split
    train_labels = [dataset.samples[i]['class_name'] if isinstance(dataset.samples[i], dict)
                    else dataset.classes[dataset[i][1]]
                    for i in train_ds.indices]
    val_labels = [dataset.samples[i]['class_name'] if isinstance(dataset.samples[i], dict)
                  else dataset.classes[dataset[i][1]]
                  for i in val_ds.indices]
    test_labels = [dataset.samples[i]['class_name'] if isinstance(dataset.samples[i], dict)
                   else dataset.classes[dataset[i][1]]
                   for i in test_ds.indices]

    # Count unique classes
    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    val_unique, val_counts = np.unique(val_labels, return_counts=True)
    test_unique, test_counts = np.unique(test_labels, return_counts=True)

    print("\n  Class distribution per split:")
    print(f"  {'Class':<20} {'Train':<10} {'Val':<10} {'Test':<10}")
    print(f"  {'-'*50}")

    all_classes = sorted(set(train_unique) | set(val_unique) | set(test_unique))
    for cls in all_classes:
        train_c = train_counts[list(train_unique).index(cls)] if cls in train_unique else 0
        val_c = val_counts[list(val_unique).index(cls)] if cls in val_unique else 0
        test_c = test_counts[list(test_unique).index(cls)] if cls in test_unique else 0
        print(f"  {cls:<20} {train_c:<10} {val_c:<10} {test_c:<10}")
