"""
Base Dataset class for all SPDNet benchmark datasets.
Provides common functionality and interface for data loading and splitting.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .core.outputs import ALL_OUTPUTS, pack_outputs, validate_output_request
from .core.splits import RANDOM_SPLIT_DEFAULTS, _random_split, shares_groups

RANDOM = "random"  # name of the sample-level split added to every dataset.splits


def _caller_stacklevel():
    """``stacklevel`` of the first frame outside this package (for warnings)."""
    import sys

    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_globals.get("__name__", "").startswith("spdnet_datasets"):
        frame, level = frame.f_back, level + 1
    return level


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
        classes: list[str] | None = None,
        max_samples_per_class: int | None = None,
        max_classes: int | None = None,
        transform: Callable | None = None,
        seed: int = 42,
        verbose: bool = True,
        return_mean: bool = False,
        return_raw: bool = False,
        estimator: str = "scm",
        task: str = "fine",
        drop_classes: tuple = (),
        merge_classes: dict | None = None,
        balance: str | None = None,
        normalize: bool = True,
        random_split_options: dict | None = None,
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
            return_mean: Also return the per-channel mean (see ADR-0002 / core.outputs)
            return_raw: Also return the preprocessed raw data
            estimator: Covariance estimator ('scm', 'ledoit_wolf', 'tyler', 'student')
            task: 'fine', 'coarse' or 'regression' (see core.tasks)
            drop_classes: Classes removed from the dataset
            merge_classes: Mapping {old_class: new_class}
            balance: None, 'undersample', 'oversample' or 'weights'
            normalize: Apply the geometric scale fitted on the training split
            random_split_options: Options of the sample-level ``"random"`` split
                added to ``splits`` (keys ``val_ratio`` 0.1, ``test_ratio`` 0.2,
                ``seed`` = ``seed``, ``stratify`` True; see
                :func:`spdnet_datasets.core.splits.random_split`)
            **kwargs: Additional dataset-specific parameters
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.max_classes = max_classes
        self._requested_classes = classes  # explicit class list takes priority over max_classes
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)
        validate_output_request(self.OUTPUTS, return_mean, return_raw)
        self.return_mean = return_mean
        self.return_raw = return_raw
        self.estimator = estimator
        self.task = task
        self.drop_classes = tuple(drop_classes)
        self.merge_classes = merge_classes
        self.balance = balance
        self.normalize = normalize
        unknown = set(random_split_options or {}) - set(RANDOM_SPLIT_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown random_split_options {sorted(unknown)}; "
                             f"expected keys among {sorted(RANDOM_SPLIT_DEFAULTS)}")
        self.random_split_options = {**RANDOM_SPLIT_DEFAULTS, **(random_split_options or {})}
        if self.random_split_options["seed"] is None:
            self.random_split_options["seed"] = seed
        self.random_split_stratified = None  # set when the "random" split is built
        self.random_split_leaky = None  # idem: True if a group holds several samples
        self.split_leaky = False  # True when the selected split mixes groups (split="random")

        # Will be set by subclasses
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        self.groups = None  # one group id per sample (subject, patch, block...)
        self.splits = {}  # name -> {"train": idx, "val": idx, "test": idx}

        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    # Outputs available for this dataset (raw/mean-only SAR datasets override it)
    OUTPUTS = ALL_OUTPUTS

    def _pack(self, cov=None, mean=None, raw=None):
        """Assemble the input of one sample according to the output contract."""
        return pack_outputs(cov, mean, raw, return_mean=self.return_mean,
                            return_raw=self.return_raw, outputs=self.OUTPUTS)

    @abstractmethod
    def _load_data(self):
        """Load dataset files and populate self.samples, self.classes, etc."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a single sample."""
        pass

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)

    def _limit_classes(self, all_classes: list[str]) -> list[str]:
        """Limit classes: use explicit list if provided, else apply max_classes."""
        if self._requested_classes:
            missing = [c for c in self._requested_classes if c not in all_classes]
            if missing:
                raise ValueError(f"Requested classes not found in dataset: {missing}")
            if self.verbose:
                print(f"Using {len(self._requested_classes)} specified classes: {sorted(self._requested_classes)}")
            return sorted(self._requested_classes)
        if self.max_classes is not None and self.max_classes < len(all_classes):
            rng_classes = np.random.RandomState(self.seed)
            selected = rng_classes.choice(all_classes, size=self.max_classes, replace=False).tolist()
            if self.verbose:
                print(f"Using {self.max_classes}/{len(all_classes)} classes: {sorted(selected)}")
            return sorted(selected)
        return all_classes

    def _limit_samples_per_class(self, samples_by_class: dict[str, list]) -> list:
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

    def _make_random_split(self, labels, groups=None):
        """Sample-level ``"random"`` split of the loaded samples.

        ``labels`` are the final labels of the samples (after the task and the
        class selection); float labels (regression) are not stratified. Sets
        :attr:`random_split_stratified` and :attr:`random_split_leaky`.
        """
        o = self.random_split_options
        parts, self.random_split_stratified = _random_split(
            labels, val_ratio=o["val_ratio"], test_ratio=o["test_ratio"], seed=o["seed"],
            stratify=o["stratify"])
        self.random_split_leaky = shares_groups(groups)
        return parts

    def _check_selected_split(self, split, groups=None):
        """Set :attr:`split_leaky`; warn once when ``split="random"`` mixes groups."""
        self.split_leaky = split == RANDOM and shares_groups(groups)
        if self.split_leaky:
            warnings.warn(
                f"{type(self).__name__}(split='random'): sample-level split on a dataset "
                "whose groups hold several samples; samples of one group fall in train and "
                "test, so the scores are optimistic. Use split='recommended' for a "
                "leakage-free evaluation.", UserWarning, stacklevel=_caller_stacklevel())

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
    verbose: bool = True,
    split: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
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
        split: Name of a leakage-free split provided by the dataset
            (``dataset.splits[split]``, e.g. 'official' or 'recommended').
            When None, a sample-level random/stratified split is made; it
            warns if the dataset provides groups, since samples of one group
            may then fall on both sides.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    total_size = len(dataset)

    if split is not None:
        splits = getattr(dataset, "splits", {}) or {}
        if split not in splits:
            raise ValueError(f"Unknown split {split!r}; available: {sorted(splits)}")
        parts = splits[split]
        fitted = getattr(dataset, "split", None)
        if (isinstance(fitted, str) and fitted != split and fitted in splits
                and not np.array_equal(np.sort(np.asarray(splits[fitted].get("train", []))),
                                       np.sort(np.asarray(parts.get("train", []))))):
            raise ValueError(
                f"The dataset fitted its normalisation and class weights on split "
                f"{fitted!r}, but split {split!r} was requested: construct the dataset "
                f"with split={split!r} so that the statistics come from its training part.")
        train_ds, val_ds, test_ds = (
            Subset(dataset, np.asarray(parts.get(k, []), dtype=int).tolist())
            for k in ("train", "val", "test")
        )
        if verbose:
            print(f"\nSplit '{split}': train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")
    elif getattr(dataset, "groups", None) is not None or getattr(dataset, "splits", None):
        warnings.warn(
            "Sample-level split on a dataset that provides groups or named splits: "
            "samples of one group may fall in train and test. Pass split='official' "
            "or another name from dataset.splits.",
            UserWarning, stacklevel=2,
        )

    if split is None and stratified:
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
            print("\nStratified split - Original class distribution:")
            unique, counts = np.unique(all_labels, return_counts=True)
            for cls, count in zip(unique, counts, strict=False):
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
    elif split is None:
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
