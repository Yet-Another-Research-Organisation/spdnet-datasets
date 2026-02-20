"""
Dataset Manager - Central router for all datasets.
Automatically selects and instantiates the correct dataset based on configuration.
"""

from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader

from spdnet_datasets.base import create_dataloaders


class DatasetManager:
    """
    Manages dataset creation and dataloader instantiation.
    Routes to the appropriate dataset class based on dataset name.
    """

    # Registry of available datasets
    DATASETS = {}

    @classmethod
    def register_dataset(cls, name: str):
        """Decorator to register a dataset class."""
        def decorator(dataset_class):
            cls.DATASETS[name] = dataset_class
            return dataset_class
        return decorator

    @classmethod
    def create_dataset(cls, config: Dict[str, Any]):
        """
        Create a dataset from a configuration dictionary.

        Args:
            config: Configuration dictionary with at least:
                - name: Dataset name (e.g., 'rices90', 'hyperleaf')
                - path: Path to dataset directory
                And optionally:
                - max_classes: Maximum number of classes
                - max_samples_per_class: Maximum samples per class
                - seed: Random seed
                - Any other dataset-specific parameters

        Returns:
            Dataset instance

        Example config:
            {
                'name': 'rices90',
                'path': '/path/to/data',
                'max_classes': 10,
                'max_samples_per_class': 1000,
                'seed': 42
            }
        """
        dataset_name = config.get('name')
        if dataset_name not in cls.DATASETS:
            available = ', '.join(cls.DATASETS.keys())
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available datasets: {available}"
            )

        # Get dataset class
        dataset_class = cls.DATASETS[dataset_name]

        # Prepare kwargs - rename 'path' to 'data_dir' if needed
        kwargs = config.copy()
        kwargs.pop('name')  # Remove name as it's not a dataset parameter

        if 'path' in kwargs:
            path = kwargs.pop('path')
            # Handle subdir if present
            subdir = kwargs.pop('subdir', None)
            if subdir:
                from pathlib import Path
                path = str(Path(path) / subdir)
            kwargs['data_dir'] = path

        # Remove dataloader-specific params that shouldn't go to dataset
        dataloader_params = [
            'batch_size', 'num_workers', 'pin_memory',
            'persistent_workers', 'shuffle', 'val_ratio',
            'test_ratio', 'num_classes', 'task_type'
        ]
        for param in dataloader_params:
            kwargs.pop(param, None)

        # Add verbose if not specified
        if 'verbose' not in kwargs:
            kwargs['verbose'] = True

        # Create dataset
        print(f"\n{'='*80}")
        print(f"Creating dataset: {dataset_name}")
        print(f"Data directory: {kwargs.get('data_dir')}")
        print(f"{'='*80}")

        dataset = dataset_class(**kwargs)

        return dataset

    @classmethod
    def create_dataloaders(
        cls,
        config: Dict[str, Any]
    ) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """
        Create dataset and dataloaders from configuration.

        Args:
            config: Complete configuration dictionary

        Returns:
            Tuple of (train_loader, val_loader, test_loader, num_classes)
        """
        # Create dataset
        dataset = cls.create_dataset(config)

        # Extract dataloader parameters
        batch_size = config.get('batch_size', 32)
        val_ratio = config.get('val_ratio', 0.1)
        test_ratio = config.get('test_ratio', 0.2)
        num_workers = config.get('num_workers', 4)
        pin_memory = config.get('pin_memory', True)
        persistent_workers = config.get('persistent_workers', True)
        shuffle = config.get('shuffle', True)
        seed = config.get('seed', 42)
        verbose = config.get('verbose', True)
        stratified = config.get('stratified', False)

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            shuffle=shuffle,
            seed=seed,
            verbose=verbose,
            stratified=stratified
        )

        num_classes = dataset.get_num_classes()

        if verbose:
            print("\nDataloaders created successfully")
            print(f"Number of classes: {num_classes}")
            print(f"{'='*80}\n")

        return train_loader, val_loader, test_loader, num_classes

    @classmethod
    def get_available_datasets(cls) -> list:
        """Return list of available dataset names."""
        return list(cls.DATASETS.keys())
