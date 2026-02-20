"""
PyTorch Dataset and DataLoader utilities for simulation experiments.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional

from .data_generator import shuffle_data


class SimulationDataset(Dataset):
    """PyTorch Dataset wrapping pre-generated SPD matrices and labels."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            data: SPD matrices of shape (n_samples, matrix_size, matrix_size).
            labels: Class labels of shape (n_samples,).
            dtype: Torch dtype for the data tensors.
        """
        self.data = torch.from_numpy(data).to(dtype)
        self.labels = torch.from_numpy(labels).long()
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

    @property
    def num_classes(self) -> int:
        return len(torch.unique(self.labels))

    @property
    def matrix_size(self) -> int:
        return self.data.shape[1]


def create_dataloaders(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 12,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train / validation / test dataloaders from raw data arrays.

    Args:
        data: SPD matrices of shape (n_samples, matrix_size, matrix_size).
        labels: Class labels.
        batch_size: Batch size for all dataloaders.
        val_ratio: Fraction of data used for validation.
        test_ratio: Fraction of data used for testing.
        seed: Random seed for the train/val/test split.
        dtype: Torch dtype.
        num_workers: DataLoader worker count.

    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """
    data, labels = shuffle_data(data, labels, seed=seed)

    n_total = len(data)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val

    train_dataset = SimulationDataset(data[:n_train], labels[:n_train], dtype=dtype)
    val_dataset = SimulationDataset(
        data[n_train : n_train + n_val], labels[n_train : n_train + n_val], dtype=dtype
    )
    test_dataset = SimulationDataset(
        data[n_train + n_val :], labels[n_train + n_val :], dtype=dtype
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, train_dataset.num_classes
