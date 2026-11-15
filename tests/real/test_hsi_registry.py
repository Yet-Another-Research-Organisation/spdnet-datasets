"""Every hyperspectral class built with its defaults through the registry (real data)."""

import numpy as np
import pytest
import torch

from spdnet_datasets.base import create_dataloaders
from spdnet_datasets.core.config import get_data_root
from spdnet_datasets.manager import DatasetManager

pytestmark = pytest.mark.real_data

CASES = {
    "gsoff": ("GSOFF", {}),
    "chikusei": ("Chikusei", {}),
    "uav": ("UAV-HSI-Crop", {}),
    "aviris": ("AVIRIS-C_CA_TS14_FINE", {}),
    "aviris_mix": ("AVIRIS-C_CA_TS14_MIX", {"name": "aviris", "version": "mix",
                                             "task": "regression"}),
    "kaggle_wheat": ("Wheat_Disease_ICPR26", {}),
    "placenta": ("Hyperspectral_Placenta_Dataset", {}),
    "rices90": ("Rices_90", {}),
    "hyperleaf": ("HyperLeaf2024", {}),
    "deephsfruit": ("DeepHS-Fruit", {}),
}


@pytest.mark.parametrize("key", sorted(CASES))
def test_registry_default_construction(key):
    folder, extra = CASES[key]
    if not (get_data_root() / "HYPERSPECTRAL" / folder).exists():
        pytest.skip(f"{folder} not available")
    config = {"name": key, "verbose": False, **extra}
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", UserWarning)
        ds = DatasetManager.create_dataset(config)
    assert len(ds) > 0 and np.isfinite(ds.scale) and ds.scale > 0
    x, y = ds[0]
    assert x.dtype == torch.float64 and x.ndim == 2 and x.shape[0] == x.shape[1]
    tr, _, _ = create_dataloaders(ds, batch_size=4, num_workers=0, split="recommended",
                                  verbose=False)
    xb, yb = next(iter(tr))
    assert xb.shape[1:] == x.shape
