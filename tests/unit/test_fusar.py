"""FUSAR-Ship loader: class filtering, capping, image format, split."""

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from spdnet_datasets import DatasetManager
from spdnet_datasets.real.fusar import FUSARShipDataset

# category -> {subcategory: number of chips}
LAYOUT = {
    "Cargo": {"BulkCarrier": 25, "Container": 20},  # 45
    "Fishing": {"Fishing": 35},  # 35
    "Tanker": {"OilTanker": 12},  # 12 -> dropped with min 30
}


@pytest.fixture
def fusar_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for category, subs in LAYOUT.items():
        for sub, n in subs.items():
            folder = tmp_path / category / sub
            folder.mkdir(parents=True)
            for i in range(n):
                chip = rng.integers(0, 256, size=(48, 48), dtype=np.uint8)
                tifffile.imwrite(folder / f"Ship_{category}_{sub}_{i:04d}.tiff", chip)
    return tmp_path


def _make(data_dir, **kwargs):
    return FUSARShipDataset(data_dir=str(data_dir), verbose=False, **kwargs)


def test_minority_classes_are_dropped_and_labels_contiguous(fusar_dir):
    ds = _make(fusar_dir, min_samples_per_class=30, image_size=32)
    assert ds.classes == ["Cargo", "Fishing"]
    assert sorted({label for _, label in ds.samples}) == [0, 1]
    assert len(ds) == 45 + 35


def test_cap_after_dropping(fusar_dir):
    ds = _make(fusar_dir, min_samples_per_class=30, max_samples_per_class=30)
    counts = np.bincount([label for _, label in ds.samples])
    assert counts.tolist() == [30, 30]  # Cargo capped, Fishing capped


def test_fine_level_and_exclusion(fusar_dir):
    ds = _make(fusar_dir, class_level="fine", min_samples_per_class=20)
    assert ds.classes == ["Cargo_BulkCarrier", "Cargo_Container", "Fishing_Fishing"]
    ds = _make(fusar_dir, min_samples_per_class=10, exclude_classes=["Fishing"])
    assert ds.classes == ["Cargo", "Tanker"]


def test_image_format(fusar_dir):
    ds = _make(fusar_dir, image_size=32)
    x, label = ds[0]
    assert x.shape == (1, 32, 32) and x.dtype == torch.float32
    assert abs(x.mean().item()) < 1e-4 and abs(x.std().item() - 1) < 1e-3
    assert isinstance(label, int)


def test_through_the_manager_with_stratified_split(fusar_dir):
    train, val, test, n_classes = DatasetManager.create_dataloaders(
        {
            "name": "fusar",
            "path": str(fusar_dir),
            "image_size": 32,
            "batch_size": 8,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "stratified": True,
            "num_workers": 0,
            "persistent_workers": False,
            "verbose": False,
        }
    )
    assert n_classes == 2
    x, y = next(iter(train))
    assert x.shape[1:] == (1, 32, 32)
    for loader in (val, test):  # both classes in every split
        labels = torch.cat([y for _, y in loader])
        assert set(labels.tolist()) == {0, 1}


FUSAR_ROOT = (
    Path(
        os.environ.get(
            "SPDNET_DATA_ROOT", "/home/mgallet/Documents/Dataset/BATCHNORM_dataset"
        )
    )
    / "FUSAR_Ship1.0"
)


@pytest.mark.skipif(not FUSAR_ROOT.exists(), reason="FUSAR-Ship data not available")
def test_real_data_classes():
    ds = FUSARShipDataset(
        data_dir=str(FUSAR_ROOT),
        min_samples_per_class=30,
        max_samples_per_class=350,
        verbose=False,
    )
    assert ds.classes == sorted(
        [
            "Cargo",
            "Dredger",
            "Fishing",
            "LawEnforce",
            "Other",
            "Passenger",
            "Reserved",
            "Tanker",
            "Tug",
            "Unspecified",
        ]
    )
    assert len(ds) == 3 * 350 + 248 + 142 + 64 + 64 + 61 + 37 + 35
    x, _ = ds[0]
    assert x.shape == (1, 224, 224)
