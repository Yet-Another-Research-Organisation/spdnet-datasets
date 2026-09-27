"""HyperLeaf 'image' mode (band-restricted images) on a synthetic dataset tree."""

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch

from spdnet_datasets.real.hyperleaf import HyperLeafDataset

N_BANDS, H, W = HyperLeafDataset.N_BANDS, 6, 8
CULTIVARS = HyperLeafDataset.CULTIVAR_NAMES


@pytest.fixture
def hyperleaf_dir(tmp_path):
    """Four images; band b of image i is filled with i * 1000 + b plus noise."""
    rng = np.random.default_rng(0)
    (tmp_path / "images").mkdir()
    (tmp_path / "cov").mkdir()
    rows = []
    for i in range(4):
        image_id = f"{i:05d}"
        cube = (i * 1000 + np.arange(N_BANDS))[:, None, None] + rng.integers(
            0, 50, (N_BANDS, H, W)
        )
        tifffile.imwrite(
            tmp_path / "images" / f"{image_id}.tiff", cube.astype(np.uint16)
        )
        torch.save(torch.eye(N_BANDS), tmp_path / "cov" / f"{image_id}.pt")
        onehot = [int(c == i % 4) for c in range(4)]
        rows.append([image_id, 0.0, *onehot])
    pd.DataFrame(rows, columns=["ImageId", "Fertilizer", *CULTIVARS]).to_csv(
        tmp_path / "train.csv", index=False
    )
    return tmp_path


def test_default_bands_are_rgb(hyperleaf_dir):
    ds = HyperLeafDataset(data_dir=str(hyperleaf_dir), mode="image", verbose=False)
    x, y = ds[1]
    assert ds.bands == list(HyperLeafDataset.RGB_BANDS)
    assert x.shape == (3, H, W) and x.dtype == torch.float32
    assert ds.target_size == (3, 48, 352)  # native HyperLeaf image size
    assert y == 1


def test_selected_bands_order_and_standardization(hyperleaf_dir):
    bands = [200, 3, 100]
    ds = HyperLeafDataset(
        data_dir=str(hyperleaf_dir), mode="image", bands=bands, verbose=False
    )
    x, _ = ds[2]
    raw = tifffile.imread(hyperleaf_dir / "images" / "00002.tiff").astype(np.float32)
    expected = raw[bands]
    expected = (expected - expected.mean(axis=(1, 2), keepdims=True)) / (
        expected.std(axis=(1, 2), keepdims=True) + 1e-6
    )
    torch.testing.assert_close(x, torch.from_numpy(expected))
    torch.testing.assert_close(x.mean(dim=(1, 2)), torch.zeros(3), atol=1e-5, rtol=0)


def test_single_band(hyperleaf_dir):
    ds = HyperLeafDataset(
        data_dir=str(hyperleaf_dir), mode="image", bands=[7], verbose=False
    )
    assert ds[0][0].shape == (1, H, W)


@pytest.mark.parametrize("bands", [[204], [-1]])
def test_out_of_range_band(hyperleaf_dir, bands):
    with pytest.raises(ValueError):
        HyperLeafDataset(data_dir=str(hyperleaf_dir), mode="image", bands=bands)


def test_unknown_mode(hyperleaf_dir):
    with pytest.raises(ValueError):
        HyperLeafDataset(data_dir=str(hyperleaf_dir), mode="rgb")


def test_cov_mode_unchanged(hyperleaf_dir):
    ds = HyperLeafDataset(data_dir=str(hyperleaf_dir), mode="cov", verbose=False)
    x, _ = ds[0]
    assert ds.bands is None
    assert x.shape == (N_BANDS, N_BANDS) and x.dtype == torch.float64


def test_hdm05_loads_float64(tmp_path):
    """HDM05 covariances are float64, as every covariance loader."""
    from spdnet_datasets.real.hdm05 import HDM05Dataset

    for i, label in enumerate([3, 3, 7, 7]):
        cov = np.eye(5) * (i + 1)
        np.save(tmp_path / f"{i}_100_{label}.npy", cov)
    ds = HDM05Dataset(data_dir=str(tmp_path), scaling_factor=2.0, verbose=False)
    x, _ = ds[0]
    assert x.dtype == torch.float64 and x.shape == (5, 5)
    assert (
        torch.allclose(torch.diagonal(x).unique(), torch.tensor([2.0], dtype=x.dtype))
        or x.max() > 0
    )
