"""HyperLeaf2024 covariance mode, labels and regression targets on synthetic leaves."""

import numpy as np
import pandas as pd
import pytest
import torch

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.core.estimators import estimate_covariance
from spdnet_datasets.hyperspectral.hyperleaf import (
    CULTIVARS,
    HyperLeafDataset,
    fetch_hyperleaf,
)


@pytest.fixture
def leaf_dir(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    root = tmp_path / "HyperLeaf2024"
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        cube = rng.integers(100, 4000, size=(204, 20, 30)).astype(np.uint16)
        cube[:, :, :10] = 0  # background
        tifffile.imwrite(root / "images" / f"{i:05d}.tiff", cube)
        cult = [int(c == i % 4) for c in range(4)]
        rows.append([f"{i:05d}", 1000 + i // 2, 0.05 + i, 0.6, (0.0, 1.0)[i % 2], *cult])
    pd.DataFrame(rows, columns=["ImageId", "GrainWeight", "Gsw", "PhiPS2", "Fertilizer",
                                *CULTIVARS]).to_csv(root / "train.csv", index=False)
    return root


def test_leaf_pixels_and_plot_groups(leaf_dir, tmp_path):
    b = fetch_hyperleaf(leaf_dir, cache_dir=tmp_path / "c", band_step=4)
    assert b.covariances.shape == (24, 51, 51)
    assert (b.n_obs == 400).all()  # background excluded
    sp = b.splits["recommended"]
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_leakage(b.groups[sp[a]], b.groups[sp[c]])


def test_targets_and_regression(leaf_dir, tmp_path):
    ds = HyperLeafDataset(leaf_dir, cache_dir=tmp_path / "c", band_step=4, verbose=False,
                          target="fertilizer")
    assert ds.classes == ["0.0", "1.0"]
    reg = HyperLeafDataset(leaf_dir, cache_dir=tmp_path / "c", band_step=4, verbose=False,
                           task="regression", regression_targets=("GrainWeight",))
    x, y = reg[0]
    assert x.shape == (51, 51) and y.dtype == torch.float64 and y.shape == (1,)


def test_raw_is_the_leaf_pixels_of_the_covariance(leaf_dir, tmp_path):
    """raw = the (n, p) leaf pixels: its covariance and mean are cov and mean (x s, x sqrt s)."""
    ds = HyperLeafDataset(leaf_dir, cache_dir=tmp_path / "c", band_step=4, verbose=False,
                          return_mean=True, return_raw=True)
    assert ds.scale != 1.0
    for i in (0, 5):
        (cov, mean, raw), _ = ds[i]
        assert raw.dtype == torch.float32 and raw.shape == (400, 51)  # background excluded
        X = raw.numpy().astype(np.float64)
        np.testing.assert_allclose(estimate_covariance(X), cov.numpy(), rtol=1e-4,
                                   atol=1e-5 * float(cov.abs().max()))  # float32 raw
        np.testing.assert_allclose(X.mean(0), mean.numpy(), rtol=1e-5)


def test_full_bands_need_enough_leaf_pixels(leaf_dir, tmp_path):
    """p = 204 needs 408 leaf pixels: the 400-pixel leaves are dropped and counted."""
    with pytest.raises(RuntimeError, match="No sample"):
        fetch_hyperleaf(leaf_dir, cache_dir=tmp_path / "c")
