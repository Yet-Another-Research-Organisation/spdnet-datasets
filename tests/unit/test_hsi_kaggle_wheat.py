"""Wheat disease (Kaggle) on synthetic HS patches."""

import numpy as np
import pytest

from spdnet_datasets.core import check_spd
from spdnet_datasets.hyperspectral.kaggle_wheat import (
    KaggleWheatDataset,
    fetch_kaggle_wheat,
)


@pytest.fixture
def wheat_dir(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    root = tmp_path / "Wheat_Disease_ICPR26"
    rng = np.random.default_rng(0)
    for part in ("train", "val"):
        (root / part / "HS").mkdir(parents=True)
    for cls in ("Health", "Rust", "Other"):
        for k in range(10):
            bands = 126 if k % 3 == 0 else 125
            hs = rng.integers(100, 4000, size=(32, 32, bands)).astype(np.uint16)
            if cls == "Health" and k == 0:
                hs[:] = 0  # entirely zero patch
            if cls == "Other" and k == 1:
                hs[:, 5:] = 0  # 160 non-zero pixels < 2p = 214
            tifffile.imwrite(root / "train" / "HS" / f"{cls}_hyper_{k}.tif", hs)
    for k in range(3):
        hs = rng.integers(100, 4000, size=(32, 32, 125)).astype(np.uint16)
        tifffile.imwrite(root / "val" / "HS" / f"val_{k:08x}.tif", hs)
    return root


def test_zero_patches_are_excluded(wheat_dir, tmp_path):
    b = fetch_kaggle_wheat(wheat_dir, cache_dir=tmp_path / "c")
    assert b.covariances.shape == (28, 107, 107)
    assert b.info["excluded_all_zero"] == 1 and b.info["excluded_n_lt_2p"] == 1
    check_spd(np.asarray(b.covariances))  # no zero matrix any more
    sp = b.splits["recommended"]
    assert sorted(np.concatenate([sp["train"], sp["val"], sp["test"]])) == list(range(28))


def test_unlabelled_val_part(wheat_dir, tmp_path):
    b = fetch_kaggle_wheat(wheat_dir, cache_dir=tmp_path / "c", part="val")
    assert len(b.labels) == 3 and b.splits == {}


def test_dataset_raw_and_centred_scm(wheat_dir, tmp_path):
    ds = KaggleWheatDataset(wheat_dir, cache_dir=tmp_path / "c", verbose=False,
                            return_raw=True, return_mean=True, normalize=False)
    (cov, mean, raw), y = ds[0]
    X = raw.numpy().reshape(-1, 107).astype(float)
    np.testing.assert_allclose(mean.numpy(), X.mean(0), rtol=1e-6)
    np.testing.assert_allclose(cov.numpy(), np.cov(X.T, bias=True), rtol=1e-6)
    assert ds.classes == ["Health", "Other", "Rust"]
