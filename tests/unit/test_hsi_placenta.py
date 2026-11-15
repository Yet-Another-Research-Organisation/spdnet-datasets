"""Placenta windows on a synthetic imagettes/ folder."""

import json

import numpy as np
import pytest

from spdnet_datasets.core import InsufficientSamplesError, assert_no_leakage
from spdnet_datasets.hyperspectral.placenta import (
    METADATA,
    NAME,
    WINDOWS,
    PlacentaDataset,
    fetch_placenta,
)


def _make(root, size=24, n=40):
    (root / "imagettes").mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(root / WINDOWS, rng.random((n, size, size, 37), dtype=np.float32))
    records = [{"sample_id": f"P{i % 10:03d}", "dye_type": "ICG" if i % 10 >= 5 else "red_blue",
                "class": ("Artery", "Stroma")[i % 2], "original_class": "x",
                "position": [i, 0]} for i in range(n)]
    (root / METADATA).write_text(json.dumps(records))
    return root


@pytest.fixture
def placenta_dir(tmp_path):
    return _make(tmp_path / "Placenta")


def test_split_grouped_by_image(placenta_dir, tmp_path):
    b = fetch_placenta(placenta_dir, cache_dir=tmp_path / "c")
    assert b.covariances.shape == (40, 37, 37)
    sp = b.splits["recommended"]
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_leakage(b.groups[sp[a]], b.groups[sp[c]])
    dye = b.splits["dye_transfer"]
    assert set(b.meta.dye_type[dye["test"]]) == {"ICG"}


def test_cache_key_has_source_and_estimator_only_in_covariances(placenta_dir, tmp_path):
    cache = tmp_path / "c"
    a = fetch_placenta(placenta_dir, cache_dir=cache)
    b = fetch_placenta(_make(tmp_path / "copy"), cache_dir=cache)
    assert a.covariance_path != b.covariance_path
    fetch_placenta(placenta_dir, cache_dir=cache, estimator="ledoit_wolf")
    assert len(list((cache / NAME / "processed").iterdir())) == 2  # one per source
    assert len(list((cache / NAME / "covariance").iterdir())) == 3


def test_raw_mode_returns_numpy_window(placenta_dir, tmp_path):
    """Regression: the former raw mode passed a torch tensor to a numpy estimator."""
    ds = PlacentaDataset(placenta_dir, cache_dir=tmp_path / "c", verbose=False,
                         return_raw=True, normalize=False)
    (cov, raw), _ = ds[3]
    assert raw.shape == (24, 24, 37)
    X = raw.numpy().reshape(-1, 37).astype(float)
    np.testing.assert_allclose(cov.numpy(), np.cov(X.T, bias=True), rtol=1e-6)


def test_small_windows_violate_n_ge_2p(tmp_path):
    root = _make(tmp_path / "Small", size=8)  # 64 pixels < 2 * 37
    with pytest.raises(InsufficientSamplesError):
        fetch_placenta(root, cache_dir=tmp_path / "c")
    assert fetch_placenta(root, cache_dir=tmp_path / "c", band_step=3).covariances.shape[1] == 13
