"""Chikusei on a tiny synthetic v2 file (same HDF5 layout as prepared_v2)."""

import h5py
import numpy as np
import pytest

from spdnet_datasets.core import InsufficientSamplesError, assert_no_leakage
from spdnet_datasets.hyperspectral.chikusei import (
    PREPARED,
    ChikuseiDataset,
    fetch_chikusei,
    read_scene_crop,
)


@pytest.fixture
def chikusei_dir(tmp_path):
    root = tmp_path / "Chikusei"
    (root / PREPARED).parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = 30
    with h5py.File(root / PREPARED, "w") as f:
        f["scene/class_names"] = np.array([b"Water", b"Forest"])
        f["w15/windows"] = rng.integers(1, 4000, size=(n, 15, 15, 128)).astype(np.uint16)
        f["w15/centers"] = np.stack([np.arange(n) * 4 + 7, np.full(n, 7)], 1).astype(np.int32)
        f["w15/labels"] = (np.arange(n) % 2 + 1).astype(np.uint8)
        f["w15/split"] = np.array([0] * 14 + [-1] * 2 + [1] * 6 + [2] * 8, np.int8)
        f["w15/block"] = np.array([0] * 16 + [1] * 6 + [2] * 8, np.int32)
        f["w15/purity"] = np.ones(n, np.float32)
        f["w15/n_valid"] = np.full(n, 225, np.int16)
    return root


def test_fetch_drops_buffer_and_keeps_split(chikusei_dir, tmp_path):
    b = fetch_chikusei(chikusei_dir, cache_dir=tmp_path / "c")
    assert b.covariances.shape == (28, 64, 64)
    assert b.info["excluded_buffer"] == 2
    sp = b.splits["recommended"]
    assert [len(sp[k]) for k in ("train", "val", "test")] == [14, 6, 8]
    assert_no_leakage(b.groups[sp["train"]], b.groups[sp["test"]])
    assert set(b.labels) == {"Water", "Forest"}


def test_band_step_one_violates_n_ge_2p(chikusei_dir, tmp_path):
    with pytest.raises(InsufficientSamplesError):
        fetch_chikusei(chikusei_dir, cache_dir=tmp_path / "c", band_step=1)


def test_raw_is_the_band_subsampled_window(chikusei_dir, tmp_path):
    ds = ChikuseiDataset(chikusei_dir, cache_dir=tmp_path / "c", verbose=False, return_raw=True)
    (cov, raw), y = ds[0]
    with h5py.File(chikusei_dir / PREPARED) as f:
        ref = f["w15/windows"][0][..., ::2]
    assert raw.shape == (15, 15, 64)
    np.testing.assert_allclose(raw.numpy(), np.sqrt(ds.scale) * ref, rtol=1e-6)
    X = ref.reshape(-1, 64).astype(float)
    np.testing.assert_allclose(cov.numpy() / ds.scale, np.cov(X.T, bias=True), rtol=1e-10)


def test_scene_crop_undoes_the_matlab_transposition(tmp_path):
    scene = np.arange(5 * 7 * 3, dtype=float).reshape(5, 7, 3)  # (H, W, B)
    path = tmp_path / "scene.mat"
    with h5py.File(path, "w") as f:
        f["chikusei"] = np.transpose(scene, (2, 1, 0))  # what h5py reads: (B, W, H)
    np.testing.assert_array_equal(read_scene_crop(path, 1, 4, 2, 6), scene[1:4, 2:6])
