"""GSOFF on a tiny synthetic scene (same file layout as the real dataset)."""

import numpy as np
import pytest
import torch

from spdnet_datasets.core import InsufficientSamplesError, assert_no_window_overlap
from spdnet_datasets.hyperspectral._common import window_boxes
from spdnet_datasets.hyperspectral.gsoff import (
    GROUND_TRUTH,
    SCENE,
    GSOFFDataset,
    fetch_gsoff,
)


@pytest.fixture
def gsoff_dir(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    root = tmp_path / "GSOFF"
    (root / SCENE).parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    cube = rng.random((150, 150, 125), dtype=np.float32) + 0.1
    gt = np.zeros((150, 150), np.uint8)
    gt[:75] = 1
    gt[75:] = 2
    gt[140:, 140:] = 0
    cube[:30, :30] = 0  # no-data corner: windows there have too few valid pixels
    tifffile.imwrite(root / SCENE, cube)
    tifffile.imwrite(root / GROUND_TRUTH, gt)
    return root


def test_fetch_windows_and_band_step(gsoff_dir, tmp_path):
    b = fetch_gsoff(gsoff_dir, cache_dir=tmp_path / "cache", block=30)
    assert b.covariances.shape[1:] == (63, 63)
    assert set(b.labels) == {"code 1", "code 2"}
    assert b.info["excluded_n_lt_2p"] == 4  # the 2 x 2 windows of the zero corner
    assert np.all(b.n_obs >= 2 * 63)
    sp = b.splits["recommended"]
    boxes = window_boxes(b.positions, 15)
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_window_overlap(boxes[sp[a]], boxes[sp[c]])


def test_band_step_one_violates_n_ge_2p(gsoff_dir, tmp_path):
    with pytest.raises(InsufficientSamplesError):
        fetch_gsoff(gsoff_dir, cache_dir=tmp_path / "cache", band_step=1)


def test_dataset_outputs_and_normalisation(gsoff_dir, tmp_path):
    ds = GSOFFDataset(gsoff_dir, cache_dir=tmp_path / "cache", block=30, verbose=False,
                      return_mean=True, return_raw=True)
    (cov, mean, raw), y = ds[0]
    assert cov.dtype == torch.float64 and cov.shape == (63, 63)
    assert mean.shape == (63,) and raw.shape == (15, 15, 63) and raw.dtype == torch.float32
    assert y in (0, 1)
    train = ds.splits["recommended"]["train"]
    eig = np.linalg.eigvalsh(np.stack([ds[i][0][0].numpy() for i in train]))
    assert abs(np.mean(np.log(eig))) < 1e-8
    sub = GSOFFDataset(gsoff_dir, cache_dir=tmp_path / "cache", block=30, verbose=False,
                       subset="test")
    assert len(sub) == len(ds.splits["recommended"]["test"])


def test_dataset_class_selection(gsoff_dir, tmp_path):
    ds = GSOFFDataset(gsoff_dir, cache_dir=tmp_path / "cache", block=30, verbose=False,
                      classes=["code 2"])
    assert ds.classes == ["code 2"] and {ds[i][1] for i in range(len(ds))} == {0}
    assert all(idx.max() < len(ds) for idx in ds.splits["recommended"].values() if len(idx))
