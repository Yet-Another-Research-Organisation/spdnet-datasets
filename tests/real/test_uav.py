"""Regression tests for the UAV multi-scene cov-path fix."""

from pathlib import Path

import numpy as np

from spdnet_datasets.real.uav import UAVDataset
from tests.conftest import dataset_dir, skip_if_missing

UAV_DIR = dataset_dir("Hyperspectral", "UAV-HSI-Crop")


def test_resolve_scene_dir_matches_parameter_suffixed_dirs():
    """
    Real scene dirs are suffixed with preprocessing params (e.g.
    MJK_N_W15_T80), not the bare scene name (MJK_N) UAVDataset.TRAIN_SCENES
    uses -- this is exactly what _resolve_scene_dir's prefix match handles.
    """
    skip_if_missing(UAV_DIR)
    ds = UAVDataset(data_dir=str(UAV_DIR), split_geographic=True, mode="cov", verbose=False)
    resolved = ds._resolve_scene_dir("MJK_N")
    assert resolved is not None
    assert resolved.name.startswith("MJK_N")
    assert resolved.exists()


def _assert_per_scene_cov_paths(ds) -> None:
    """Each scene is a contiguous block of samples whose cov files are numbered
    0..n-1 under that scene's own cov/ directory."""
    by_index = sorted(ds.samples, key=lambda s: s["index"])
    roots = [str(Path(s["cov_path"]).parent) for s in by_index]
    blocks = [roots[0]]
    for root in roots[1:]:
        if root != blocks[-1]:
            assert root not in blocks, "scene samples are not contiguous"
            blocks.append(root)
    assert len(blocks) >= 2, "expected samples from at least two scenes"
    for root in blocks:
        numbers = sorted(
            int(Path(s["cov_path"]).stem)
            for s in by_index
            if str(Path(s["cov_path"]).parent) == root
        )
        assert numbers == list(range(len(numbers))), root


def test_multi_scene_uses_per_sample_cov_paths():
    """
    Regression for the bug where a single dataset-level cov_dir was used for
    every scene: with 2+ scenes loaded together, cov_path must be resolved
    per sample, from the scene the sample actually came from. The standard
    split loads every scene (the geographic split loads one scene per split).
    """
    skip_if_missing(UAV_DIR)
    ds = UAVDataset(
        data_dir=str(UAV_DIR), split_geographic=False, mode="raw", verbose=False
    )
    _assert_per_scene_cov_paths(ds)


def test_multi_scene_cov_paths_synthetic(tmp_path):
    """Same regression on a synthetic two-scene tree (no /DATA needed)."""
    for scene, n in (("MJK_N_W15_T80", 3), ("MJK_S", 4)):
        scene_dir = tmp_path / scene
        scene_dir.mkdir()
        np.save(scene_dir / "uav_windows_data.npy", np.zeros((n, 2, 2, 5)))
        np.save(scene_dir / "uav_windows_labels.npy", np.arange(n) % 2)
    ds = UAVDataset(
        data_dir=str(tmp_path), split_geographic=False, mode="raw", verbose=False
    )
    assert len(ds) == 7
    _assert_per_scene_cov_paths(ds)
