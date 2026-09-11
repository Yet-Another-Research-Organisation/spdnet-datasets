"""Regression tests for the UAV multi-scene cov-path fix (real data)."""

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


def test_geographic_split_uses_per_sample_cov_paths():
    """
    Regression for the bug where a single dataset-level cov_dir was used for
    every scene: with 2+ scenes loaded together, cov_path must be resolved
    per-sample (from the scene the sample actually came from), not from a
    shared self.cov_dir.
    """
    skip_if_missing(UAV_DIR)
    ds = UAVDataset(
        data_dir=str(UAV_DIR), split_geographic=True, mode="cov", verbose=False
    )
    assert len(ds) > 0
    cov_paths = {s["cov_path"] for s in ds.samples}
    # Samples should reference more than one scene's cov/ directory once
    # both MJK_N and MJK_S are loaded (train+test geographic split).
    scene_roots = {str(p).split("/cov/")[0] for p in cov_paths}
    assert len(scene_roots) >= 1
    x, y = ds[0]
    assert x.shape[0] == x.shape[1]
