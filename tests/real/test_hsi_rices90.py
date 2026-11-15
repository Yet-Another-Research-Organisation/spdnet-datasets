"""Rices_90 grains on the real data (skipped when absent)."""

import pytest

from spdnet_datasets.core import assert_no_leakage, verify_local
from spdnet_datasets.hyperspectral.rices90 import fetch_rices90
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT = dataset_dir("HYPERSPECTRAL", "Rices_90")


def test_manifest_is_checked():
    skip_if_missing(ROOT)
    report = verify_local(ROOT)
    assert report.n_checked > 0 and not report.missing and not report.wrong_size


def test_grains_vs_fiche():
    """Fiche: 180 cubes, 8796 grains (48 to 54 per cube); 64 bands give n >= 2p."""
    skip_if_missing(ROOT)
    b = fetch_rices90()
    assert b.info["n_cubes"] == 180 and b.info["excluded_n_lt_2p"] == 0
    assert abs(len(b.labels) - 8796) < 0.01 * 8796
    per_cube = b.meta.groupby("cube").size()
    assert len(per_cube) == 180 and per_cube.min() >= 45
    assert len(set(b.labels)) == 90 and b.covariances.shape[1:] == (64, 64)
    sp = b.splits["recommended"]
    assert_no_leakage(b.groups[sp["train"]], b.groups[sp["test"]])
    assert b.meta.label[b.splits["cross_session"]["test"]].nunique() == 17
