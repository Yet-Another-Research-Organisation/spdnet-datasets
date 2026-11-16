"""HyperLeaf2024 on the real data (skipped when absent)."""

import collections

import pytest

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.hyperspectral.hyperleaf import fetch_hyperleaf
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT = dataset_dir("HYPERSPECTRAL", "HyperLeaf2024")


def test_counts_vs_fiche():
    """Fiche: 1590 labelled leaves, 24 plots (GrainWeight | cultivar | Fertilizer)."""
    skip_if_missing(ROOT)
    b = fetch_hyperleaf()
    assert b.covariances.shape == (1590, 204, 204)
    assert dict(collections.Counter(b.meta.cultivar)) == {
        "Heerup": 410, "Kvium": 350, "Rembrandt": 430, "Sheriff": 400}
    assert len(set(b.groups)) == 24 and b.n_obs.min() >= 408
    sp = b.splits["recommended"]
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_leakage(b.groups[sp[a]], b.groups[sp[c]])
