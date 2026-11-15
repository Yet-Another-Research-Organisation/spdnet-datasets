"""Placenta windows on the real data (skipped when absent)."""

import collections

import pytest

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.hyperspectral.placenta import fetch_placenta
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT = dataset_dir("HYPERSPECTRAL", "Hyperspectral_Placenta_Dataset")


def test_counts_vs_fiche():
    skip_if_missing(ROOT)
    b = fetch_placenta()
    # 77203 windows (fiche); the singular SCM (0.45 % in the fiche) are dropped and counted
    assert len(b.labels) + b.info["excluded_not_spd"] == 77203
    assert b.info["excluded_not_spd"] < 0.005 * 77203
    fiche = {"Artery": 10162, "Stroma": 56272, "Suture": 114, "Umbilical": 3345, "Vein": 7310}
    counts = collections.Counter(b.labels)
    assert set(counts) == set(fiche) and all(counts[c] <= fiche[c] for c in fiche)
    assert len(set(b.groups)) == 81
    sp = b.splits["recommended"]
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_leakage(b.groups[sp[a]], b.groups[sp[c]])
    dye = b.splits["dye_transfer"]
    assert len(dye["train"]) + len(dye["test"]) == len(b.labels)
    assert 46406 - len(dye["train"]) + 30797 - len(dye["test"]) == b.info["excluded_not_spd"]
