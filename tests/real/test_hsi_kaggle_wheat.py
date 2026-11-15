"""Wheat disease (Kaggle) on the real data (skipped when absent)."""

import collections

import numpy as np
import pytest

from spdnet_datasets.core import check_spd, verify_local
from spdnet_datasets.hyperspectral.kaggle_wheat import fetch_kaggle_wheat
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT = dataset_dir("HYPERSPECTRAL", "Wheat_Disease_ICPR26")


def test_manifest_is_checked():
    skip_if_missing(ROOT)
    report = verify_local(ROOT)
    assert report.n_checked > 1000 and not report.missing and not report.wrong_size


def test_counts_vs_fiche():
    """Fiche: 600 patches, 23 entirely zero (Health 9, Other 14) -> 577 valid."""
    skip_if_missing(ROOT)
    b = fetch_kaggle_wheat()
    assert b.info["n_patches"] == 600 and b.info["excluded_all_zero"] == 23
    assert len(b.labels) + b.info["excluded_n_lt_2p"] == 577
    counts = collections.Counter(b.labels)
    assert counts["Rust"] == 200 and counts["Health"] <= 191 and counts["Other"] <= 186
    check_spd(np.asarray(b.covariances))
    val = fetch_kaggle_wheat(part="val")
    assert val.info["n_patches"] == 300 and val.info["excluded_all_zero"] == 11
