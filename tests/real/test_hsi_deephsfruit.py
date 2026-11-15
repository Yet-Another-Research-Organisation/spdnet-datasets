"""DeepHS-Fruit precomputed covariances on the real data (skipped when absent)."""

import collections

import pytest

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.hyperspectral.deephsfruit import fetch_deephsfruit
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT = dataset_dir("HYPERSPECTRAL", "DeepHS-Fruit")


def test_counts_and_physical_fruit_groups():
    skip_if_missing(ROOT)
    b = fetch_deephsfruit(verify=None)
    assert len(b.labels) == 3193  # fiche: 3 193 cov_VIS files
    sp = b.splits["recommended"]
    assert_no_leakage(b.groups[sp["train"]], b.groups[sp["test"]])
    r = fetch_deephsfruit(verify=None, classification_mode="ripeness")
    # fiche: 384 VIS cubes in train_only_labeled_v2; val/test cubes have no cov file
    assert len(r.labels) == 384 and "official" not in r.splits
    assert set(collections.Counter(r.labels)) == {"unripe", "perfect", "overripe"}
