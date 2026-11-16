"""PASTIS-R loader on the real dataset (skipped when absent)."""

import json

import numpy as np
import pandas as pd
import pytest

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.sits.pastis_r import (
    CLASSES,
    SCENARIOS,
    PastisRDataset,
    _read_metadata,
    _resolve,
    fetch_pastis_r,
)
from tests.conftest import DATA_ROOT, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT, FILES = _resolve(None)
CATALOG = DATA_ROOT.parent / "_atlas_v2" / "site" / "catalog.json"


@pytest.fixture(scope="module")
def few_patches():
    skip_if_missing(FILES / "metadata.geojson")
    props = _read_metadata(FILES)
    return [int(p["ID_PATCH"]) for p in props[:: len(props) // 12]][:12]


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_scenarios_on_a_few_patches(tmp_path, few_patches, scenario):
    b = fetch_pastis_r(scenario=scenario, patches=few_patches, verify=None,
                       cache_dir=tmp_path)
    p = SCENARIOS[scenario]["p"]
    assert b.covariances.shape[1:] == (p, p) and len(b.covariances) > 50
    assert np.all(np.linalg.eigvalsh(b.covariances)[:, 0] > 0)
    assert set(b.groups) <= set(few_patches)
    assert (b.meta.groupby("patch")["s2_dates"].nunique() == 1).all()
    assert (b.meta["n_pixels"] >= 2 * p).all()
    assert b.labels.min() >= 0 and b.labels.max() < 18
    if SCENARIOS[scenario]["s1"]:
        assert (b.meta["max_abs_gap"] <= 3).all()
    parts = b.splits["official"]
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_leakage(b.groups[parts[a]], b.groups[parts[c]])


def test_dataset_outputs_on_real_patches(tmp_path, few_patches):
    ds = PastisRDataset(scenario="C", patches=few_patches, verify=None, cache_dir=tmp_path,
                        return_mean=True, return_raw=True, verbose=False)
    (cov, mean, raw), y = ds[0]
    assert cov.shape == (16, 16) and mean.shape == (16,) and raw.shape[1] == 16
    assert raw.shape[0] == ds.meta.n_pixels[0] and 0 <= y < len(ds.classes)


@pytest.mark.slow
def test_parcel_counts_match_the_atlas(tmp_path):
    """All parcels >= 10 px with a crop majority label: 85 951, as in the atlas."""
    skip_if_missing(FILES / "metadata.geojson")
    skip_if_missing(CATALOG)
    entry = next(d for d in json.loads(CATALOG.read_text())["datasets"]
                 if d["id"] == "pastis_r")
    b = fetch_pastis_r(scenario="A", min_pixels=10, max_cloud_fraction=None,
                       cache_dir=tmp_path)
    meta = pd.read_csv(b.processed_path / "parcels.csv")
    assert len(meta) == 85951
    assert b.report["n_patches_without_metadata"] == 35
    for fold in range(1, 6):
        atlas = list(entry["class_distribution"][f"pli {fold}"].values())
        counts = meta[meta.fold == fold].label.value_counts()
        assert [int(counts.get(k, 0)) for k in range(len(CLASSES))] == atlas
    # parcels with 10-19 px cannot give a 10 x 10 covariance (n < 2p): counted, not fatal
    assert b.report["covariance"]["failures"]["InsufficientSamplesError"] == \
        int((meta.n_pixels < 20).sum())
