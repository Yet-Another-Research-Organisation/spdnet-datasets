"""AVIRIS-C CA TS14 FINE / MIX on the real data (skipped when absent)."""

import collections

import numpy as np
import pytest

from spdnet_datasets.core import assert_no_leakage, verify_local
from spdnet_datasets.hyperspectral.aviris import NAMES, fetch_aviris
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data

FICHE_FINE = {"amandiers": 57, "arbustes": 171, "eau": 171, "foret_coniferes": 171,
              "foret_mixte": 171, "prairie_paturage": 171, "sol_nu": 171, "urbain_dense": 111,
              "urbain_faible": 62, "urbain_moyen": 171, "urbain_ouvert": 106, "vignes": 57}


@pytest.mark.parametrize("version", sorted(NAMES))
def test_manifest_is_checked(version):
    root = dataset_dir("HYPERSPECTRAL", NAMES[version])
    skip_if_missing(root)
    report = verify_local(root)
    assert report.n_checked > 1000 and not report.missing and not report.wrong_size


def test_fine_counts_vs_fiche():
    """Fiche: 1590 windows; the 8 windows without any QC-valid date are dropped."""
    skip_if_missing(dataset_dir("HYPERSPECTRAL", NAMES["fine"]))
    b = fetch_aviris(version="fine")
    assert b.covariances.shape == (1582, 83, 83)
    assert b.info["excluded_no_valid_date"] == 8
    counts = collections.Counter(b.labels)
    assert {k: v for k, v in counts.items() if k != "sol_nu"} == \
        {k: v for k, v in FICHE_FINE.items() if k != "sol_nu"}
    assert counts["sol_nu"] == 171 - 8
    assert np.all(b.n_obs >= 2 * 83)
    sp = b.splits["recommended"]
    assert_no_leakage(b.groups[sp["train"]], b.groups[sp["test"]])
    folds = b.meta["fold5"].to_numpy()
    assert set(folds[sp["test"]]) == {0} and set(folds[sp["val"]]) == {1}


def test_mix_regression_targets():
    skip_if_missing(dataset_dir("HYPERSPECTRAL", NAMES["mix"]))
    b = fetch_aviris(version="mix")
    assert b.targets.shape == (len(b.labels), 13)
    np.testing.assert_allclose(b.targets.sum(1), 1, atol=1e-3)
    assert len(b.labels) + b.info["excluded_no_valid_date"] == 2034
    bk = b.splits["block10km"]
    assert_no_leakage(b.groups[bk["train"]], b.groups[bk["val"]])
    assert_no_leakage(b.groups[bk["train"]], b.groups[bk["test"]])
