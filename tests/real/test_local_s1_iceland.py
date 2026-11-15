"""S1 Iceland SAR on the real local copy: counts against the atlas fiche ``iceland_sar``."""

import numpy as np
import pytest

from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data

# Atlas fiche iceland_sar, class_distribution["souspatchs32_train"] (32 px
# sub-patches, same selection rules): matched exactly.
# Test: the fiche (6728) checks pixel validity over the 16 dates of 2025; the
# loader checks it over the 15 kept dates (20250914 dropped), which keeps 63
# more sub-patches (6791, measured 2026-11-15). Val is not counted in the fiche.
EXPECTED = {
    "train": {"other_vegetation": 1721, "sealed": 1190, "periodic_herbaceous": 23080,
              "sparse_vegetated": 3820, "water": 6322, "snow_ice": 2055, "open_spaces": 1602},
    "val": {"other_vegetation": 233, "sealed": 105, "periodic_herbaceous": 3991,
            "sparse_vegetated": 596, "water": 3823, "snow_ice": 173, "open_spaces": 2445},
    "test": {"other_vegetation": 421, "sealed": 50, "periodic_herbaceous": 4237,
             "sparse_vegetated": 1181, "water": 552, "snow_ice": 350, "open_spaces": 0},
}
FICHE_TEST_16_DATES = {"other_vegetation": 418, "sealed": 50, "periodic_herbaceous": 4197,
                       "sparse_vegetated": 1183, "water": 548, "snow_ice": 332,
                       "open_spaces": 0}


@pytest.fixture(scope="module")
def bunch():
    root = dataset_dir("SAR", "ICELAND_SAR")
    skip_if_missing(root)
    from spdnet_datasets.sar.s1_iceland import fetch_s1_iceland

    return fetch_s1_iceland(root)


def test_counts_match_the_fiche(bunch):
    from spdnet_datasets.sar.s1_iceland import CLASS_NAMES, SPLITS

    cdir = bunch.covariance_path
    m, ok = np.load(cdir / "meta.npz"), np.load(cdir / "spd_ok.npy")
    assert sum(FICHE_TEST_16_DATES.values()) == 6728
    for sp, expected in EXPECTED.items():
        sel = m["split"] == sp  # before the exclusion of non-SPD covariances
        counts = np.bincount(m["label"][sel], minlength=7)
        assert dict(zip(CLASS_NAMES, counts.tolist(), strict=True)) == expected
    for sp in SPLITS:
        n_all = int((m["split"] == sp).sum())
        assert len(bunch.splits["official"][sp]) == n_all - bunch.n_excluded[sp]
        assert bunch.n_excluded[sp] == int((~ok[m["split"] == sp]).sum())


def test_dates_and_shapes(bunch):
    assert bunch.dropped_dates == {"train": [], "val": [], "test": ["20250914"]}
    assert all(len(d) == 15 for d in bunch.dates.values())
    assert bunch.covariances.shape[1:] == (30, 30)
    eig = np.linalg.eigvalsh(np.asarray(bunch.covariances[:500]))
    assert eig.min() > 0


def test_official_split_without_leakage(bunch):
    from spdnet_datasets.core.splits import assert_no_leakage, assert_no_window_overlap

    parts, meta = bunch.splits["official"], bunch.meta
    boxes = np.stack([meta.row, meta.col, meta.row + 32, meta.col + 32], 1)
    years = {sp: set(meta.year[idx]) for sp, idx in parts.items()}
    assert years == {"train": {2019}, "val": {2019}, "test": {2025}}
    for a, c in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert_no_leakage(bunch.groups[parts[a]], bunch.groups[parts[c]])
    # window overlap on the val/test sides (the checker loops over its first argument)
    assert_no_window_overlap(boxes[parts["val"]], boxes[parts["train"]])
    assert_no_window_overlap(boxes[parts["test"]], boxes[parts["train"]])
    assert_no_window_overlap(boxes[parts["test"]], boxes[parts["val"]])


def test_dataset_outputs_and_loaders(bunch):
    from spdnet_datasets.base import create_dataloaders
    from spdnet_datasets.sar.s1_iceland import S1IcelandDataset

    ds = S1IcelandDataset(dataset_dir("SAR", "ICELAND_SAR"), return_mean=True,
                          return_raw=True, verbose=False)
    (cov, mean, raw), y = ds[len(ds) - 1]
    assert cov.shape == (30, 30) and mean.shape == (30,) and tuple(raw.shape) == (32, 32, 30)
    tr, va, te = create_dataloaders(ds, batch_size=64, num_workers=0, split="official",
                                    verbose=False)
    assert (len(tr.dataset), len(va.dataset), len(te.dataset)) == (39790, 11366, 6791)
    eig = np.linalg.eigvalsh(ds.covariances[ds.splits["official"]["train"]])
    assert abs(np.log(eig).mean()) < 1e-8
