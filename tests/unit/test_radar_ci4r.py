"""CI4R on a tiny synthetic copy (jet-coloured PNG spectrograms under tmp_path)."""

import numpy as np
import pytest
import torch
from PIL import Image

from spdnet_datasets import DatasetManager, create_dataloaders
from spdnet_datasets.core import assert_no_leakage, check_spd
from spdnet_datasets.radar_microdoppler import ci4r
from spdnet_datasets.radar_microdoppler.ci4r import (
    CI4RDataset,
    fetch_ci4r,
    jet_to_scalar,
)

CLASSES = ["Away", "Bend", "Crawl"]
PARTICIPANTS = ["10", "14", "29", "30", "00"]  # "00" is merged with "14"
H, W, P = 80, 60, 8


def _jet_table():
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, *zip(*ci4r._JET[c], strict=True)) for c in "rgb"], 1) * 255


def _to_rgb(values):
    return np.round(_jet_table()[np.round(values * 255).astype(int)]).astype(np.uint8)


def make_ci4r(root, folder="77GHz", constant_rows=False, margin=False):
    rng = np.random.default_rng(0)
    base = root / "11_class_activity_data" / folder
    ts = 1573837000
    for c, name in enumerate(CLASSES):
        (base / name).mkdir(parents=True)
        for k, code in enumerate(PARTICIPANTS):
            for rep in range(2):
                values = rng.uniform(0, 1, (H, W)) * (0.3 + 0.2 * c)
                if constant_rows and k == 0 and rep == 0:
                    values[30:50] = 0.5  # constant Doppler rows inside the band
                rgb = _to_rgb(values)
                if margin:
                    rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), constant_values=128)
                ts += 1
                Image.fromarray(rgb).save(base / name / f"04{c + 5:02d}00{code}_{ts}_Raw_0.png")
    first = sorted((base / CLASSES[0]).glob("*.png"))[0]
    (base / CLASSES[0] / (first.stem + " (copy).png")).write_bytes(first.read_bytes())
    return root


@pytest.fixture
def data(tmp_path):
    return make_ci4r(tmp_path / "CI4R-MULTI3")


def test_jet_inversion_round_trip():
    values = np.random.default_rng(1).integers(0, 256, (20, 30)) / 255
    rgb = np.pad(_to_rgb(values), ((2, 2), (3, 3), (0, 0)), constant_values=128)
    # grey margins cropped; jet entries 28-32 are the same 8-bit colour (blue plateau)
    assert np.allclose(jet_to_scalar(rgb), values, atol=2.5 / 255)


def test_fetch_shapes_duplicates_and_groups(data, tmp_path):
    b = fetch_ci4r(data, tmp_path / "cache", band=0.5, p=P, verify=None)
    n = len(CLASSES) * len(PARTICIPANTS) * 2
    assert b.covariances.shape == (n, P, P)
    assert b.info["duplicates_removed"] == 1 and b.info["n_files"] == n + 1
    check_spd(b.covariances)
    assert (b.meta.n_obs >= 2 * P).all() and (b.meta.n_obs == W).all()
    assert set(b.groups) == {"10", "14", "29", "30"}
    for name in ("fold0", "fold1", "fold2", "fold3", "recommended"):
        s = b.splits[name]
        assert_no_leakage(b.groups[s["train"]], b.groups[s["test"]])
        assert len(s["train"]) + len(s["test"]) == n
    tests = np.concatenate([b.splits[f"fold{k}"]["test"] for k in range(4)])
    assert sorted(tests) == list(range(n))  # leave-one-subject-out covers all samples


def test_band_and_p_define_the_features():
    values = np.arange(100, dtype=float)[:, None] * np.ones((1, 7))
    X = ci4r.spectrogram_features(values, band=0.5, p=5)
    assert X.shape == (7, 5)
    assert np.allclose(X[0], [69.5, 59.5, 49.5, 39.5, 29.5])  # rows flipped, central 50 rows, blocks of 10
    with pytest.raises(ValueError, match="fewer than p"):
        ci4r.spectrogram_features(values, band=0.04, p=5)


def test_dataset_outputs_normalisation_and_registry(data, tmp_path):
    kw = {"band": 0.5, "p": P, "cache_dir": tmp_path / "cache", "verify": None, "verbose": False}
    ds = CI4RDataset(data, **kw)
    assert DatasetManager.DATASETS["ci4r"] is CI4RDataset
    x, y = ds[0]
    assert x.dtype == torch.float64 and x.shape == (P, P) and isinstance(y, int)
    train = ds.splits["recommended"]["train"]
    eig = np.linalg.eigvalsh(np.stack([ds[i][0].numpy() for i in train]))
    assert np.exp(np.mean(np.log(eig))) == pytest.approx(1.0, rel=1e-8)

    (cov, mean, raw), _ = CI4RDataset(data, return_mean=True, return_raw=True, **kw)[0]
    assert mean.shape == (P,) and raw.shape == (W, P) and raw.dtype == torch.float32
    tr, va, te = create_dataloaders(ds, batch_size=4, num_workers=0, split="fold1",
                                    verbose=False)
    assert len(tr.dataset) + len(te.dataset) == len(ds) and len(va.dataset) == 0


def test_transform_is_applied(data, tmp_path):
    kw = {"band": 0.5, "p": P, "cache_dir": tmp_path / "cache", "verify": None, "verbose": False}
    ds = CI4RDataset(data, **kw)
    doubled = CI4RDataset(data, transform=lambda x: 2 * x, **kw)
    torch.testing.assert_close(doubled[0][0], 2 * ds[0][0])


def test_class_selection_and_limits(data, tmp_path):
    kw = {"band": 0.5, "p": P, "cache_dir": tmp_path / "cache", "verify": None, "verbose": False}
    ds = CI4RDataset(data, classes=["Crawl", "Away"], **kw)
    assert ds.classes == ["Away", "Crawl"] and set(ds.labels) == {0, 1}
    assert len(CI4RDataset(data, max_classes=2, **kw).classes) == 2
    limited = CI4RDataset(data, max_samples_per_class=3, **kw)
    assert np.bincount(limited.labels).tolist() == [3, 3, 3]
    with pytest.raises(ValueError):
        CI4RDataset(data, classes=["Away", "zzz"], **kw)
    test_only = CI4RDataset(data, subset="test", **kw)
    full = CI4RDataset(data, **kw)
    assert len(test_only) == len(full.splits["recommended"]["test"]) and test_only.splits == {}


def test_keep_covariance_drops_raw(data, tmp_path):
    kw = {"band": 0.5, "p": P, "cache_dir": tmp_path / "c2", "verify": None, "verbose": False}
    b = fetch_ci4r(data, kw["cache_dir"], keep="keep_covariance", band=0.5, p=P, verify=None)
    assert b.processed_path is None
    with pytest.raises(ValueError, match="return_raw"):
        CI4RDataset(data, return_raw=True, keep="keep_covariance", **kw)


def test_xethru_constant_bins(tmp_path):
    root = make_ci4r(tmp_path / "CI4R-MULTI3", "Xethru", constant_rows=True, margin=True)
    lw = fetch_ci4r(root, tmp_path / "cache", mode="xethru", band=0.5, p=P, verify=None)
    assert lw.info["estimator"] == "ledoit_wolf" and lw.info["rejected_constant_channel"] == 0
    scm = fetch_ci4r(root, tmp_path / "cache", "keep_full", "scm", mode="xethru", band=0.5,
                     p=P, verify=None)
    assert scm.info["rejected_constant_channel"] == len(CLASSES)
    assert len(scm.labels) == len(lw.labels) - len(CLASSES)


def test_insufficient_time_steps_raise(tmp_path):
    root = make_ci4r(tmp_path / "CI4R-MULTI3")
    with pytest.raises(ValueError, match="observations"):
        fetch_ci4r(root, tmp_path / "cache", band=1.0, p=40, verify=None)  # T=60 < 80


def test_deprecated_alias():
    with pytest.warns(DeprecationWarning):
        import importlib

        import spdnet_datasets.real.ci4r as legacy

        importlib.reload(legacy)
    assert legacy.CI4RDataset is CI4RDataset
