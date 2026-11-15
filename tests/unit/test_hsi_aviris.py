"""AVIRIS-C CA TS14 FINE / MIX on a tiny synthetic copy (same layout as the dataset)."""

import numpy as np
import pandas as pd
import pytest
import torch

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.hyperspectral.aviris import (
    PROPORTIONS,
    AVIRISDataset,
    fetch_aviris,
)

N = 20


def _make(root, version):
    rng = np.random.default_rng(0)
    (root / "meta").mkdir(parents=True)
    (root / "windows").mkdir()
    clean = np.zeros(224, int)
    clean[10:176] = 1  # 166 clean bands
    pd.DataFrame({"band": range(224), "wavelength_nm": range(224), "clean": clean}).to_csv(
        root / "meta" / "bands.csv", index=False)
    classes = ["eau", "vignes"]
    ok = np.ones((N, 14), bool)
    ok[:, 5] = False
    ok[3] = False  # a window without any valid date
    rows = []
    for i in range(N):
        cls = classes[i % 2]
        cube = rng.random((14, 224, 32, 32)).astype(np.float16)
        cube[0, :, :4] = np.nan
        row = {"window_id": 1000 + i, "fold5": i % 5, "block10km": i // 2}
        if version == "fine":
            row.update(cls=cls, class_id=i % 2)
        else:
            props = rng.dirichlet(np.ones(13))
            row.update(dominant=cls, dominant_id=i % 2,
                       **{f"p_{c}": v for c, v in zip(PROPORTIONS, props, strict=True)})
        rows.append(row)
        np.savez_compressed(root / "windows" / f"{1000 + i}_{cls}.npz", cube=cube)
    pd.DataFrame(rows).to_csv(root / "meta" / "labels.csv", index=False)
    np.savez(root / "meta" / "date_mask.npz", window_id=np.arange(1000, 1000 + N), ok=ok)
    return root


@pytest.fixture
def fine_dir(tmp_path):
    return _make(tmp_path / "AVIRIS-C_CA_TS14_FINE", "fine")


@pytest.fixture
def mix_dir(tmp_path):
    return _make(tmp_path / "AVIRIS-C_CA_TS14_MIX", "mix")


def test_fine_pooled_covariances(fine_dir, tmp_path):
    b = fetch_aviris(fine_dir, cache_dir=tmp_path / "c", version="fine")
    assert b.covariances.shape == (N - 1, 83, 83)
    assert b.info["excluded_no_valid_date"] == 1
    # 13 valid dates x 1024 pixels, minus the NaN rows of date 0
    assert b.n_obs[0] == 13 * 1024 - 4 * 32
    sp = b.splits["recommended"]
    assert_no_leakage(b.meta["fold5"].to_numpy()[sp["train"]],
                      b.meta["fold5"].to_numpy()[sp["test"]])
    bk = b.splits["block10km"]
    assert_no_leakage(b.groups[bk["train"]], b.groups[bk["test"]])


def test_fine_dataset_raw_matches_covariance(fine_dir, tmp_path):
    ds = AVIRISDataset(fine_dir, cache_dir=tmp_path / "c", verbose=False, return_raw=True,
                       normalize=False)
    (cov, raw), y = ds[0]
    assert raw.shape == (13, 1024, 83)
    X = raw.numpy().reshape(-1, 83).astype(np.float64)
    X = X[np.isfinite(X).all(1)]
    np.testing.assert_allclose(cov.numpy(), np.cov(X.T, bias=True), rtol=1e-6, atol=1e-10)
    assert ds.classes == ["eau", "vignes"]
    with pytest.raises(ValueError, match="regression"):
        AVIRISDataset(fine_dir, cache_dir=tmp_path / "c", task="regression")


def test_mix_regression_and_dominant(mix_dir, tmp_path):
    ds = AVIRISDataset(mix_dir, version="mix", task="regression", cache_dir=tmp_path / "c",
                       verbose=False)
    x, y = ds[0]
    assert x.shape == (83, 83) and y.shape == (13,) and y.dtype == torch.float64
    assert ds.classes == PROPORTIONS
    dom = AVIRISDataset(mix_dir, version="mix", task="fine", cache_dir=tmp_path / "c",
                        verbose=False)
    assert isinstance(dom[0][1], int) and dom.classes == ["eau", "vignes"]
