"""Rices_90 on synthetic ENVI cubes (grains above a white panel)."""

import numpy as np
import pandas as pd
import pytest

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.hyperspectral.rices90 import Rices90Dataset, fetch_rices90


def _write_envi(path, cube):
    lines, samples, bands = cube.shape
    path.with_suffix(".hdr").write_text(
        f"ENVI\nsamples = {samples}\nlines = {lines}\nbands = {bands}\n"
        "interleave = bil\ndata type = 12\n")
    np.ascontiguousarray(cube.transpose(0, 2, 1)).astype("<u2").tofile(path.with_suffix(".raw"))


def _cube(rng, n_grains):
    cube = rng.integers(90, 110, size=(120, 60, 256)).astype(np.uint16)
    cube[95:115] = 3000  # white panel
    for g in range(n_grains):
        r, c = 10 + 20 * (g // 2), 5 + 30 * (g % 2)
        cube[r:r + 15, c:c + 15] = rng.integers(1400, 1600, size=(15, 15, 256))
    return cube


@pytest.fixture
def rices_dir(tmp_path):
    root = tmp_path / "Rices_90"
    rng = np.random.default_rng(0)
    rows = []
    layout = [("A", 1, "S1"), ("A", 2, "S1"), ("B", 1, "S1"), ("B", 2, "S2")]
    for variety, bundle, session in layout:
        (root / session).mkdir(parents=True, exist_ok=True)
        if not (root / session / "black.hdr").exists():
            _write_envi(root / session / "black.raw", np.full((5, 60, 256), 90, np.uint16))
        name = f"{variety}-0{bundle}"
        _write_envi(root / session / f"{name}.raw", _cube(rng, 4))
        rows.append([variety, variety, bundle, session, name])
    pd.DataFrame(rows, columns=["Species Full Name", "Species Short Name", "Bundle Number",
                                "Folder", "File Name"]).to_csv(root / "index.csv", index=False)
    return root


def test_grains_and_bundle_split(rices_dir, tmp_path):
    b = fetch_rices90(rices_dir, cache_dir=tmp_path / "c")
    assert b.covariances.shape == (16, 64, 64)
    assert b.meta.n_pixels.between(200, 225).all()  # opening trims the corners
    sp = b.splits["recommended"]
    assert set(b.meta.bundle[sp["test"]]) == {2}
    assert_no_leakage(b.groups[sp["train"]], b.groups[sp["test"]])
    cross = b.splits["cross_session"]
    assert set(b.labels[cross["test"]]) == {"B"}  # A has both bundles in session S1


def test_raw_is_the_binned_grain(rices_dir, tmp_path):
    ds = Rices90Dataset(rices_dir, cache_dir=tmp_path / "c", verbose=False, return_raw=True,
                        normalize=False)
    (cov, raw), y = ds[0]
    assert raw.shape == (ds.meta.n_pixels[0], 64)
    X = raw.numpy().astype(float)
    np.testing.assert_allclose(cov.numpy(), np.cov(X.T, bias=True), rtol=1e-5)
    assert ds.classes == ["A", "B"]
