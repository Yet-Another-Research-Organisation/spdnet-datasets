"""HyperPistachio on synthetic ENVI cubes (one bright square nut per cube, 24 bands)."""

import numpy as np
import pytest

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.hyperspectral import hyperpistachio as HP
from spdnet_datasets.hyperspectral._common import envi_cube

N_BANDS = 24


def _write_cube(path, cube, shutter):
    lines, samples, bands = cube.shape
    wl = [f"{v:.2f}" for v in np.linspace(400, 1000, bands)]
    wl = ", ".join(wl[:10]) + ",\n " + ", ".join(wl[10:])  # value over two lines
    path.with_suffix(".hdr").write_text(
        f"ENVI\ninterleave = bil\ndata type = 12\nlines = {lines}\nsamples = {samples}\n"
        f"bands = {bands}\nshutter = {shutter}\nframerate = 66.5\nbyte order = 0\n"
        f"header offset = 0\nwavelength = {{{wl}}}\n"
        "reflectance scale factor = 10000\n")
    np.ascontiguousarray(cube.transpose(0, 2, 1)).astype("<u2").tofile(path.with_suffix(".bil"))


@pytest.fixture
def pistachio_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(HP, "N_BANDS", N_BANDS)
    root = tmp_path / "HyperPistachio"
    rng = np.random.default_rng(0)
    for level in range(1, 27):
        (root / f"Level {level:02d}").mkdir(parents=True)
        for index in (1, 2):
            cube = rng.integers(100, 110, size=(40, 48, N_BANDS)).astype(np.uint16)  # dark
            size = 20 + 2 * index  # 22 or 24 pixels
            cube[8:8 + size, 10:10 + size] = rng.integers(3000, 6000, size=(size, size, N_BANDS))
            _write_cube(root / f"Level {level:02d}" / f"L{level:02d}_{index:04d}.bil", cube,
                        14.74 if level > 17 else 12.73)
    return root


def test_envi_reader_layouts(tmp_path):
    cube = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    for layout, order in (("bip", (0, 1, 2)), ("bil", (0, 2, 1)), ("bsq", (2, 0, 1))):
        hdr = tmp_path / f"c_{layout}.hdr"
        hdr.write_text(f"ENVI\nlines = 3\nsamples = 4\nbands = 5\ninterleave = {layout}\n"
                       "data type = 4\nbyte order = 0\n")
        np.ascontiguousarray(cube.transpose(order)).tofile(tmp_path / f"c_{layout}.bin")
        np.testing.assert_array_equal(envi_cube(hdr), cube)


def test_pistachio_unit_full_bands(pistachio_dir, tmp_path):
    b = HP.fetch_hyperpistachio(pistachio_dir, cache_dir=tmp_path / "c")
    assert b.covariances.shape == (52, N_BANDS, N_BANDS)
    # the opening (cross, 2 iterations) trims 3 pixels at each corner of the square
    assert set(b.meta.n_pixels) == {22 * 22 - 12, 24 * 24 - 12}
    assert sorted(set(b.labels)) == HP.LEVELS
    sp = b.splits["recommended"]
    assert_no_leakage(b.groups[sp["train"]], b.groups[sp["test"]])
    assert_no_leakage(b.groups[sp["train"]], b.groups[sp["val"]])
    idx = b.splits["by_index"]
    assert set(b.meta["index"][idx["train"]]) == {1} and set(b.meta["index"][idx["test"]]) == {2}
    assert len(idx["val"]) == 0


def test_decimation_keeps_one_band_in_f(pistachio_dir, tmp_path):
    ds = HP.HyperPistachioDataset(pistachio_dir, cache_dir=tmp_path / "c", decimate=4,
                                  return_raw=True, normalize=False, verbose=False)
    (cov, raw), _ = ds[0]
    assert cov.shape == (6, 6)
    header = pistachio_dir / "Level 01" / "L01_0001.hdr"
    mask = HP.nut_mask(envi_cube(header), HP.wavelengths(header))
    expected = np.asarray(envi_cube(header))[mask][:, ::4] / 1e4
    np.testing.assert_allclose(raw.numpy(), expected, rtol=1e-6)
    np.testing.assert_allclose(cov.numpy(), np.cov(raw.numpy().astype(float).T, bias=True),
                               rtol=1e-4)


def test_windows_inside_the_mask(pistachio_dir, tmp_path):
    b = HP.fetch_hyperpistachio(pistachio_dir, cache_dir=tmp_path / "c", unit="window",
                                window=8)
    assert b.covariances.shape[1:] == (4, 4)  # 24 bands, one in 6
    # windows of 8 on a grid of step 4 inside a square at rows 8 and columns 10 of side
    # 22 (4 x 4 windows, minus the top-right one that meets a trimmed corner) or 24 (5 x 4)
    per_cube = b.meta.groupby("cube").size()
    assert len(per_cube) == 52
    assert set(per_cube[per_cube.index.str.endswith("1")]) == {15}
    assert set(per_cube[per_cube.index.str.endswith("2")]) == {20}
    assert ((b.meta.y0 >= 8) & (b.meta.y0 + 8 <= 8 + 24)).all()
    assert (b.meta.y0 % 4 == 0).all() and (b.meta.x0 % 4 == 0).all()
    sp = b.splits["recommended"]
    assert_no_leakage(b.groups[sp["train"]], b.groups[sp["test"]])
    assert len(b.splits["by_index"]["val"]) > 0


def test_tasks(pistachio_dir, tmp_path):
    kw = {"cache_dir": tmp_path / "c", "verbose": False}
    coarse = HP.HyperPistachioDataset(pistachio_dir, **kw)
    assert coarse.classes == ["above_limit", "below_limit"]
    above = coarse.labels == coarse.class_to_idx["above_limit"]
    assert above.sum() == 18  # levels 18-26, 2 cubes each
    fine = HP.HyperPistachioDataset(pistachio_dir, task="fine", **kw)
    assert len(fine.classes) == 26
    train = fine.splits["recommended"]["train"]
    assert len(set(fine.labels[train])) == 26  # by_index: every level seen in training
    reg = HP.HyperPistachioDataset(pistachio_dir, task="regression", **kw)
    assert reg.num_classes == 0
    assert np.isclose(max(float(reg[i][1]) for i in range(len(reg))), 114.67)
