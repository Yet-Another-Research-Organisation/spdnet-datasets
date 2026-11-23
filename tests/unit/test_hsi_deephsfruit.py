"""DeepHS-Fruit v2 on a synthetic tree of raw ENVI cubes and v2 annotation files."""

import json

import numpy as np
import pytest
import torch

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.hyperspectral._common import envi_cube
from spdnet_datasets.hyperspectral.deephsfruit import (
    DeepHSFruitDataset,
    fetch_deephsfruit,
    fruit_mask,
)

B = 224


def _write_cube(path, rng, size):
    cube = np.zeros((28, 30, B), np.float32)  # zero background, as the cut-out cubes
    cube[2:2 + size, 3:3 + size] = rng.uniform(0.3, 0.8, (size, size, B))
    cube[3, 4, :5] = 1.7  # aberrant values, clipped to 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".hdr").write_text(
        f"ENVI\nsamples = 30\nlines = 28\nbands = {B}\nheader offset = 0\n"
        "file type = ENVI Standard\ndata type = 4\ninterleave = bip\nbyte order = 0\n")
    cube.tofile(path.with_suffix(".bin"))


@pytest.fixture
def fruit_dir(tmp_path):
    """2 fruits x 8 physical fruits x 2 faces. Labelled: fruits 0-3 (train), 4 (val) and
    5 (front in train, back in test: the leak of the official split); 6-7 unlabelled."""
    root = tmp_path / "DeepHS-Fruit"
    rng = np.random.default_rng(0)
    records, labelled = [], {"train": [], "val": [], "test": []}
    rid = 0
    for fruit in ("Avocado", "Kiwi"):
        for k in range(8):
            for side in ("front", "back"):
                header = f"{fruit}/VIS/day_02/{fruit.lower()}_day_02_{k:02d}_{side}.hdr"
                _write_cube(root / header, rng, 22 + (k % 3))  # n >= 2p = 448
                rec = {"id": rid, "fruit": fruit, "side": side, "day": "day_02",
                       "camera_type": "VIS", "files": {"header_file": header,
                                                       "data_file": header[:-4] + ".bin"}}
                records.append(rec)
                part = {4: "val", 5: "train" if side == "front" else "test"}.get(
                    k, "train" if k < 4 else None)
                if part:
                    labelled[part].append((rec, ("unripe", "ripe", "overripe")[k % 3]))
                rid += 1
    ann = root / "annotations"
    ann.mkdir()
    (ann / "train_all_v2.json").write_text(json.dumps({"records": records}))
    names = {"train": "train_only_labeled_v2.json", "val": "val_v2.json",
             "test": "test_v2.json"}
    for part, items in labelled.items():
        recs = [dict(r, id=100 + i) for i, (r, _) in enumerate(items)]  # own numbering
        anns = [{"id": i, "record_id": 100 + i, "ripeness_state": s, "firmness": 1000 + i,
                 "storage_days": 2} for i, (_, s) in enumerate(items)]
        (ann / names[part]).write_text(json.dumps({"records": recs, "annotations": anns}))
    return root


def test_mask_and_clipped_pixels(fruit_dir, tmp_path):
    header = fruit_dir / "Avocado/VIS/day_02/avocado_day_02_00_front.hdr"
    assert fruit_mask(envi_cube(header)).sum() == 22 * 22
    ds = DeepHSFruitDataset(fruit_dir, cache_dir=tmp_path / "c", return_raw=True,
                            normalize=False, verbose=False)
    (cov, raw), _ = ds[0]
    assert cov.shape == (B, B) and cov.dtype == torch.float64
    X = raw.numpy().astype(float)
    assert X.max() <= 1.0
    np.testing.assert_allclose(cov.numpy(), np.cov(X.T, bias=True), rtol=1e-4, atol=1e-10)


def test_fruit_mode_groups_are_physical_fruits(fruit_dir, tmp_path):
    b = fetch_deephsfruit(fruit_dir, cache_dir=tmp_path / "c")
    assert len(b.labels) == 32 and len(set(b.groups)) == 16
    assert set(b.labels) == {"Avocado", "Kiwi"} and "official" not in b.splits
    sp = b.splits["recommended"]
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_leakage(b.groups[sp[a]], b.groups[sp[c]])


def test_ripeness_mode_and_official_split(fruit_dir, tmp_path):
    b = fetch_deephsfruit(fruit_dir, cache_dir=tmp_path / "c", classification_mode="ripeness",
                          band_bin=4)
    assert b.covariances.shape == (24, 56, 56)
    assert set(b.labels) == {"unripe", "perfect", "overripe"}  # "ripe" -> "perfect"
    off = b.splits["official"]
    assert [len(off[k]) for k in ("train", "val", "test")] == [18, 4, 2]
    assert set(b.groups[off["train"]]) & set(b.groups[off["test"]])  # the leak
    sp = b.splits["recommended"]
    assert_no_leakage(b.groups[sp["train"]], b.groups[sp["test"]])
    ds = DeepHSFruitDataset(fruit_dir, cache_dir=tmp_path / "c", verbose=False,
                            classification_mode="ripeness", band_bin=4, split="official")
    assert len(ds.classes) == 3 and len(ds) == 24
