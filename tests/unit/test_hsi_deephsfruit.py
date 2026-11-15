"""DeepHS-Fruit minimal adaptation on a synthetic cov_VIS/ + annotations tree."""

import json

import numpy as np
import pytest
import torch

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.hyperspectral.deephsfruit import (
    DeepHSFruitDataset,
    fetch_deephsfruit,
)


@pytest.fixture
def fruit_dir(tmp_path):
    root = tmp_path / "DeepHS-Fruit"
    (root / "cov_VIS").mkdir(parents=True)
    (root / "annotations").mkdir()
    merged, v2, ann = [], [], []
    rid = 0
    for fruit in ("Avocado", "Kiwi"):
        for k in range(10):
            for side in ("front", "back"):
                header = f"{fruit}/VIS/day_01/{fruit.lower()}_day_01_{k:02d}_{side}.hdr"
                rec = {"id": rid, "fruit": fruit, "side": side, "day": "day_01",
                       "camera_type": "VIS", "files": {"header_file": header}}
                merged.append(rec)
                torch.save(torch.eye(4, dtype=torch.float64) * (rid + 1),
                           root / "cov_VIS" / f"{fruit.lower()}_VIS_day_01_{rid:04d}_{side}.pt")
                # the v2 files number their records differently
                v2.append(dict(rec, id=100 + rid))
                ann.append({"record_id": 100 + rid,
                            "ripeness_state": ("unripe", "ripe", "overripe")[k % 3]})
                rid += 1
    (root / "annotations/merged_annotations.json").write_text(json.dumps({"records": merged}))
    (root / "annotations/train_only_labeled_v2.json").write_text(
        json.dumps({"records": v2, "annotations": ann}))
    for name in ("val_v2.json", "test_v2.json"):
        (root / "annotations" / name).write_text(json.dumps({"records": [], "annotations": []}))
    return root


def test_groups_are_physical_fruits(fruit_dir):
    b = fetch_deephsfruit(fruit_dir)
    assert len(b.labels) == 40 and len(set(b.groups)) == 20
    sp = b.splits["recommended"]
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_leakage(b.groups[sp[a]], b.groups[sp[c]])


def test_ripeness_matched_by_header_not_record_id(fruit_dir):
    b = fetch_deephsfruit(fruit_dir, classification_mode="ripeness")
    row = b.meta.set_index("record").loc[2]  # fruit 01 front -> k = 1 -> "ripe"
    assert row.ripeness == "perfect"
    assert np.allclose(b.covariances[int(np.flatnonzero(b.meta.record == 2)[0])], 3 * np.eye(4))


def test_dataset_warns_and_serves_covariances_only(fruit_dir):
    with pytest.warns(UserWarning, match="DeepHS-Fruit"):
        ds = DeepHSFruitDataset(fruit_dir, verbose=False)
    x, y = ds[0]
    assert x.shape == (4, 4) and x.dtype == torch.float64 and ds.classes == ["Avocado", "Kiwi"]
    with pytest.raises(ValueError), pytest.warns(UserWarning):
        DeepHSFruitDataset(fruit_dir, return_raw=True)
