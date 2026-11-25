"""FUSAR-Ship: MMSI union-find groups and leakage-free folds."""

import numpy as np
import pytest
import tifffile

from spdnet_datasets import DatasetManager
from spdnet_datasets.sar.fusar_ship import FUSARShipDataset, mmsi_groups


def test_union_find_groups(tmp_path):
    csv = tmp_path / "meta.csv"
    csv.write_text("id,mmsi,length,path\n"
                   "A,1,10,Cargo\\X\nB,1,10,Cargo\\X\nB,2,10,Cargo\\X\nC,2,10,Cargo\\X\n"
                   "D,,,Cargo\\X\nE,3,10,Cargo\\X\n")
    g = mmsi_groups(["A", "B", "C", "D", "E", "F"], csv)
    assert g[0] == g[1] == g[2]  # A-B share 1, B-C share 2
    assert len({g[0], g[3], g[4], g[5]}) == 4  # D (empty MMSI) and F (no row) alone


@pytest.fixture
def fusar_dir(tmp_path):
    rng = np.random.default_rng(0)
    rows = ["id,mmsi,length,path"]
    k = 0
    for cat, n in (("Cargo", 40), ("Fishing", 40)):
        (tmp_path / cat / "Sub").mkdir(parents=True)
        for _i in range(n):
            k += 1
            cid = f"Ship_C01S01N{k:04d}"
            tifffile.imwrite(tmp_path / cat / "Sub" / f"{cid}.tiff", rng.integers(0, 256, (16, 16), dtype=np.uint8))
            rows.append(f"{cid},{1000 + (k - 1) // 4},1,{cat}\\Sub")  # 4 chips per vessel
    (tmp_path / "meta.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_folds_never_split_a_vessel(fusar_dir):
    ds = FUSARShipDataset(fusar_dir, image_size=16, return_raw=True, verbose=False)
    assert len(np.unique(ds.groups)) == 20
    for name, parts in ds.splits.items():
        if name == "random":  # sample-level split, leaky by design
            continue
        for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
            assert not set(ds.groups[parts[a]]) & set(ds.groups[parts[b]])
    assert DatasetManager.DATASETS["fusar"] is DatasetManager.DATASETS["fusar_ship"] is FUSARShipDataset
    from spdnet_datasets.real.fusar import FUSARShipDataset as Old

    assert Old is FUSARShipDataset
