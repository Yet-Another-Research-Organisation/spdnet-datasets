"""UAV-HSI-Crop: UAV hyperspectral crop mapping (Niu et al., Computers and
Electronics in Agriculture 201, 107297, 2022; ScienceDB doi 10.57760/sciencedb.01898).

Source (``source/``): 433 patches ``rs`` 96 x 96 x 200 (reflectance, 400-1000
nm, 0.1 m) and ``gt`` 96 x 96 (crop codes, 0 = unlabelled), named
``<SCENE>_patch_<k>_<row>_<col>.npy`` under ``Train/Training``,
``Train/Validation`` and ``Test`` (official split drawn at random by patch).
The (row, column) indices form complete grids: XJM 9 x 17, MJK_N 13 x 10,
MJK_S 15 x 10 patches. Class names are not available: classes are ``"code k"``.

Construction (from ``source/``, on the official grid):

1. Each scene is recomposed from its patches at (96 row, 96 column). The
   former prepared windows (``*_W15_T80``) used ``k`` as the row index and
   juxtaposed non-neighbouring patches; they are only read by the deprecated
   ``UAVLegacyDataset``.
2. 15 x 15 windows on the per-class best grid offset, purity >= 0.8
   (:func:`._common.purity_grid_windows`, as the original script).
3. One band in ``band_step`` (default 2): p = 100; n = 225 >= 2p = 200.
   Windows with fewer than 2p non-zero pixels are discarded and counted.

Splits (``positions`` are per scene; scenes never share a block):

- ``"spatial"`` = ``"recommended"``: StratifiedGroupKFold(5) over 480-pixel
  blocks (5 x 5 patches) of each scene, test = fold ``fold``, val = fold
  ``fold + 1``, overlapping val/test windows removed;
- ``"official"``: windows lying inside a single patch take the official split
  of that patch; windows straddling two patches are left out;
- ``"scene_transfer"`` (when both are loaded): train MJK_N, test MJK_S.

Download
--------
Manual (ScienceDB, doi:10.57760/sciencedb.01898, version V6, CC BY-SA 4.0; the
page is rendered in JavaScript and no stable file URL was found). The record
holds one file of 1 017 930 890 bytes (name not recorded, no checksum). Without a
local copy, ``fetch_uav()`` opens the page and raises
:class:`~spdnet_datasets.core.access.ManualDownloadRequired`; then pass the file
(zip or tar; its size is checked) or the extracted folder::

    fetch_uav(archive="~/Downloads/<file of the ScienceDB record>")

It is extracted into ``<cache_dir>/UAV-HSI-Crop/downloads/``; the folder holding
``Train/Training/gt`` and ``Test/gt`` is used as ``source/`` (symbolic link
``downloads/root/source``).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import dataset_path
from ..core.estimators import check_sample_size
from ..core.fetch import Bunch, run_pipeline
from ..manager import DatasetManager
from ..utils.sources import downloads_dir, existing_dir, expose_as, source_dir
from ._common import (
    HSIDataset,
    band_index,
    block_kfold_split,
    covariance_stage,
    load_covariance_dir,
    map_splits,
    purity_grid_windows,
    valid_pixels,
    window_pixels,
    write_json,
)

DOMAIN, NAME = "HYPERSPECTRAL", "UAV-HSI-Crop"
PAGE = "https://www.scidb.cn/detail?dataSetId=6de15e4ec9b74dacab12e29cb557f041"
ARCHIVE_SIZE = 1_017_930_890  # single file of the ScienceDB record V6 (name unknown)
FILES = {}  # manual download (ScienceDB 10.57760/sciencedb.01898)
SCENES = ("MJK_N", "MJK_S", "XJM")
PARTS = {"Train/Training": "train", "Train/Validation": "val", "Test": "test"}
PATCH, N_BANDS = 96, 200
_PATCH_RE = re.compile(r"^(?P<scene>.+)_patch_(?P<k>\d+)_(?P<row>\d+)_(?P<col>\d+)\.npy$")


def list_patches(source):
    """DataFrame of the source patches: scene, k, row, col, part, rs, gt paths."""
    rows = []
    for sub, part in PARTS.items():
        for gt in sorted((Path(source) / sub / "gt").glob("*.npy")):
            m = _PATCH_RE.match(gt.name)
            if m is None:
                continue
            rows.append({"scene": m["scene"], "k": int(m["k"]), "row": int(m["row"]),
                         "col": int(m["col"]), "part": part, "gt": str(gt),
                         "rs": str(gt.parent.parent / "rs" / gt.name)})
    return pd.DataFrame(rows)


def recompose_scene(patches, bands):
    """Scene (H, W, p) float32, ground truth (H, W) and patch-part map from its patches."""
    n_r, n_c = patches["row"].max() + 1, patches["col"].max() + 1
    if len(patches) != n_r * n_c:
        raise ValueError(f"Incomplete patch grid: {len(patches)} patches for {n_r} x {n_c}")
    cube = np.zeros((n_r * PATCH, n_c * PATCH, len(bands)), np.float32)
    gt = np.zeros((n_r * PATCH, n_c * PATCH), np.int16)
    part = np.empty((n_r, n_c), dtype=object)
    for p in patches.itertuples():
        r, c = p.row * PATCH, p.col * PATCH
        cube[r:r + PATCH, c:c + PATCH] = np.load(p.rs, mmap_mode="r")[..., bands]
        gt[r:r + PATCH, c:c + PATCH] = np.load(p.gt)
        part[p.row, p.col] = p.part
    return cube, gt, part


def fetch_uav(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
              verify="size", *, scenes=SCENES, window=15, band_step=2, threshold=0.8,
              block=480, n_splits=5, fold=0, seed=0, archive=None):
    """Window covariances of UAV-HSI-Crop, extracted from the source patches.

    Parameters
    ----------
    archive : path, optional
        File of the ScienceDB record, or its extracted folder (see "Download"
        in the module docstring).

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means``, ``labels``
    (``"code k"``), ``groups`` (scene and 480-pixel block), ``positions``
    (top-left corner in its scene), ``scene`` (index into ``scenes``),
    ``meta``, ``info``, ``splits`` and ``root`` (dataset folder holding
    ``source/``, to pass as ``data_dir`` to :class:`UAVDataset`).
    """
    scenes = tuple(scenes)
    if not set(scenes) <= set(SCENES):
        raise ValueError(f"scenes must be among {SCENES}")
    source = source_dir(
        NAME, dataset_path(DOMAIN, NAME, data_dir) / "source",
        required=("Test/gt", "Train/Training/gt"), page_url=PAGE,
        expected=f"the single file of the record, version V6 ({ARCHIVE_SIZE} bytes)",
        cache_dir=cache_dir, archive=archive, size=ARCHIVE_SIZE, verify=verify)
    root = expose_as(source, "source", downloads_dir(NAME, cache_dir) / "root")
    source = root / "source"
    bands = band_index(N_BANDS, band_step)
    check_sample_size(window * window, len(bands))
    params = {"scenes": list(scenes), "window": window, "band_step": band_step,
              "threshold": threshold, "source": str(root.resolve()), "version": 2}

    def process(_, out_dir):
        patches = list_patches(source)
        windows, rows, info = [], [], {"p": len(bands), "window": window, "scenes": {}}
        for s, scene in enumerate(scenes):
            cube, gt, part = recompose_scene(patches[patches["scene"] == scene], bands)
            positions, codes, purity = purity_grid_windows(gt, window, threshold)
            short = 0
            for (r, c), code, pur in zip(positions, codes, purity, strict=True):
                w = cube[r:r + window, c:c + window]
                n_valid = len(valid_pixels(w.reshape(-1, len(bands))))
                if n_valid < 2 * len(bands):
                    short += 1
                    continue
                pr0, pc0 = r // PATCH, c // PATCH
                inside = (r + window - 1) // PATCH == pr0 and (c + window - 1) // PATCH == pc0
                windows.append(w)
                rows.append({"scene": scene, "scene_index": s, "row": int(r), "col": int(c),
                             "code": int(code), "label": f"code {int(code)}",
                             "purity": float(pur), "n_valid": n_valid,
                             "official": part[pr0, pc0] if inside else "cross"})
            info["scenes"][scene] = {"shape": list(gt.shape), "n_candidates": len(positions),
                                     "excluded_n_lt_2p": short}
        np.save(Path(out_dir) / "windows.npy", np.stack(windows))
        pd.DataFrame(rows).to_csv(Path(out_dir) / "meta.csv", index=False)
        info["excluded_n_lt_2p"] = sum(v["excluded_n_lt_2p"] for v in info["scenes"].values())
        write_json(Path(out_dir) / "info.json", info)

    res = run_pipeline(NAME, params=params, process_fn=process,
                       covariance_fn=lambda p, o: covariance_stage(p, o, window_pixels, estimator),
                       covariance_params={"estimator": estimator},
                       cache_dir=cache_dir, keep=keep, force=force)
    out = load_covariance_dir(res.covariance_dir)
    meta, full = out.meta, out.meta_all
    spatial = block_kfold_split(full[["row", "col"]].to_numpy(), full["label"].to_numpy(),
                                window=window, block=block, n_splits=n_splits, fold=fold,
                                seed=seed, scene=full["scene_index"].to_numpy())
    out.info["split_dropped_overlap"] = spatial.pop("dropped")
    spatial = map_splits({"s": spatial}, meta["proc_index"], len(full))["s"]
    positions = meta[["row", "col"]].to_numpy()
    scene = meta["scene_index"].to_numpy()
    labels = meta["label"].to_numpy()
    official = {k: np.flatnonzero(meta["official"].to_numpy() == k)
                for k in ("train", "val", "test")}
    splits = {"spatial": spatial, "recommended": spatial, "official": official}
    if {"MJK_N", "MJK_S"} <= set(scenes):
        names = meta["scene"].to_numpy()
        splits["scene_transfer"] = {"train": np.flatnonzero(names == "MJK_N"),
                                    "val": np.zeros(0, int),
                                    "test": np.flatnonzero(names == "MJK_S")}
    groups = (scene * 10**6 + positions[:, 0] // block) * 10**6 + positions[:, 1] // block
    return Bunch(
        covariances=out.covariances, means=out.means, n_obs=out.n_obs, labels=labels,
        groups=groups, positions=positions, scene=scene, scenes=scenes, window=window,
        meta=meta, info=out.info, splits=splits,
        processed_path=res.processed_dir, covariance_path=res.covariance_dir, root=root,
    )


@DatasetManager.register_dataset("uav")
class UAVDataset(HSIDataset):
    """UAV-HSI-Crop 15 x 15 windows (p = 100) on the official patch grid.

    Parameters
    ----------
    scenes : tuple of {"MJK_N", "MJK_S", "XJM"}
        Scenes to load (all by default; evaluate per scene or with
        ``split="scene_transfer"``).
    window, band_step, threshold, block, n_splits, fold
        See :func:`fetch_uav`.
    split, subset
        See :class:`~spdnet_datasets.hyperspectral._common.HSIDataset`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    def __init__(self, data_dir=None, *, scenes=SCENES, window=15, band_step=2,
                 threshold=0.8, block=480, n_splits=5, fold=0, split="recommended",
                 subset=None, cache_dir=None, keep=None, force=False, verify="size",
                 **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(data_dir=str(existing_dir(root, cache_dir)), **kwargs)
        bunch = fetch_uav(root, cache_dir, keep, self.estimator, force, verify,
                          scenes=scenes, window=window, band_step=band_step,
                          threshold=threshold, block=block, n_splits=n_splits, fold=fold)
        self.data_dir = Path(bunch.root)
        self._setup(bunch, split, subset)


UAVHSICropDataset = UAVDataset

__all__ = ["UAVDataset", "UAVHSICropDataset", "fetch_uav", "list_patches", "recompose_scene"]
