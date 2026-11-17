"""GSOFF: Gaofeng State Owned Forest Farm, tree species from airborne hyperspectral data.

Source: ``source/data/testarea2_new/sg_quac_test2.tif`` (572 x 906 pixels,
125 bands, float32) and its ground truth ``sg_quac_test2_gt.tif`` (codes 1-12,
0 = unlabelled), rasterised from ``tree_species.shp`` by ``1vector2raster.py``
(https://github.com/Bin-Zh/Three-dimensional-convolutional-neural-network-model-for-tree-species-classification-using-airborne-).
Class names and wavelengths are not documented: classes are ``"code k"``.

Construction (recomputed from the scene, not read from the prepared 16 x 16
windows of the dataset folder):

1. ``window x window`` windows (default 15) on the grid offset that maximises
   the number of windows of each class whose labelled pixels are at least
   ``threshold`` (0.8) of that class (:func:`._common.purity_grid_windows`,
   same rule as the original preprocessing script). Windows of different
   classes may overlap.
2. One band in ``band_step`` (default 2): p = 63. With ``band_step=1``
   (p = 125) a 15 x 15 window gives n = 225 < 2p and
   :class:`~spdnet_datasets.core.InsufficientSamplesError` is raised.
3. Covariance over the non-zero pixels of the window; windows with fewer than
   2p of them are discarded and counted.

Split (atlas recommendation): StratifiedGroupKFold(5) over 96 x 96 pixel
blocks, test = fold ``fold``, val = fold ``fold + 1``; val/test windows that
overlap a training window (and test windows overlapping a val window) are
removed.

Download
--------
Manual. ``SOURCES.json`` listed the source as unknown, but the authors' repository
above (cloned in the local ``source/``) gives a Baidu Pan link
(https://pan.baidu.com/s/1MW8Wz3_I7xf_5isV8osS9A, extraction code ``77ws``; licence
not stated, no checksum; Baidu needs an account). Its content was not checked
against the local copy: if the scene cannot be obtained there, contact the
maintainer of this package. Without a local copy, ``fetch_gsoff()`` opens the
repository page and raises
:class:`~spdnet_datasets.core.access.ManualDownloadRequired`; then pass a zip or a
folder holding ``data/testarea2_new/sg_quac_test2.tif`` and
``sg_quac_test2_gt.tif`` (the ground truth is produced from ``tree_species.shp``
by ``1vector2raster.py`` of the repository if it is missing)::

    fetch_gsoff(archive="~/Downloads/gsoff")
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import dataset_path, existing_dir
from ..core.estimators import check_sample_size
from ..core.fetch import Bunch, check_local, run_pipeline
from ..manager import DatasetManager
from ..utils.sources import downloads_dir, expose_as, source_dir
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

DOMAIN, NAME = "HYPERSPECTRAL", "GSOFF"
PAGE = ("https://github.com/Bin-Zh/Three-dimensional-convolutional-neural-network-model-"
        "for-tree-species-classification-using-airborne-")
FILES = {}  # manual: Baidu Pan link of the repository README (code 77ws), no checksum
SCENE = "source/data/testarea2_new/sg_quac_test2.tif"
GROUND_TRUTH = "source/data/testarea2_new/sg_quac_test2_gt.tif"
N_BANDS = 125


def _extract_windows(cube, gt, out_dir, *, window, threshold, bands):
    """Write windows.npy / meta.csv / info.json from a (H, W, B) scene."""
    positions, codes, purity = purity_grid_windows(gt, window, threshold)
    p = len(bands)
    windows, rows, short = [], [], 0
    for (r, c), code, pur in zip(positions, codes, purity, strict=True):
        w = np.asarray(cube[r:r + window, c:c + window][..., bands], dtype=np.float32)
        n_valid = len(valid_pixels(w.reshape(-1, p)))
        if n_valid < 2 * p:
            short += 1
            continue
        windows.append(w)
        rows.append({"row": int(r), "col": int(c), "code": int(code),
                     "label": f"code {int(code)}", "purity": float(pur), "n_valid": n_valid})
    np.save(Path(out_dir) / "windows.npy", np.stack(windows) if windows
            else np.zeros((0, window, window, p), np.float32))
    pd.DataFrame(rows).to_csv(Path(out_dir) / "meta.csv", index=False)
    write_json(Path(out_dir) / "info.json", {
        "n_candidates": len(positions), "excluded_n_lt_2p": short, "window": window,
        "p": p, "scene_shape": list(gt.shape)})


def fetch_gsoff(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                verify="size", *, window=15, band_step=2, threshold=0.8, block=96,
                n_splits=5, fold=0, seed=0, archive=None):
    """Windows covariances of GSOFF, recomputed from the source scene.

    Parameters
    ----------
    archive : path, optional
        Zip or folder holding ``data/testarea2_new/sg_quac_test2{,_gt}.tif``
        (see "Download" in the module docstring).

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means`` (B, p),
    ``labels`` (``"code k"``), ``groups`` (96-pixel block), ``positions``
    (top-left row, column), ``meta``, ``info``, ``splits``
    (``"spatial"`` = ``"recommended"``), ``window``, ``processed_path`` and
    ``root`` (dataset folder, to pass as ``data_dir`` to :class:`GSOFFDataset`).
    """
    root = dataset_path(DOMAIN, NAME, data_dir)
    if (root / SCENE).exists() and (root / GROUND_TRUTH).exists():
        check_local(root, verify=verify, required=(SCENE, GROUND_TRUTH))  # MANIFEST_NAS.tsv
    else:
        source = source_dir(
            NAME, root / "source",
            required=tuple(p.removeprefix("source/") for p in (SCENE, GROUND_TRUTH)),
            page_url=PAGE, cache_dir=cache_dir, archive=archive, verify=verify,
            expected="data/testarea2_new/sg_quac_test2.tif (Baidu Pan link of the README, "
                     "code 77ws); source otherwise unknown: contact the package maintainer")
        root = expose_as(source, "source", downloads_dir(NAME, cache_dir) / "root")
    bands = band_index(N_BANDS, band_step)
    check_sample_size(window * window, len(bands))
    params = {"window": window, "band_step": band_step, "threshold": threshold,
              "source": str(root.resolve()), "version": 2}

    def process(_, out_dir):
        import tifffile

        cube = tifffile.imread(root / SCENE, maxworkers=1)  # (H, W, B)
        gt = tifffile.imread(root / GROUND_TRUTH)
        if cube.shape[:2] != gt.shape or cube.shape[-1] != N_BANDS:
            raise ValueError(f"Unexpected scene {cube.shape} / ground truth {gt.shape}")
        _extract_windows(cube, gt, out_dir, window=window, threshold=threshold, bands=bands)

    def covariance(proc_dir, out_dir):
        covariance_stage(proc_dir, out_dir, window_pixels, estimator)

    res = run_pipeline(NAME, params=params, process_fn=process, covariance_fn=covariance,
                       covariance_params={"estimator": estimator},
                       cache_dir=cache_dir, keep=keep, force=force)
    out = load_covariance_dir(res.covariance_dir)
    meta, full = out.meta, out.meta_all
    spatial = block_kfold_split(full[["row", "col"]].to_numpy(), full["label"].to_numpy(),
                                window=window, block=block, n_splits=n_splits, fold=fold,
                                seed=seed)
    out.info["split_dropped_overlap"] = spatial.pop("dropped")
    spatial = map_splits({"s": spatial}, meta["proc_index"], len(full))["s"]
    positions = meta[["row", "col"]].to_numpy()
    labels = meta["label"].to_numpy()
    return Bunch(
        covariances=out.covariances, means=out.means, n_obs=out.n_obs, labels=labels,
        groups=(positions[:, 0] // block) * 10**6 + positions[:, 1] // block,
        positions=positions, window=window, meta=meta, info=out.info,
        splits={"spatial": spatial, "recommended": spatial},
        processed_path=res.processed_dir, covariance_path=res.covariance_dir, root=root,
    )


@DatasetManager.register_dataset("gsoff")
class GSOFFDataset(HSIDataset):
    """GSOFF tree species windows (15 x 15, p = 63 by default), spatial-block split.

    Parameters
    ----------
    data_dir : path, optional
        Dataset folder (default ``<SPDNET_DATA_ROOT>/HYPERSPECTRAL/GSOFF``).
    window, band_step, threshold, block, n_splits, fold
        See :func:`fetch_gsoff`.
    split : str
        Split used for the normalisation and balancing (``"recommended"``).
    subset : {None, "train", "val", "test"}
        Restrict the dataset to one part of ``split``.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options (``return_mean``,
        ``return_raw``, ``estimator``, ``task``, ``classes``, ``balance``...).
    """

    def __init__(self, data_dir=None, *, window=15, band_step=2, threshold=0.8, block=96,
                 n_splits=5, fold=0, split="recommended", subset=None, cache_dir=None,
                 keep=None, force=False, verify="size", **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(data_dir=str(existing_dir(root, cache_dir)), **kwargs)
        bunch = fetch_gsoff(root, cache_dir, keep, self.estimator, force, verify,
                            window=window, band_step=band_step, threshold=threshold,
                            block=block, n_splits=n_splits, fold=fold)
        self.data_dir = Path(bunch.root)
        self._setup(bunch, split, subset)


GSOffDataset = GSOFFDataset  # former class name

__all__ = ["GSOFFDataset", "GSOffDataset", "fetch_gsoff"]
