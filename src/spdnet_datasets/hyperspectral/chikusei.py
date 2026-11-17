"""Chikusei: airborne VNIR scene over Chikusei (Japan), 19 land-cover classes.

Source: ``source/HyperspecVNIR_Chikusei_20140729.mat`` (MATLAB v7.3, 128 bands,
2517 x 2335 pixels at 2.5 m, Headwall Hyperspec-VNIR-C, 29/07/2014) and its
ground truth (Yokoya & Iwasaki, technical report, University of Tokyo, 2016;
CC BY 3.0).

Orientation: h5py reads the MATLAB variable as (B, W, H) = (128, 2335, 2517);
the scene is ``transpose(2, 1, 0)`` = (H, W, B), the frame of the ground
truth (:func:`read_scene_crop`). The former v1 windows were cut from the
non-transposed array and 718 / 760 of them were mislabelled; they are not
used.

Construction: the windows of the atlas re-extraction v2
(``prepared_v2/chikusei_v2.h5``, built from ``source/`` in the right frame):
15 x 15 windows centred on labelled pixels of a 4-pixel grid, label = class
of the central pixel. One band in ``band_step`` (default 2): p = 64, n = 225
>= 2p (``band_step=1`` gives p = 128 and raises
:class:`~spdnet_datasets.core.InsufficientSamplesError`).

Split: the v2 spatial split (8 x 8 blocks of 320 pixels assigned to
train / val / test; a window is kept when its centre is at Chebyshev
distance >= w + 1 from any pixel of another split). Windows of the buffer
(645 in 15 x 15) are discarded and counted. Windows of the same split overlap
(grid step 4 < 15): never split them at random.

Download
--------
Direct (no account). Without a local copy holding ``prepared_v2/chikusei_v2.h5``,
``fetch_chikusei`` downloads ``Hyperspec_Chikusei_MATLAB.zip`` (1 437 816 076
bytes) from the Space Application Laboratory of the University of Tokyo
(https://www.sal.t.u-tokyo.ac.jp/hyperdata/) into
``<cache_dir>/Chikusei/downloads/``. No checksum is published: the size of the zip
is checked and the CRC-32 of every member while it is extracted. The v2 windows
are then rebuilt from the scene (:func:`prepare_v2`, about 1.5 GB of memory) with
the block assignment of the atlas extraction (:data:`BLOCK_SPLIT`), so they are
identical to the local ``prepared_v2/chikusei_v2.h5`` (checked on BERTHA). Licence:
Creative Commons (CC BY 3.0 on the mirrors); acknowledge the Space Application
Laboratory and cite the report SAL-2016-05-27. Already downloaded::

    fetch_chikusei(archive="~/Downloads/Hyperspec_Chikusei_MATLAB.zip")

(the zip, or a folder holding the two ``.mat`` files).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import dataset_path
from ..core.estimators import check_sample_size
from ..core.fetch import Bunch, check_local, download, run_pipeline
from ..manager import DatasetManager
from ..utils.sources import downloads_dir, existing_dir, source_dir
from ._common import (
    HSIDataset,
    band_index,
    covariance_stage,
    load_covariance_dir,
    valid_pixels,
    write_json,
)

DOMAIN, NAME = "HYPERSPECTRAL", "Chikusei"
PAGE = "https://www.sal.t.u-tokyo.ac.jp/hyperdata/"
ZIP, ZIP_SIZE = "Hyperspec_Chikusei_MATLAB.zip", 1_437_816_076
URL = PAGE + ZIP
FILES = {URL: None}  # no published checksum: size checked, member CRC-32 on extraction
SCENE_FILE = "HyperspecVNIR_Chikusei_20140729.mat"
GT_FILE = "HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat"
SOURCE = f"source/{SCENE_FILE}"
PREPARED = "prepared_v2/chikusei_v2.h5"
# Spatial split of the atlas v2 extraction (prepared_v2/summary_v2.json): squares of
# BLOCK pixels, block id = (row // BLOCK) * BLOCK_COLS + col // BLOCK, code 0/1/2 =
# train/val/test (best of 60 000 random assignments, seed 0).
BLOCK, BLOCK_COLS, STRIDE = 320, 8, 4
BLOCK_SPLIT = (0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 2, 2, 1, 0, 2, 0, 0, 1, 0, 1, 1, 0,
               0, 2, 0, 0, 0, 1, 0, 1, 1, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0,
               0, 0, 0, 2, 0, 1, 0, 0, 0, 1)
N_BANDS = 128
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


def read_scene_crop(mat_path, r0, r1, c0, c1, key="chikusei"):
    """Rows ``r0:r1`` and columns ``c0:c1`` of the scene, as (h, w, B).

    The MATLAB v7.3 file stores the cube column-major: h5py returns it as
    (B, W, H), hence the transposition.
    """
    import h5py

    with h5py.File(mat_path, "r") as f:
        return np.transpose(f[key][:, c0:c1, r0:r1], (2, 1, 0))


def prepare_v2(source, out, sizes=(15, 23)):
    """Rebuild the v2 windows of ``prepared_v2/chikusei_v2.h5`` from the two ``.mat`` files.

    Same rule as the atlas extraction: centres = labelled pixels on a
    :data:`STRIDE`-pixel grid at least ``w // 2`` from the border, label = GT
    of the centre, split of the centre's block (:data:`BLOCK_SPLIT`), kept when
    the centre is at Chebyshev distance >= ``w + 1`` of any pixel of another
    split (``split = -1`` otherwise). Only the datasets read by
    :func:`fetch_chikusei` are written (``scene/class_names`` and, per size,
    ``windows``, ``centers``, ``labels``, ``split``, ``block``, ``purity``,
    ``n_valid``).
    """
    import h5py
    from scipy import io as sio
    from scipy import ndimage

    source, out = Path(source), Path(out)
    g = sio.loadmat(source / GT_FILE, squeeze_me=True, struct_as_record=False)["GT"]
    gt = np.asarray(g.gt).astype(np.uint8)
    names = [str(x) for x in g.classnames]
    with h5py.File(source / SCENE_FILE, "r") as f:
        cube = np.transpose(f["chikusei"][()], (2, 1, 0))  # (H, W, B), frame of gt
    H, W, B = cube.shape
    rr, cc = np.mgrid[0:H, 0:W]
    block_map = ((rr // BLOCK) * BLOCK_COLS + cc // BLOCK).astype(np.int32)
    split_map = np.asarray(BLOCK_SPLIT, dtype=np.int8)[block_map]
    del rr, cc
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with h5py.File(tmp, "w") as fh:
        fh.create_dataset("scene/class_names", data=np.array(names, dtype=h5py.string_dtype()))
        for w in sizes:
            h = w // 2
            r, c = np.nonzero(gt > 0)
            m = ((r % STRIDE == 0) & (c % STRIDE == 0) & (r >= h) & (r < H - h)
                 & (c >= h) & (c < W - h))
            C = np.stack([r[m], c[m]], 1).astype(np.int32)
            sp = split_map[C[:, 0], C[:, 1]]
            keep = np.zeros(len(C), bool)
            for s in range(3):
                dt = ndimage.distance_transform_cdt(split_map == s, metric="chessboard")
                keep |= (sp == s) & (dt[C[:, 0], C[:, 1]] >= w + 1)
            y = gt[C[:, 0], C[:, 1]]
            grp = fh.create_group(f"w{w}")
            win = grp.create_dataset("windows", shape=(len(C), w, w, B), dtype=cube.dtype,
                                     chunks=(1, w, w, B), compression="gzip", compression_opts=4)
            purity = np.zeros(len(C), np.float32)
            n_valid = np.zeros(len(C), np.int16)
            for i, (r0, c0) in enumerate(C):
                x = cube[r0 - h:r0 + h + 1, c0 - h:c0 + h + 1]
                win[i] = x
                purity[i] = (gt[r0 - h:r0 + h + 1, c0 - h:c0 + h + 1] == y[i]).mean()
                n_valid[i] = x.reshape(-1, B).any(1).sum()
            for key, val in {"centers": C, "labels": y,
                             "split": np.where(keep, sp, -1).astype(np.int8),
                             "block": block_map[C[:, 0], C[:, 1]], "purity": purity,
                             "n_valid": n_valid}.items():
                grp.create_dataset(key, data=val)
    tmp.rename(out)
    return out


def _download_root(root, cache_dir, archive, verify):
    """Folder with ``prepared_v2/chikusei_v2.h5`` built in the cache from the SAL zip."""
    out = downloads_dir(NAME, cache_dir)
    if (out / PREPARED).exists():
        return out
    src = source_dir(
        NAME, root / "source", required=(SCENE_FILE, GT_FILE), page_url=PAGE,
        expected=f"{ZIP} ({ZIP_SIZE} bytes)", cache_dir=cache_dir, archive=archive,
        size=ZIP_SIZE, verify=verify,
        download=lambda d: download(URL, d / ZIP, size=ZIP_SIZE))
    prepare_v2(src, out / PREPARED)
    return out


def fetch_chikusei(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                   verify="size", *, window=15, band_step=2, archive=None):
    """Covariances of the Chikusei v2 windows (15 x 15 or 23 x 23).

    Parameters
    ----------
    archive : path, optional
        ``Hyperspec_Chikusei_MATLAB.zip`` already downloaded, or a folder with
        its two ``.mat`` files (see "Download" in the module docstring).

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means``, ``labels``
    (class names), ``groups`` (320-pixel block), ``positions`` (top-left
    corners in the scene frame), ``meta``, ``info``, ``splits``
    (``"spatial"`` = ``"recommended"``) and ``root`` (dataset folder, to pass
    as ``data_dir`` to :class:`ChikuseiDataset`).
    """
    if window not in (15, 23):
        raise ValueError("window must be 15 or 23 (sizes of the v2 extraction)")
    root = dataset_path(DOMAIN, NAME, data_dir)
    if not (root / PREPARED).exists():
        root = _download_root(root, cache_dir, archive, verify)
    check_local(root, verify=verify, required=(PREPARED,))
    bands = band_index(N_BANDS, band_step)
    check_sample_size(window * window, len(bands))
    group = f"w{window}"
    params = {"window": window, "band_step": band_step, "source": str(root.resolve()), "version": 2}

    def process(_, out_dir):
        import h5py

        with h5py.File(root / PREPARED, "r") as f:
            names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in f["scene/class_names"][:]]
            split = f[f"{group}/split"][:]
            meta = pd.DataFrame({
                "h5_index": np.arange(len(split)),
                "center_row": f[f"{group}/centers"][:, 0],
                "center_col": f[f"{group}/centers"][:, 1],
                "code": f[f"{group}/labels"][:].astype(int),
                "split": split.astype(int),
                "block": f[f"{group}/block"][:],
                "purity": f[f"{group}/purity"][:],
                "n_valid": f[f"{group}/n_valid"][:],
            })
        meta["label"] = [names[c - 1] for c in meta["code"]]
        meta["row"] = meta["center_row"] - window // 2
        meta["col"] = meta["center_col"] - window // 2
        buffer = meta["split"] < 0
        short = meta["n_valid"] < 2 * len(bands)
        meta[~buffer & ~short].to_csv(Path(out_dir) / "meta.csv", index=False)
        write_json(Path(out_dir) / "info.json", {
            "n_windows_v2": len(meta), "excluded_buffer": int(buffer.sum()),
            "excluded_n_lt_2p": int((short & ~buffer).sum()), "window": window,
            "p": len(bands), "class_names": names})

    def pixels(proc_dir, meta):
        import h5py

        with h5py.File(root / PREPARED, "r") as f:
            windows = f[f"{group}/windows"]
            for i in meta["h5_index"]:
                w = windows[int(i)][..., bands].astype(np.float64)
                yield valid_pixels(w.reshape(-1, len(bands)))

    res = run_pipeline(NAME, params=params, process_fn=process,
                       covariance_fn=lambda p, o: covariance_stage(p, o, pixels, estimator),
                       covariance_params={"estimator": estimator},
                       cache_dir=cache_dir, keep=keep, force=force)
    out = load_covariance_dir(res.covariance_dir)
    meta = out.meta
    split = meta["split"].to_numpy()
    spatial = {name: np.flatnonzero(split == code) for code, name in SPLIT_NAMES.items()}
    return Bunch(
        covariances=out.covariances, means=out.means, n_obs=out.n_obs,
        labels=meta["label"].to_numpy(), groups=meta["block"].to_numpy(),
        positions=meta[["row", "col"]].to_numpy(), window=window, meta=meta, info=out.info,
        splits={"spatial": spatial, "recommended": spatial}, bands=bands,
        prepared_path=root / PREPARED, covariance_path=res.covariance_dir, root=root,
    )


@DatasetManager.register_dataset("chikusei")
class ChikuseiDataset(HSIDataset):
    """Chikusei v2 windows (15 x 15, p = 64 by default), spatial split by 320-pixel blocks.

    Parameters
    ----------
    data_dir : path, optional
        Dataset folder (default ``<SPDNET_DATA_ROOT>/HYPERSPECTRAL/Chikusei``).
    window, band_step
        See :func:`fetch_chikusei`.
    split, subset
        See :class:`~spdnet_datasets.hyperspectral._common.HSIDataset`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    def __init__(self, data_dir=None, *, window=15, band_step=2, split="recommended",
                 subset=None, cache_dir=None, keep=None, force=False, verify="size",
                 **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(data_dir=str(existing_dir(root, cache_dir)), **kwargs)
        bunch = fetch_chikusei(root, cache_dir, keep, self.estimator, force, verify,
                               window=window, band_step=band_step)
        self.data_dir = Path(bunch.root)
        self._setup(bunch, split, subset)
        self._h5 = None

    def _load_raw(self, k):
        import h5py

        if self._h5 is None:
            self._h5 = h5py.File(self.bunch.prepared_path, "r")
        i = int(self.meta_all["h5_index"].iloc[k])
        return self._h5[f"w{self.bunch.window}/windows"][i][..., self.bunch.bands]


__all__ = ["ChikuseiDataset", "fetch_chikusei", "prepare_v2", "read_scene_crop"]
