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

Download (manual): the scene is distributed after a form on
http://park.itc.u-tokyo.ac.jp/sal/hyperdata (CC BY 3.0). The windows read here
(``prepared_v2/chikusei_v2.h5``) are a LISTIC re-extraction from ``source/``
(atlas), not a published file; copy both folders and place it under ``data_dir``
(default ``<data root>/HYPERSPECTRAL/Chikusei``).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import dataset_path
from ..core.estimators import check_sample_size
from ..core.fetch import Bunch, check_local, run_pipeline
from ..manager import DatasetManager
from ._common import (
    HSIDataset,
    band_index,
    covariance_stage,
    load_covariance_dir,
    valid_pixels,
    write_json,
)

DOMAIN, NAME = "HYPERSPECTRAL", "Chikusei"
FILES = {}  # http://park.itc.u-tokyo.ac.jp/sal/hyperdata (form); local copy
SOURCE = "source/HyperspecVNIR_Chikusei_20140729.mat"
PREPARED = "prepared_v2/chikusei_v2.h5"
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


def fetch_chikusei(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                   verify="size", *, window=15, band_step=2):
    """Covariances of the Chikusei v2 windows (15 x 15 or 23 x 23).

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means``, ``labels``
    (class names), ``groups`` (320-pixel block), ``positions`` (top-left
    corners in the scene frame), ``meta``, ``info``, ``splits``
    (``"spatial"`` = ``"recommended"``).
    """
    if window not in (15, 23):
        raise ValueError("window must be 15 or 23 (sizes of the v2 extraction)")
    root = dataset_path(DOMAIN, NAME, data_dir)
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
        prepared_path=root / PREPARED, covariance_path=res.covariance_dir,
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
        super().__init__(data_dir=str(root), **kwargs)
        bunch = fetch_chikusei(root, cache_dir, keep, self.estimator, force, verify,
                               window=window, band_step=band_step)
        self._setup(bunch, split, subset)
        self._h5 = None

    def _load_raw(self, k):
        import h5py

        if self._h5 is None:
            self._h5 = h5py.File(self.bunch.prepared_path, "r")
        i = int(self.meta_all["h5_index"].iloc[k])
        return self._h5[f"w{self.bunch.window}/windows"][i][..., self.bunch.bands]


__all__ = ["ChikuseiDataset", "fetch_chikusei", "read_scene_crop"]
