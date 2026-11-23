"""HyperPistachio: VNIR cubes of single pistachios contaminated with aflatoxin B1.

52 ENVI cubes (BIL, uint16, reflectance x 10 000; 256 lines x 384 samples x 462
bands from 386.88 to 1 003.60 nm; Resonon Pika XC2), one pistachio on a dark
background per cube: 26 contamination levels (``Level NN/LNN_000k``, k = 1, 2),
2 pistachios per level, from 0 to 114.67 µg/kg of aflatoxin B1 (table
:data:`CONCENTRATION`, copied from the description of the Zenodo record; the
concentration is known per level, not measured per nut).

Construction:

1. Pistachio mask (as the dataset atlas): Otsu threshold on the mean
   reflectance over 450-900 nm, opening (2 iterations), largest connected
   component, holes filled.
2. Bands **decimated**, not averaged: one band in ``decimate``
   (``bands = range(0, 462, decimate)``, p = ceil(462 / decimate)).
3. ``unit="pistachio"`` (default): one covariance per cube, observations = the
   pixels of the mask (about 25 000), ``decimate=1`` by default (p = 462,
   n / p ≈ 55). ``decimate=2`` gives p = 231.
   ``unit="window"``: windows of ``window x window`` pixels on a grid of step
   ``window // 2`` that lie **entirely** in the mask (2 541 windows of 32 x 32,
   20 to 71 per cube), ``decimate=6`` by default (p = 77, n = 1 024).

Labels (``groups`` = contamination level, the batch of 2 pistachios):

- ``task="coarse"`` (default): above or at most the limit of the European Union
  for aflatoxin B1 in pistachios sold to the final consumer, 8 µg/kg
  (Regulation (EU) 2023/915): 9 levels above (18-26), 17 at most;
- ``task="fine"``: the 26 levels (``"L01"`` ... ``"L26"``);
- ``task="regression"``: the concentration in µg/kg.

Splits:

- ``"by_level"`` = ``"recommended"`` (``coarse`` and ``regression``): 5
  stratified folds grouped by level (``fold`` = test fold, the next one is the
  validation): the two pistachios of a level, and all their windows, stay on the
  same side;
- ``"by_index"`` / ``"by_index_reverse"``: train = the ``_0001`` cubes, test = the
  ``_0002`` cubes (and the other way round); validation = ``val_fraction`` of the
  training windows (empty with ``unit="pistachio"``). It is the only split
  where every level is seen in training, hence the recommended split of
  ``task="fine"``; train and test pistachios of a level come from the same
  acquisition session.

.. warning::
   The level is confounded with the acquisition session: the two cubes of a
   level share shutter time, frame rate and acquisition minute, and every level
   above 8 µg/kg was acquired with a shutter of 14.74 ms (Spearman correlation
   shutter / concentration 0.87). ``meta`` has the ``shutter`` and
   ``framerate`` columns to control it, for instance by keeping the 13 levels
   at 14.74 ms.

Download
--------
Direct (Zenodo record 20027441, doi:10.5281/zenodo.20027441, no account).
Without a local copy, ``fetch_hyperpistachio`` downloads ``Dataset.zip``
(2 859 334 911 bytes, md5 checked) into ``<cache_dir>/HyperPistachio/downloads/``
and extracts it there (4.7 GB). Already downloaded::

    fetch_hyperpistachio(archive="~/Downloads/Dataset.zip")

Licence: the Zenodo metadata say CC BY-ND 4.0, but the description restricts
this subset to academic teaching and research, without redistribution or
commercial use. Apply the stricter rule.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import dataset_path
from ..core.fetch import Bunch, download, run_pipeline
from ..manager import DatasetManager
from ..utils.sources import existing_dir, source_dir
from ._common import (
    HSIDataset,
    _box_sums,
    band_index,
    covariance_stage,
    envi_cube,
    group_kfold_split,
    load_covariance_dir,
    map_splits,
    read_envi_header,
    write_json,
)

DOMAIN, NAME = "HYPERSPECTRAL", "HyperPistachio"
PAGE = "https://zenodo.org/records/20027441"
ZIP, ZIP_MD5, ZIP_SIZE = "Dataset.zip", "e1069857719b28d6a87d363c891f4d36", 2_859_334_911
URL = PAGE + "/files/Dataset.zip?download=1"
FILES = {URL: ZIP_MD5}
N_BANDS, N_LEVELS = 462, 26
EU_LIMIT = 8.0  # µg/kg, aflatoxin B1 in pistachios for the final consumer (EU 2023/915)
#: Aflatoxin B1 concentration (µg/kg) of each level, from the Zenodo description.
CONCENTRATION = {
    1: 0.00, 2: 0.40, 3: 0.67, 4: 0.88, 5: 1.13, 6: 1.66, 7: 2.15, 8: 2.30, 9: 2.48,
    10: 2.82, 11: 3.01, 12: 3.05, 13: 3.85, 14: 4.43, 15: 5.12, 16: 5.30, 17: 6.37,
    18: 8.93, 19: 12.16, 20: 17.12, 21: 24.03, 22: 26.14, 23: 33.17, 24: 56.06, 25: 57.29,
    26: 114.67}
LEVELS = [f"L{lv:02d}" for lv in range(1, N_LEVELS + 1)]
COARSE_MAP = {f"L{lv:02d}": ("above_limit" if c > EU_LIMIT else "below_limit")
              for lv, c in CONCENTRATION.items()}
DEFAULT_DECIMATE = {"pistachio": 1, "window": 6}


def cube_header(root, level, index):
    return Path(root) / f"Level {level:02d}" / f"L{level:02d}_{index:04d}.hdr"


def wavelengths(header):
    text = read_envi_header(header)["wavelength"]
    return np.array([float(v) for v in text.strip("{} ").split(",") if v.strip()])


def otsu(values, bins=256):
    """Otsu threshold (centre of the best histogram bin), as the dataset atlas."""
    hist, edges = np.histogram(values[np.isfinite(values)], bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    return float(centres[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def nut_mask(cube, wl):
    """Boolean (lines, samples) mask of the pistachio (see the module docstring)."""
    from scipy import ndimage

    sel = np.flatnonzero((wl >= 450) & (wl <= 900))
    brightness = np.asarray(cube[..., sel], dtype=np.float64).mean(-1) / 1e4
    mask = ndimage.binary_opening(brightness > otsu(brightness), iterations=2)
    labels, n = ndimage.label(mask)
    if n > 1:
        mask = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return ndimage.binary_fill_holes(mask)


def inner_windows(mask, window):
    """Top-left corners (y, x) of the windows on the grid of step ``window // 2``
    that lie entirely in ``mask``."""
    full = _box_sums(mask.astype(np.int64), window) == window * window
    step = max(window // 2, 1)
    grid = np.zeros_like(full)
    grid[::step, ::step] = True
    return np.argwhere(full & grid)


def fetch_hyperpistachio(data_dir=None, cache_dir=None, keep=None, estimator="scm",
                         force=False, verify="size", *, unit="pistachio", decimate=None,
                         window=32, n_splits=5, fold=0, val_fraction=0.2, seed=0,
                         archive=None):
    """Covariances of HyperPistachio (one per pistachio or per window).

    Parameters
    ----------
    unit : {"pistachio", "window"}
        Sample unit (see the module docstring).
    decimate : int, optional
        Keep one band in ``decimate``; default 1 (pistachio) or 6 (window).
    window : int
        Window size with ``unit="window"``.
    n_splits, fold, val_fraction, seed
        Folds of the ``by_level`` split and validation of ``by_index``.
    archive : path, optional
        ``Dataset.zip`` already downloaded, or its extracted folder.

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means``, ``labels``
    (level ``"LNN"``), ``targets`` (concentration, µg/kg), ``groups`` (level),
    ``meta`` (``cube``, ``level``, ``index``, ``concentration``, ``shutter``,
    ``framerate``, ``y0``, ``x0``, ``n_pixels``), ``info``, ``splits`` and
    ``root``.
    """
    if unit not in DEFAULT_DECIMATE:
        raise ValueError("unit must be 'pistachio' or 'window'")
    decimate = DEFAULT_DECIMATE[unit] if decimate is None else int(decimate)
    bands = band_index(N_BANDS, decimate)
    root = source_dir(
        NAME, dataset_path(DOMAIN, NAME, data_dir),
        required=("Level 01/L01_0001.hdr", "Level 26/L26_0002.hdr"), page_url=PAGE,
        expected=f"{ZIP} ({ZIP_SIZE} bytes, md5 {ZIP_MD5})", cache_dir=cache_dir,
        archive=archive, checksum=ZIP_MD5, size=ZIP_SIZE, verify=verify,
        download=lambda d: download(URL, d / ZIP, checksum=ZIP_MD5, algo="md5", size=ZIP_SIZE))
    params = {"unit": unit, "decimate": decimate,
              "window": window if unit == "window" else None,
              "source": str(Path(root).resolve()), "version": 1}

    def process(_, out_dir):
        rows, masks = [], []
        for level in range(1, N_LEVELS + 1):
            for index in (1, 2):
                header = cube_header(root, level, index)
                fields = read_envi_header(header)
                mask = nut_mask(envi_cube(header), wavelengths(header))
                masks.append(mask)
                base = {"cube": header.stem, "level": level, "index": index,
                        "label": f"L{level:02d}", "concentration": CONCENTRATION[level],
                        "shutter": float(fields["shutter"]),
                        "framerate": float(fields["framerate"]), "mask": len(masks) - 1}
                if unit == "pistachio":
                    rows.append(dict(base, y0=-1, x0=-1, n_pixels=int(mask.sum())))
                else:
                    for y0, x0 in inner_windows(mask, window):
                        rows.append(dict(base, y0=int(y0), x0=int(x0),
                                         n_pixels=window * window))
        np.save(Path(out_dir) / "masks.npy", np.stack(masks))
        pd.DataFrame(rows).to_csv(Path(out_dir) / "meta.csv", index=False)
        write_json(Path(out_dir) / "info.json", {
            "n_cubes": len(masks), "unit": unit, "p": len(bands), "decimate": decimate,
            "n_samples_processed": len(rows)})

    def pixels(proc_dir, meta):
        yield from _pixels(root, proc_dir, meta, bands, window)

    res = run_pipeline(NAME, params=params, process_fn=process,
                       covariance_fn=lambda pr, o: covariance_stage(pr, o, pixels, estimator),
                       covariance_params={"estimator": estimator},
                       cache_dir=cache_dir, keep=keep, force=force)
    out = load_covariance_dir(res.covariance_dir)
    meta = out.meta
    splits = level_splits(out.meta_all, n_splits=n_splits, fold=fold,
                          val_fraction=val_fraction, seed=seed)
    return Bunch(
        covariances=out.covariances, means=out.means, n_obs=out.n_obs,
        labels=meta["label"].to_numpy(), targets=meta["concentration"].to_numpy(np.float64),
        target_names=["concentration_ug_per_kg"], groups=meta["label"].to_numpy(),
        meta=meta, info=out.info, bands=bands,
        splits=map_splits(splits, meta["proc_index"], len(out.meta_all)),
        processed_path=res.processed_dir, covariance_path=res.covariance_dir, root=root,
    )


def _pixels(root, proc_dir, meta, bands, window):
    """Reflectances (n, p) of each sample of ``meta``, a cube read once per run of rows."""
    masks = np.load(Path(proc_dir) / "masks.npy", mmap_mode="r")
    current, cube = None, None
    for row in meta.itertuples(index=False):
        if row.cube != current:
            current = row.cube
            cube = envi_cube(cube_header(root, int(row.level), int(row.index)))
        if row.y0 < 0:
            X = np.asarray(cube[np.asarray(masks[int(row.mask)])][:, bands], dtype=np.float32)
        else:
            block = cube[row.y0:row.y0 + window, row.x0:row.x0 + window][..., bands]
            X = np.asarray(block, dtype=np.float32).reshape(-1, len(bands))
        yield X / 1e4


def level_splits(meta, *, n_splits=5, fold=0, val_fraction=0.2, seed=0):
    """``by_level`` (grouped by level) and ``by_index`` splits (module docstring)."""
    coarse = meta["label"].map(COARSE_MAP).to_numpy()
    by_level = group_kfold_split(coarse, meta["label"].to_numpy(), n_splits=n_splits,
                                 fold=fold, seed=seed)
    rng = np.random.default_rng(seed)
    index = meta["index"].to_numpy()
    windows = (meta["y0"] >= 0).any()

    def one_way(train_index):
        train = np.flatnonzero(index == train_index)
        n_val = int(round(val_fraction * len(train))) if windows else 0
        val = np.sort(rng.choice(train, n_val, replace=False))
        return {"train": np.setdiff1d(train, val), "val": val,
                "test": np.flatnonzero(index != train_index)}

    return {"by_level": by_level, "recommended": by_level, "by_index": one_way(1),
            "by_index_reverse": one_way(2)}


@DatasetManager.register_dataset("hyperpistachio")
class HyperPistachioDataset(HSIDataset):
    """HyperPistachio, aflatoxin B1 in single pistachios (52 cubes, 26 levels).

    Parameters
    ----------
    unit, decimate, window, n_splits, fold, val_fraction
        See :func:`fetch_hyperpistachio`.
    task : {"coarse", "fine", "regression"}
        ``"coarse"`` (default): above / at most the EU limit of 8 µg/kg;
        ``"fine"``: the 26 levels; ``"regression"``: the concentration.
    split : str
        ``"recommended"`` = ``"by_level"``, except with ``task="fine"`` where it
        is ``"by_index"`` (a split by level leaves the test levels unseen);
        ``"by_index"`` or ``"by_index_reverse"``.
    subset
        See :class:`~spdnet_datasets.hyperspectral._common.HSIDataset`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    COARSE_MAP = COARSE_MAP

    def __init__(self, data_dir=None, *, unit="pistachio", decimate=None, window=32,
                 task="coarse", n_splits=5, fold=0, val_fraction=0.2, split="recommended",
                 subset=None, cache_dir=None, keep=None, force=False, verify="size",
                 archive=None, **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(data_dir=str(existing_dir(root, cache_dir)), task=task, **kwargs)
        bunch = fetch_hyperpistachio(root, cache_dir, keep, self.estimator, force, verify,
                                     unit=unit, decimate=decimate, window=window,
                                     n_splits=n_splits, fold=fold, val_fraction=val_fraction,
                                     seed=self.seed, archive=archive)
        if task == "fine" and split == "recommended":
            bunch.splits["recommended"] = bunch.splits["by_index"]
        self.unit, self.window = unit, window
        self.data_dir = Path(bunch.root)
        self._setup(bunch, split, subset)

    def _load_raw(self, k):
        """Reflectances of sample ``k``: (n_pixels, p) for a pistachio, (window, window,
        p) for a window."""
        proc = self.bunch.processed_path
        if proc is None or not os.path.exists(os.path.join(proc, "masks.npy")):
            raise ValueError("return_raw needs the processed stage (keep != 'keep_covariance')")
        row = self.meta_all.iloc[[k]]
        X = next(_pixels(self.bunch.root, proc, row, self.bunch.bands, self.window))
        return X if self.unit == "pistachio" else X.reshape(self.window, self.window, -1)


__all__ = ["CONCENTRATION", "EU_LIMIT", "HyperPistachioDataset", "fetch_hyperpistachio",
           "inner_windows", "nut_mask"]
