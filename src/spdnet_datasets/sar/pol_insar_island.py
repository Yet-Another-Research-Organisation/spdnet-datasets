"""Pol-InSAR-Island: F-SAR (DLR) L- and S-band Pol-InSAR over Baltrum island.

Reference: Hochstuhl S. et al., *Pol-InSAR-Island - A benchmark dataset for
multi-frequency Pol-InSAR data land cover classification*, ISPRS Open Journal
of Photogrammetry and Remote Sensing, 2023 (KIT RADAR4KIT). Manual download
(BagIt archive of KIT); the data must be under ``<root>/SAR/Pol-InSAR-Island``.

Data: two flight paths (``FP1``, ``FP2``), each with the 6 x 6 Pol-InSAR
coherency matrix T6 of every 1 m pixel per band (``data/<FP>/<band>/T6``:
6 real diagonals and 15 complex upper terms, ENVI rasters) and the official
labels ``label/<FP>/label_{train,test}.bin`` (12 classes, 0 = unlabelled).
Georeferencing uses the label headers (UTM 32N); the FP1 L-band headers are
wrong (zone 30, origin 0). FP1 and FP2 overlap on 728 m.

Sample unit: a non-overlapping 32 x 32 px cell of a flight path, labelled by its
majority class when at least 512 of its pixels are labelled (256 for the
official per-split cells) and the majority class holds ``purity`` (0.75) of
them (atlas definition).

Representations (``representation``)
-------------------------------------
- ``"logpower"`` (default): real features = ``log`` of the 6 diagonal terms
  T11..T66 of each band (``p = 6 * len(bands)``, 12 with L and S), observations
  = the labelled pixels of the cell (``n >= 512 >= 2p``). Any estimator of
  :mod:`spdnet_datasets.core.estimators`. It is the only construction with
  cross-band (L-S) correlations; it ignores the phases.
- ``"t6"``: the mean T6 of the labelled pixels of the cell (a 6 x 6 Hermitian
  matrix per band: each pixel T6 has rank one, so the mean is the sample
  covariance of the scattering vectors). Each band is mapped to the real SPD
  matrix ``[[Re T, -Im T], [Im T, Re T]]`` (12 x 12; injective, it commutes
  with products, inverses and logarithms and doubles every eigenvalue) and the
  bands are put on a block diagonal (24 x 24 with L and S: the L-S
  cross terms cannot be computed from per-band T6). Only ``estimator="scm"``
  applies; ``return_raw`` is not available.

Outputs: covariance; mean = per-channel mean over the labelled pixels
(``logpower``: ``p`` log-powers; ``t6``: the 36 real T6 parameters per band,
diagonals then ``Re``/``Im`` of the upper terms); raw (``logpower``) = the
``(32, 32, p)`` features of every pixel of the cell (unlabelled ones included).

Splits
------
- ``"spatial512_fold{k}"`` (``"recommended"`` = fold 0): ``StratifiedGroupKFold
  (5)`` on 512 m UTM blocks (shared by FP1 and FP2) of the union cells
  (train + test labels merged). The validation part is a grouped fold of the
  rest. Guard band: training and validation cells touching a test cell (cell
  centres closer than two cells, Chebyshev, in UTM) are removed.
- ``"fp1_to_fp2"``: train on FP1 cells, test on FP2 cells, both outside the
  728 m overlap (acquisition-geometry shift).
- ``"official"`` (with ``cells="official"``): cells built separately on the
  official ``label_train`` and ``label_test`` pixels. Train and test form a
  256 x 256 px checkerboard: neighbouring cells sit on both sides and the
  scores are optimistic (atlas: +0.03 to +0.08 of balanced accuracy).

Download (manual): KITopenData / RADAR4KIT, doi:10.35097/1700 (CC BY-NC-SA 4.0): the
BagIt archive holds ``data/dataset/Pol-InSAR-Island_updated.zip`` (md5
``3d10bb4afa2eb4ece3da546df0f741fb`` in ``manifest-md5.txt``); no stable direct URL
was found. Extract the inner zip and place it under ``data_dir`` (default ``<data
root>/SAR/Pol-InSAR-Island``).
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import BaseDataset
from ..core.config import dataset_path
from ..core.estimators import check_sample_size, check_spd, estimate_covariance
from ..core.fetch import Bunch, check_local, run_pipeline
from ..core.normalize import apply_scale
from ..manager import DatasetManager
from ._common import (
    check_split_groups,
    finalize_labels,
    group_kfold_splits,
    train_scale,
)

DOMAIN, NAME = "SAR", "Pol-InSAR-Island"
FILES = {}  # manual download (KIT BagIt archive)
CLASSES = ("tidal flat", "water", "coastal shrub", "dense high vegetation", "white dune",
           "peat bog", "grey dune", "couch grass", "upper saltmarsh", "lower saltmarsh", "sand",
           "settlement")
PATHS = ("FP1", "FP2")
UPPER = [(i, j) for i in range(6) for j in range(i + 1, 6)]


def read_envi(path, dtype):
    """Read a single-band ENVI raster (shape from ``<path>.hdr`` or ``<stem>.hdr``)."""
    path = Path(path)
    hdr = path.with_name(path.name + ".hdr")
    if not hdr.exists():
        hdr = path.with_suffix(".hdr")
    text = hdr.read_text()
    shape = (int(re.search(r"lines\s*=\s*(\d+)", text).group(1)),
             int(re.search(r"samples\s*=\s*(\d+)", text).group(1)))
    return np.fromfile(path, dtype=dtype).reshape(shape)


def utm_origin(label_path):
    """(easting, northing) of the top-left corner from a label ENVI header."""
    text = Path(str(label_path) + ".hdr").read_text()
    east, north = re.search(r"map info = \{UTM,\s*1,\s*1,\s*([\d.]+),\s*([\d.]+)", text).groups()
    return float(east), float(north)


def realify(hermitian):
    """Hermitian ``(..., p, p)`` -> real ``(..., 2p, 2p)`` ``[[Re, -Im], [Im, Re]]``."""
    re_, im = hermitian.real, hermitian.imag
    return np.concatenate([np.concatenate([re_, -im], -1), np.concatenate([im, re_], -1)], -2)


def _cells(label, cell, purity, min_labelled):
    """Kept cells of one label raster: list of (r, c, class index 0..11, mask)."""
    out = []
    for r in range(label.shape[0] // cell):
        for c in range(label.shape[1] // cell):
            w = label[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            counts = np.bincount(w[w > 0], minlength=13)
            if counts.sum() < min_labelled or counts.max() < purity * counts.sum():
                continue
            out.append((r, c, int(counts.argmax()) - 1, w > 0))
    return out


def _t6_params(folder, r0, c0, cell, masks):
    """Mean over the masked pixels of the 36 real T6 parameters of each cell."""
    names = [f"T{i}{i}" for i in range(1, 7)] + [f"T{i + 1}{j + 1}_{k}" for k in ("real", "imag")
                                                 for i, j in UPPER]
    out = np.zeros((len(masks), 36))
    for t, name in enumerate(names):
        x = read_envi(Path(folder) / f"{name}.bin", "<f4")
        for k, (r, c, m) in enumerate(zip(r0, c0, masks, strict=True)):
            out[k, t] = x[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell][m].mean(dtype=np.float64)
    return out


def _process(root, out_dir, *, cell, purity, min_labelled, block, bands, cells, representation):
    root, out_dir = Path(root), Path(out_dir)
    rows, raws, masks, params = [], [], [], []
    for fp in PATHS:
        lab_train = read_envi(root / f"label/{fp}/label_train.bin", "u1")
        lab_test = read_envi(root / f"label/{fp}/label_test.bin", "u1")
        east, north = utm_origin(root / f"label/{fp}/label_train.bin")
        sources = ({"all": np.maximum(lab_train, lab_test)} if cells == "union"
                   else {"train": lab_train, "test": lab_test})
        for part, label in sources.items():
            kept = _cells(label, cell, purity, min_labelled)
            if not kept:
                continue
            r0 = [k[0] for k in kept]
            c0 = [k[1] for k in kept]
            m = [k[3] for k in kept]
            if representation == "logpower":
                feats = np.stack([np.log(np.maximum(read_envi(root / f"data/{fp}/{b}/T6/T{i}{i}.bin", "<f4"),
                                                    1e-12)) for b in bands for i in range(1, 7)], -1)
                raws += [feats[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell].astype(np.float32)
                         for r, c in zip(r0, c0, strict=True)]
                del feats
            else:
                params.append(np.stack([_t6_params(root / f"data/{fp}/{b}/T6", r0, c0, cell, m)
                                        for b in bands], 1))
            masks += m
            for r, c, y, mk in kept:
                x = east + (c + 0.5) * cell
                yy = north - (r + 0.5) * cell
                rows.append({"path": fp, "part": part, "row": r, "col": c, "label": CLASSES[y],
                             "n_pixels": int(mk.sum()), "easting": x, "northing": yy,
                             "block": f"{int(x // block)}_{int(yy // block)}"})
    pd.DataFrame(rows).to_csv(out_dir / "meta.csv", index=False)
    np.save(out_dir / "masks.npy", np.stack(masks))
    if representation == "logpower":
        np.save(out_dir / "raw.npy", np.stack(raws))
    else:
        np.save(out_dir / "t6.npy", np.concatenate(params))


def t6_matrix(params):
    """36 real parameters ``(..., 36)`` -> Hermitian T6 ``(..., 6, 6)``."""
    T = np.zeros((*params.shape[:-1], 6, 6), complex)
    for i in range(6):
        T[..., i, i] = params[..., i]
    for k, (i, j) in enumerate(UPPER):
        z = params[..., 6 + k] + 1j * params[..., 21 + k]
        T[..., i, j], T[..., j, i] = z, np.conj(z)
    return T


def _covariances(proc, out, *, representation, estimator):
    proc, out = Path(proc), Path(out)
    masks = np.load(proc / "masks.npy")
    if representation == "logpower":
        raw = np.load(proc / "raw.npy", mmap_mode="r")
        obs = [np.asarray(raw[k])[masks[k]].astype(np.float64) for k in range(len(masks))]
        covs = np.stack([estimate_covariance(x, estimator) for x in obs])
        means = np.stack([x.mean(0) for x in obs])
    else:
        prm = np.load(proc / "t6.npy")  # (N, bands, 36)
        nb = prm.shape[1]
        covs = np.zeros((len(prm), 12 * nb, 12 * nb))
        for b in range(nb):
            covs[:, 12 * b:12 * (b + 1), 12 * b:12 * (b + 1)] = realify(t6_matrix(prm[:, b]))
        for k in range(len(prm)):
            check_sample_size(int(masks[k].sum()), covs.shape[-1])
        check_spd(covs)
        means = prm.reshape(len(prm), -1)
    np.save(out / "covariances.npy", covs)
    np.save(out / "means.npy", means)
    (out / "meta.csv").write_bytes((proc / "meta.csv").read_bytes())


def _guard(coords, keep_idx, ref_idx, gap):
    """Indices of ``keep_idx`` farther than ``gap`` (Chebyshev) from every ``ref_idx``."""
    if not len(keep_idx) or not len(ref_idx):
        return keep_idx
    from scipy.spatial import cKDTree

    d, _ = cKDTree(coords[ref_idx]).query(coords[keep_idx], p=np.inf)
    return keep_idx[d >= gap]


def make_splits(meta, *, cell=32, n_splits=5, seed=0):
    """Named splits of the cells described by ``meta`` (see the module docstring)."""
    y, groups = meta["label"].to_numpy(), meta["block"].to_numpy()
    coords = meta[["easting", "northing"]].to_numpy()
    if set(meta["part"]) <= {"train", "test"}:
        tr = np.flatnonzero(meta["part"].to_numpy() == "train")
        te = np.flatnonzero(meta["part"].to_numpy() == "test")
        return {"official": {"train": tr, "val": tr[:0], "test": te}}
    splits = {}
    for name, parts in group_kfold_splits(y, groups, n_splits=n_splits, seed=seed,
                                          prefix="spatial512_").items():
        name = name.replace("spatial512_5", "spatial512")
        test = parts["test"]
        splits[name] = {"train": _guard(coords, parts["train"], test, 2 * cell),
                        "val": _guard(coords, parts["val"], test, 2 * cell), "test": test}
    splits["recommended"] = splits["spatial512_fold0"]
    x = meta["easting"].to_numpy()
    fp = meta["path"].to_numpy()
    lo = x[fp == "FP2"].min() - cell / 2 if (fp == "FP2").any() else np.inf
    hi = x[fp == "FP1"].max() + cell / 2 if (fp == "FP1").any() else -np.inf
    outside = (x + cell / 2 <= lo) | (x - cell / 2 >= hi)
    splits["fp1_to_fp2"] = {"train": np.flatnonzero((fp == "FP1") & outside), "val": np.zeros(0, int),
                            "test": np.flatnonzero((fp == "FP2") & outside)}
    return splits


def fetch_pol_insar_island(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                           verify="size", *, representation="logpower", bands=("L", "S"),
                           cells="union", cell=32, purity=0.75, min_labelled=None, block=512):
    """Load (and cache) the Pol-InSAR-Island cell covariances.

    Parameters
    ----------
    data_dir : path, optional
        Default ``<root>/SAR/Pol-InSAR-Island``.
    cache_dir, keep, force
        See :func:`spdnet_datasets.core.fetch.run_pipeline`.
    estimator : str
        Any estimator for ``"logpower"``; ``"scm"`` only for ``"t6"``.
    verify : {"size", "md5", None}
    representation : {"logpower", "t6"}
    bands : tuple of {"L", "S"}
    cells : {"union", "official"}
        Cells on the merged labels (spatial and transfer splits) or built per
        official split (checkerboard, optimistic).
    cell, purity, block
        Cell side (px = m), majority purity, side of the spatial groups (m).
    min_labelled : int, optional
        Labelled pixels required in a cell. Default: half the cell (512) for
        union cells, a quarter (256) for official cells, where each split only
        holds part of the labels (atlas counts: 5033 / 3315 + 3403 cells).

    Returns
    -------
    Bunch with ``covariances`` (unscaled), ``means``, ``labels``, ``groups``
    (512 m blocks), ``meta``, ``splits`` and ``processed_dir``.
    """
    if representation not in ("logpower", "t6"):
        raise ValueError("representation must be 'logpower' or 't6'")
    if representation == "t6" and estimator != "scm":
        raise ValueError("representation='t6' averages given T6 matrices: only estimator='scm' applies")
    if cells not in ("union", "official"):
        raise ValueError("cells must be 'union' or 'official'")
    if not set(bands) <= {"L", "S"} or not bands:
        raise ValueError("bands must be a non-empty subset of ('L', 'S')")
    bands = tuple(bands)
    if min_labelled is None:
        min_labelled = cell * cell // (2 if cells == "union" else 4)
    root = dataset_path(DOMAIN, NAME, data_dir)
    check_local(root, verify=verify, required=("label/FP1/label_train.bin",))
    opts = {"cell": cell, "purity": purity, "min_labelled": min_labelled, "block": block, "bands": bands, "cells": cells,
            "representation": representation}
    params = dict(opts, estimator=estimator, source=str(root.resolve()))
    res = run_pipeline("pol_insar_island", params=params,
                       process_fn=lambda _src, out: _process(root, out, **opts),
                       covariance_fn=lambda a, b: _covariances(a, b, representation=representation,
                                                               estimator=estimator),
                       cache_dir=cache_dir, keep=keep, force=force)
    cdir = Path(res.covariance_dir)
    meta = pd.read_csv(cdir / "meta.csv")
    return Bunch(covariances=np.load(cdir / "covariances.npy"), means=np.load(cdir / "means.npy"),
                 labels=meta["label"].to_numpy(str), groups=meta["block"].to_numpy(str), meta=meta,
                 splits=make_splits(meta, cell=cell), processed_dir=res.processed_dir,
                 covariance_dir=cdir)


@DatasetManager.register_dataset("pol_insar_island")
class PolInSARIslandDataset(BaseDataset):
    """Pol-InSAR-Island 32 x 32 m cells as SPD covariances.

    Parameters
    ----------
    data_dir : path, optional
    split : str
        Split whose training part fits the normalisation: ``"recommended"``,
        ``"spatial512_fold{k}"``, ``"fp1_to_fp2"`` or, with ``cells="official"``,
        ``"official"`` (emits a ``UserWarning``: optimistic).
    cells : {"union", "official"}, optional
        Default ``"official"`` if ``split == "official"`` else ``"union"``.
    cache_dir, keep, representation, bands, cell, purity, min_labelled, block
        See :func:`fetch_pol_insar_island`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` arguments.
    """

    OUTPUTS = ("cov", "mean", "raw")

    def __init__(self, data_dir=None, *, split="recommended", cells=None, cache_dir=None, keep=None,
                 representation="logpower", bands=("L", "S"), cell=32, purity=0.75,
                 min_labelled=None, block=512, **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        cells = cells or ("official" if split == "official" else "union")
        if split == "official":
            warnings.warn("Pol-InSAR-Island official split: 256 x 256 px checkerboard, train and test "
                          "cells are neighbours; scores are optimistic. Prefer split='recommended'.",
                          UserWarning, stacklevel=2)
        if kwargs.get("return_raw") and representation == "t6":
            raise ValueError("return_raw is only available with representation='logpower'")
        self.split = split
        self._fetch = {"cache_dir": cache_dir, "keep": keep, "representation": representation,
                       "bands": tuple(bands), "cells": cells, "cell": cell, "purity": purity,
                       "min_labelled": min_labelled, "block": block}
        super().__init__(data_dir=str(root), **kwargs)
        self._load_data()

    def _load_data(self):
        b = fetch_pol_insar_island(self.data_dir, estimator=self.estimator, **self._fetch)
        if self.return_raw and b.processed_dir is None:
            raise ValueError("return_raw=True needs the preprocessed cells (keep != 'keep_covariance')")
        keep = finalize_labels(self, b.labels, b.groups, b.splits)
        check_split_groups({k: v for k, v in self.splits.items() if k.startswith("spatial")}, self.groups)
        self.meta = b.meta.iloc[keep].reset_index(drop=True)
        self.covariances, self.means = b.covariances[keep], b.means[keep]
        self.scale = train_scale(self.covariances, self.splits, self.split) if self.normalize else 1.0
        self._processed, self._keep = b.processed_dir, keep
        self.samples = list(range(len(keep)))
        if self.verbose:
            print(f"Pol-InSAR-Island ({self._fetch['representation']}, {self._fetch['cells']} cells): "
                  f"{len(keep)} cells, p={self.covariances.shape[-1]}, "
                  f"{len(np.unique(self.groups))} blocks")

    def __getitem__(self, idx):
        cov = torch.from_numpy(apply_scale(self.covariances[idx], self.scale))
        mean = torch.from_numpy(self.means[idx])
        raw = None
        if self.return_raw:
            if not hasattr(self, "_raw"):
                self._raw = np.load(Path(self._processed) / "raw.npy", mmap_mode="r")
            raw = torch.from_numpy(np.array(self._raw[self._keep[idx]], dtype=np.float32))
        x = self._pack(cov, mean, raw)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[idx])
