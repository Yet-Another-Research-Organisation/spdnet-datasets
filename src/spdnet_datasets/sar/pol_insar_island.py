"""Pol-InSAR-Island: F-SAR (DLR) L- and S-band Pol-InSAR over Baltrum island.

Reference: Hochstuhl S. et al., *Pol-InSAR-Island - A benchmark dataset for
multi-frequency Pol-InSAR data land cover classification*, ISPRS Open Journal
of Photogrammetry and Remote Sensing, 2023 (KIT RADAR4KIT). Downloaded from
RADAR4KIT when absent from ``<root>/SAR/Pol-InSAR-Island`` (see "Download").

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
diagonals then ``Re``/``Im`` of the upper terms); raw (``logpower``), chosen by
``raw_layout``: ``"pixels"`` (default) = the ``(n, p)`` features of the labelled
pixels of the cell, exactly the observations of the covariance and the mean
(``n`` varies: ``batch_size=1`` or a ``collate_fn``); ``"grid"`` = the
``(32, 32, p)`` grid of the cell with the unlabelled pixels set to 0.
With ``normalize=True`` the covariances are multiplied by the geometric scale
``s``, the mean and raw by ``sqrt(s)`` (``t6`` means, which are T6 entries, by
``s``).

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

Download
--------
Direct (RADAR4KIT, doi:10.35097/1700, version 2, CC BY-NC-SA 4.0, no account; the
RADAR4KIT terms of use apply). The dataset page
https://radar.kit.edu/radar/en/dataset/IAzBEMXnbTndvIZG gives the archive URL
(``schema:contentUrl``, :data:`TAR_URL`), its size (5 567 475 200 bytes) and md5
(``9ba66a216b1ae441a5d4ef91e30a4aa8``). Without a local copy,
``fetch_pol_insar_island`` downloads this BagIt ``tar`` into
``<cache_dir>/Pol-InSAR-Island/downloads/``, checks its md5, extracts it, checks the
md5 of the inner ``data/dataset/Pol-InSAR-Island_updated.zip``
(``3d10bb4afa2eb4ece3da546df0f741fb``, ``manifest-md5.txt`` of the bag) and extracts
it (the intermediate copy of the inner zip is then deleted). A HEAD request on the
archive URL answers 401 while a GET is served without login: if the download is
refused, the page is opened and
:class:`~spdnet_datasets.core.access.ManualDownloadRequired` is raised. Already
downloaded::

    fetch_pol_insar_island(archive="~/Downloads/10.35097-1700.tar")

(the ``tar``, the inner zip, or an extracted folder).
"""

import re
import shutil
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import BaseDataset
from ..core.access import extract_archive, manual_source
from ..core.config import dataset_path, existing_dir
from ..core.estimators import (
    InsufficientSamplesError,
    NotSPDError,
    check_sample_size,
    check_spd,
    estimate_covariance,
)
from ..core.fetch import Bunch, check_local, download, md5sum, run_pipeline
from ..core.normalize import apply_scale
from ..manager import DatasetManager
from ..utils.sources import check_size, downloads_dir, find_root
from ._common import (
    check_split_groups,
    finalize_labels,
    group_kfold_splits,
    train_scale,
)

DOMAIN, NAME = "SAR", "Pol-InSAR-Island"
PAGE = "https://radar.kit.edu/radar/en/dataset/IAzBEMXnbTndvIZG"
TAR_URL = "https://www.radar-service.eu/radar-backend/archives/IAzBEMXnbTndvIZG/versions/1/content"
TAR, TAR_MD5, TAR_SIZE = "10.35097-1700.tar", "9ba66a216b1ae441a5d4ef91e30a4aa8", 5_567_475_200
INNER, INNER_MD5 = "data/dataset/Pol-InSAR-Island_updated.zip", "3d10bb4afa2eb4ece3da546df0f741fb"
FILES = {TAR_URL: TAR_MD5}
REQUIRED = ("label/FP1/label_train.bin",)


def _is_zip(path):
    # zipfile.is_zipfile() also accepts a tar ending with a zip member (the BagIt tar)
    with open(path, "rb") as f:
        return f.read(4) == b"PK\x03\x04"


def _download_root(cache_dir, archive):
    """Dataset folder from the RADAR4KIT tar (downloaded or given), its inner zip or a folder."""
    out = downloads_dir(NAME, cache_dir)
    if archive is None:
        found = find_root(out, REQUIRED)
        if found is not None:
            return found
        try:
            path = download(TAR_URL, out / TAR, checksum=TAR_MD5, algo="md5", size=TAR_SIZE)
        except OSError as exc:  # access refused (terms, login), md5 mismatch, network...
            manual_source(NAME, page_url=PAGE, dest_dir=out,
                          expected=f"the archive of the page ({TAR_SIZE} bytes, md5 {TAR_MD5}; "
                                   f"automatic download failed: {exc})")
    else:
        path = Path(archive).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        if path.is_file() and not _is_zip(path):
            check_size(path, TAR_SIZE)
            if md5sum(path) != TAR_MD5:
                raise OSError(f"Checksum mismatch for {path}")
    bag = None
    if path.is_file() and not _is_zip(path) and tarfile.is_tarfile(path):
        # only the inner zip is taken from the tar (core extract_archive would see a zip)
        bag = out / "bag"
        bag.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path) as t:
            member = next((m for m in t.getmembers() if m.name.endswith(INNER)), None)
            if member is None:
                raise FileNotFoundError(f"{path} holds no {INNER}")
            part = bag / (Path(INNER).name + ".part")
            with t.extractfile(member) as src, open(part, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
        path = part.rename(bag / Path(INNER).name)
    if path.is_dir():
        found = find_root(path, REQUIRED)
        if found is not None:
            return found
        bag_root = find_root(path, (INNER,))
        if bag_root is None:
            raise FileNotFoundError(f"{path} holds neither {REQUIRED[0]} nor {INNER}")
        path = bag_root / INNER
    if md5sum(path) != INNER_MD5:
        raise OSError(f"Checksum mismatch for {path}")
    found = find_root(extract_archive(path, out / "dataset"), REQUIRED)
    if found is None:
        raise FileNotFoundError(f"{path} holds no {REQUIRED[0]}")
    if bag is not None:
        shutil.rmtree(bag)  # intermediate copy of the inner zip (5.6 GB)
    return found
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
    """Covariance stage; cells failing the ``n >= 2p`` or SPD check are rejected.

    The rejected cells are listed in ``rejected.csv`` (``proc_index``, ``path``,
    ``row``, ``col``, ``reason``); ``meta.csv`` keeps the others with their
    ``proc_index`` (row in the processed stage).
    """
    proc, out = Path(proc), Path(out)
    masks = np.load(proc / "masks.npy")
    meta = pd.read_csv(proc / "meta.csv")
    meta["proc_index"] = np.arange(len(meta))
    if representation == "logpower":
        raw = np.load(proc / "raw.npy", mmap_mode="r")
        obs = [np.asarray(raw[k])[masks[k]].astype(np.float64) for k in range(len(masks))]
        means = np.stack([x.mean(0) for x in obs])
    else:
        prm = np.load(proc / "t6.npy")  # (N, bands, 36)
        nb = prm.shape[1]
        t6 = np.zeros((len(prm), 12 * nb, 12 * nb))
        for b in range(nb):
            t6[:, 12 * b:12 * (b + 1), 12 * b:12 * (b + 1)] = realify(t6_matrix(prm[:, b]))
        means = prm.reshape(len(prm), -1)
    covs, kept, rejected = [], [], []
    for k in range(len(masks)):
        try:
            if representation == "logpower":
                C = estimate_covariance(obs[k], estimator)
            else:
                check_sample_size(int(masks[k].sum()), t6.shape[-1])
                check_spd(t6[k])
                C = t6[k]
        except (InsufficientSamplesError, NotSPDError) as exc:  # rejected and counted
            rejected.append({"proc_index": k, "path": meta["path"].iat[k], "row": meta["row"].iat[k],
                             "col": meta["col"].iat[k], "reason": f"{type(exc).__name__}: {exc}"})
            continue
        covs.append(C)
        kept.append(k)
    p = means.shape[1] if representation == "logpower" else 12 * nb
    np.save(out / "covariances.npy", np.stack(covs) if covs else np.zeros((0, p, p)))
    np.save(out / "means.npy", means[kept])
    meta.iloc[kept].to_csv(out / "meta.csv", index=False)
    pd.DataFrame(rejected, columns=["proc_index", "path", "row", "col", "reason"]).to_csv(
        out / "rejected.csv", index=False)


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
                           cells="union", cell=32, purity=0.75, min_labelled=None, block=512,
                           archive=None):
    """Load (and cache) the Pol-InSAR-Island cell covariances.

    Parameters
    ----------
    data_dir : path, optional
        Default ``<root>/SAR/Pol-InSAR-Island``; downloaded into the cache when
        absent.
    archive : path, optional
        RADAR4KIT ``tar`` already downloaded, its inner zip or an extracted folder
        (see "Download" in the module docstring).
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
    (512 m blocks), ``meta``, ``splits``, ``rejected`` (cells whose estimate
    failed the ``n >= 2p`` or SPD check, DataFrame), ``processed_dir`` and
    ``root`` (dataset folder, to pass as ``data_dir`` to the dataset class).
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
    if not (root / REQUIRED[0]).exists():
        root = _download_root(cache_dir, archive)
    check_local(root, verify=verify, required=REQUIRED)
    opts = {"cell": cell, "purity": purity, "min_labelled": min_labelled, "block": block, "bands": bands, "cells": cells,
            "representation": representation}
    params = dict(opts, estimator=estimator, source=str(root.resolve()))
    res = run_pipeline("pol_insar_island", params=params, covariance_params={"version": 2},
                       process_fn=lambda _src, out: _process(root, out, **opts),
                       covariance_fn=lambda a, b: _covariances(a, b, representation=representation,
                                                               estimator=estimator),
                       cache_dir=cache_dir, keep=keep, force=force)
    cdir = Path(res.covariance_dir)
    meta = pd.read_csv(cdir / "meta.csv")
    return Bunch(covariances=np.load(cdir / "covariances.npy"), means=np.load(cdir / "means.npy"),
                 rejected=pd.read_csv(cdir / "rejected.csv"),
                 labels=meta["label"].to_numpy(str), groups=meta["block"].to_numpy(str), meta=meta,
                 splits=make_splits(meta, cell=cell), processed_dir=res.processed_dir,
                 covariance_dir=cdir, root=root)


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
    raw_layout : {"pixels", "grid"}
        Raw output (``logpower``): the labelled pixels ``(n, p)`` of the cell,
        the observations of ``cov`` and ``mean`` (default), or the
        ``(cell, cell, p)`` grid with the unlabelled pixels set to 0.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` arguments.
    """

    OUTPUTS = ("cov", "mean", "raw")

    def __init__(self, data_dir=None, *, split="recommended", cells=None, cache_dir=None, keep=None,
                 representation="logpower", bands=("L", "S"), cell=32, purity=0.75,
                 min_labelled=None, block=512, raw_layout="pixels", **kwargs):
        if raw_layout not in ("pixels", "grid"):
            raise ValueError("raw_layout must be 'pixels' or 'grid'")
        self.raw_layout = raw_layout
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
        self._root = root
        super().__init__(data_dir=str(existing_dir(root, cache_dir)), **kwargs)
        self._load_data()

    def _load_data(self):
        b = fetch_pol_insar_island(self._root, estimator=self.estimator, **self._fetch)
        self.data_dir = Path(b.root)
        if self.return_raw and b.processed_dir is None:
            raise ValueError("return_raw=True needs the preprocessed cells (keep != 'keep_covariance')")
        keep = finalize_labels(self, b.labels, b.groups, b.splits)
        check_split_groups({k: v for k, v in self.splits.items() if k.startswith("spatial")}, self.groups)
        self.meta = b.meta.iloc[keep].reset_index(drop=True)
        self.covariances, self.means = b.covariances[keep], b.means[keep]
        self.rejected = b.rejected
        self.scale = train_scale(self.covariances, self.splits, self.split) if self.normalize else 1.0
        self._processed = b.processed_dir
        self.samples = list(range(len(keep)))
        if self.verbose:
            print(f"Pol-InSAR-Island ({self._fetch['representation']}, {self._fetch['cells']} cells): "
                  f"{len(keep)} cells, p={self.covariances.shape[-1]}, "
                  f"{len(np.unique(self.groups))} blocks")

    def __getitem__(self, idx):
        cov = torch.from_numpy(apply_scale(self.covariances[idx], self.scale))
        # mean and raw x sqrt(s); "t6" means are T6 entries (second order): x s
        t6 = self._fetch["representation"] == "t6"
        mean = torch.from_numpy(self.means[idx] * (self.scale if t6 else np.sqrt(self.scale)))
        raw = None
        if self.return_raw:
            if not hasattr(self, "_raw"):
                self._raw = np.load(Path(self._processed) / "raw.npy", mmap_mode="r")
                self._masks = np.load(Path(self._processed) / "masks.npy", mmap_mode="r")
            k = int(self.meta["proc_index"].iat[idx])
            cell, mask = np.asarray(self._raw[k]), np.asarray(self._masks[k])
            x = cell[mask] if self.raw_layout == "pixels" else cell * mask[..., None]
            raw = torch.from_numpy((x * np.sqrt(self.scale)).astype(np.float32))
        x = self._pack(cov, mean, raw)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[idx])
