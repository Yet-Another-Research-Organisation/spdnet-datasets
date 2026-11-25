"""AVIRIS-C California: hyperspectral time series of 600 m windows, multilabel land cover.

2 034 windows of 600 m x 600 m (32 x 32 pixels of 18.75 m, EPSG:5070) in
California, each observed at 14 dates (2008-2020, mostly 2013-2015) by
AVIRIS-Classic (NASA/JPL L2 surface reflectance, distributed by the ORNL DAAC),
with 155 of the 224 bands (``bands.csv``: base mask of the water-vapour and
detector-edge regions, isolated-band spikes, wings of the 940 nm feature), a
per-date quality flag (``qc_ok``) and multilabel land-cover annotations from
the USDA NASS Cropland Data Layer (CDL): 12 fine classes plus ``other``
(:data:`CLASSES`), a class being present when it covers at least 10 % of the
mapped CDL pixels. Release: Hugging Face ``musmb/aviris-ca-multilabel``
(CC BY 4.0), 10 Parquet shards ``data/fold_<k>-0000<i>-of-00002.parquet``, one
row per window (cube ``(14, 155, 32, 32)`` float16, NaN = no data).

Sample unit (``mode``)
----------------------
- ``mode="date"`` (default): one ``p x p`` covariance per (window, date) that is
  acquired and passes the quality control (24 211 of 28 476 window-dates;
  ``qc=False`` keeps the 28 245 acquired ones). Observations = the pixels of the
  window at that date: reflectance clipped to ``clip`` = [-0.05, 1.5] (the card
  reports isolated values up to 1e4 in the SWIR), pixels with a non-finite band
  dropped; a date needs ``min_pixels`` (default ``2p`` = 310) of them, so
  n = 819 to 1 024 and n / p ≈ 6.6. Default estimator: Ledoit-Wolf.
- ``mode="series"``: one sample per window, the 14 date covariances as a
  series: ``x = (tokens, mask[, mean][, raw])`` with ``tokens`` (14, p, p),
  ``mask`` (14,) bool (``False``: padded, not acquired, failed QC or too few
  pixels; such tokens are identity matrices), as the covariance series of the
  package. Windows with fewer than ``min_dates`` valid dates are dropped.

Both modes read the same cached covariances (one per valid window-date).
``band_step`` keeps one band in ``band_step`` (p = ceil(155 / band_step)).

Targets (``task``)
------------------
- ``task="multilabel"`` (default): float32 0/1 vector of the 13 classes;
- ``task="regression"``: the 13 CDL proportions (float64, sum to 1);
- ``task="fine"``: the dominant class (largest proportion), single label.

``label_level="date"`` (default of ``mode="date"``): labels of the CDL of the
acquisition year; ``label_level="window"`` (the only level of
``mode="series"``): window labels, the proportions averaged over the distinct
CDL years of the window's dates, thresholded at 10 %.

Splits
------
The release assigns whole 10.2 km blocks (``block10km``) to 5 spatial folds
balanced on the multilabels. ``"fold<k>"``: test = fold ``k``, validation =
fold ``k - 1`` (mod 5), train = the other three. ``"recommended"`` =
``"train_val_test"`` = ``"fold4"`` (train folds 0-2, val 3, test 4), the
``train_val_test`` configuration of the release. ``groups`` = ``block10km``
(the split unit: windows and dates of a block are never separated);
``"random"`` is sample-level, not stratified with the multilabel task, and
leaky (dates of a window, neighbouring windows). Windows never overlap, but
1.0 % of the windows touch a window of another fold (2.8 % within 1 km).

Download
--------
Hugging Face dataset ``musmb/aviris-ca-multilabel``, **private** for now (a read
token of an account with access: ``HF_TOKEN``, ``~/.cache/huggingface/token``,
the keyring or a prompt, see :mod:`spdnet_datasets.core.access`), pinned at
revision :data:`HF_REVISION`. Without a local copy, the shards of the requested
``folds`` and ``bands.csv`` are downloaded once (sizes and sha256 =
LFS identifiers checked) into ``<cache_dir>/aviris_ca_ml/downloads/<revision>/``
(7.1 GB for all folds), outside the ``keep`` policy. A local copy of the release
(``git clone`` / ``huggingface-cli download`` folder) is read in place::

    AVIRISCAMultilabelDataset(data_dir="~/aviris-ca-multilabel")   # or archive=

Needs ``pyarrow`` (``pip install 'spdnet-datasets[aviris]'``).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core.config import dataset_path
from ..core.estimators import InsufficientSamplesError, NotSPDError, estimate_covariance
from ..core.fetch import Bunch, run_pipeline, sha256sum
from ..manager import DatasetManager
from ..utils.sources import downloads_dir, existing_dir
from ._common import HSIDataset, write_json

DOMAIN, NAME = "HYPERSPECTRAL", "AVIRIS-CA-Multilabel"
HF_REPO = "musmb/aviris-ca-multilabel"
HF_REVISION = "0cc7afc9d49c18faa37622683e3839d4e30a24b9"
HF_PRIVATE = True  # a token is required while the repository is private
PAGE = f"https://huggingface.co/datasets/{HF_REPO}"
N_FOLDS = 5
#: Files of the pinned revision read by the loader (+ metadata tables): size, sha256.
HF_FILES = {
    "bands.csv": (24_472, "e66de5e325af0f9bddc1803f52a1e076e38e51c356676dddf320e5c637126b5e"),
    "classes.csv": (1_669, "46fce9db9172cc34c011d4b11ef490c873505d9fa592664b56b7f93e8bf9902a"),
    "metadata/windows.parquet": (
        187_995, "6d010a26110a74e7f2d9869eb2d8345f572bece031bf22531ee9577a91525195"),
    "metadata/dates.parquet": (
        702_010, "6b8dd32a179bc9017ee1ca95d3a694fcf6de3c091f595589455d89d8a693bc19"),
    "data/fold_0-00000-of-00002.parquet": (
        732_312_762, "9121b4858bccf3c39da3a2bfcc1acaab045b52ac8660c9f0121ebf07ec463255"),
    "data/fold_0-00001-of-00002.parquet": (
        705_307_426, "08264738fd0828a97041fbf28b7fb7eac420d82d497336d29547a9d1a8205de2"),
    "data/fold_1-00000-of-00002.parquet": (
        701_797_735, "827a029cdd818f7aab084aa0c56de4461e526a0d7dabb9c195232d83d43f4104"),
    "data/fold_1-00001-of-00002.parquet": (
        686_935_969, "c78ee2f752a404270e5886bfd84915c9e9ab89e9186765b805cf6bce6bf6871d"),
    "data/fold_2-00000-of-00002.parquet": (
        696_164_241, "5b9311400d5925bc7e1c59843289aa38554eb02a43b16589cdb89a1c993afa27"),
    "data/fold_2-00001-of-00002.parquet": (
        684_759_713, "64d071f4eb46e0443c3ff6aca4a45d3516122c8e81e135838c4411729d905737"),
    "data/fold_3-00000-of-00002.parquet": (
        719_460_354, "85a98f9d657fba478b42111b1008c8f47d4ff705dcf29421c2a809f9f9050b65"),
    "data/fold_3-00001-of-00002.parquet": (
        683_344_828, "9d86c179660d335fdee9c5bbba14bae8946b1aded18f56e1d104e051a5a6f84a"),
    "data/fold_4-00000-of-00002.parquet": (
        744_278_886, "53477f5b122bb9eb4704c66e9af5cb10b41f10aa1e5f3e3d1656cbab6f07740e"),
    "data/fold_4-00001-of-00002.parquet": (
        716_910_909, "2d89631da102f40594b8770ebe59c276819c0d974e83163702a0f2c0c6a1c71d"),
}
FILES = {f"{PAGE}/resolve/{HF_REVISION}/{path}": sha for path, (_, sha) in HF_FILES.items()}
CLASSES = ("almond", "shrubland", "water", "evergreen_forest", "mixed_forest",
           "grassland_pasture", "barren", "developed_high", "developed_low",
           "developed_medium", "developed_open", "grapes", "other")
CLIP = (-0.05, 1.5)
MODES = ("date", "series")
LABEL_LEVELS = ("date", "window")
TASKS = ("multilabel", "regression", "fine")
WINDOW_COLUMNS = ["window_id", "fold", "zone", "block10km", "lat", "lon", "x_min", "y_max",
                  "n_ok_dates"]


def shard_paths(folds=None):
    """Release paths of the shards of ``folds`` (default: all), in fold order."""
    folds = range(N_FOLDS) if folds is None else sorted({int(k) for k in folds})
    bad = [k for k in folds if not 0 <= k < N_FOLDS]
    if bad:
        raise ValueError(f"folds must be in 0..{N_FOLDS - 1}, got {bad}")
    return [p for k in folds for p in sorted(HF_FILES) if p.startswith(f"data/fold_{k}-")]


def _check_file(path, rel, verify):
    if verify is None or rel not in HF_FILES:
        return
    size, sha = HF_FILES[rel]
    if path.stat().st_size != size:
        raise OSError(f"{path}: {path.stat().st_size} bytes, expected {size} (revision "
                      f"{HF_REVISION[:8]}); pass verify=None for a modified copy")
    if verify == "sha256" and sha256sum(path) != sha:
        raise OSError(f"Checksum mismatch for {path} (revision {HF_REVISION[:8]})")


def download_files(paths, cache_dir=None, interactive=None):
    """Download release files of the pinned revision; return their folder.

    Files already in ``<cache_dir>/aviris_ca_ml/downloads/<revision>/`` with the
    right size are kept; the others are downloaded with a Hugging Face token
    (private repository) and checked (size, sha256).
    """
    from ..core.access import hf_download

    dest = downloads_dir("aviris_ca_ml", cache_dir) / HF_REVISION
    for rel in paths:
        size, sha = HF_FILES[rel]
        target = dest / rel
        if target.exists() and target.stat().st_size == size:
            continue
        hf_download(HF_REPO, rel, dest, revision=HF_REVISION, gated=HF_PRIVATE,
                    checksum=sha, size=size, interactive=interactive)
    return dest


def locate_release(data_dir=None, *, archive=None, cache_dir=None, folds=None, verify="size"):
    """Folder holding ``bands.csv`` and the shards of ``folds``.

    Tried in order: ``data_dir`` (default ``<root>/HYPERSPECTRAL/AVIRIS-CA-Multilabel``),
    ``archive`` (a folder with the release layout, read in place), then the
    download cache (missing files are downloaded from Hugging Face). Files of a
    local copy are checked against the pinned sizes (``verify="size"``),
    ``verify="sha256"`` hashes them, ``verify=None`` skips the check.
    """
    needed = ["bands.csv"] + shard_paths(folds)
    candidates = [dataset_path(DOMAIN, NAME, data_dir)]
    if archive is not None:
        a = Path(archive).expanduser()
        if not a.is_dir():
            raise FileNotFoundError(f"archive={a} must be a folder with the release layout "
                                    "(bands.csv, data/fold_*.parquet)")
        candidates.append(a)
    for root in candidates:
        if all((root / r).exists() for r in needed):
            for r in needed:
                _check_file(root / r, r, verify)
            return root
    if archive is not None:
        missing = [r for r in needed if not (candidates[-1] / r).exists()]
        raise FileNotFoundError(f"{candidates[-1]} lacks {missing[:4]}")
    return download_files(needed, cache_dir)


def read_bands(root):
    """The kept bands of the release (``bands.csv`` rows with ``release_index >= 0``)."""
    b = pd.read_csv(Path(root) / "bands.csv")
    return b[b["release_index"] >= 0].sort_values("release_index").reset_index(drop=True)


def decode_cubes(column, n_dates, n_bands):
    """Nested-list cube column (pyarrow) -> float16 array (rows, T, B, H, W)."""
    import pyarrow as pa

    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    n = len(arr)
    values = arr
    while pa.types.is_list(values.type) or pa.types.is_fixed_size_list(values.type) or \
            pa.types.is_large_list(values.type):
        values = values.flatten()
    flat = values.to_numpy(zero_copy_only=False)
    side = int(round(np.sqrt(flat.size / max(n * n_dates * n_bands, 1))))
    return flat.reshape(n, n_dates, n_bands, side, side)


def _pixels(cube_t, bands, clip):
    """(B, H, W) cube of one date -> clipped pixels (n, p) with every band finite."""
    X = np.asarray(cube_t, dtype=np.float64)[bands].reshape(len(bands), -1).T
    X = np.clip(X, *clip)
    return X[np.isfinite(X).all(1)]


def _process(root, shards, out_dir):
    """Windows and (window, date) tables of the shards (no cube read)."""
    import pyarrow.parquet as pq

    windows, dates = [], []
    for s, rel in enumerate(shards):
        f = pq.ParquetFile(Path(root) / rel)
        offsets = np.cumsum([0] + [f.metadata.row_group(g).num_rows
                                   for g in range(f.metadata.num_row_groups)])
        cols = WINDOW_COLUMNS + ["dates", "flight_ids", "acquired", "qc_ok", "nan_frac",
                                 "labels_date", "props_date", "labels_window", "props_window"]
        t = f.read(columns=cols).to_pydict()
        for r in range(len(t["window_id"])):
            g = int(np.searchsorted(offsets, r, side="right") - 1)
            w = {c: t[c][r] for c in WINDOW_COLUMNS}
            w.update(shard=s, row=r, row_group=g, row_in_group=r - int(offsets[g]))
            lw = np.asarray(t["labels_window"][r], dtype=np.int8)
            pw = np.asarray(t["props_window"][r], dtype=np.float64)
            w.update({f"label_{c}": int(v) for c, v in zip(CLASSES, lw, strict=True)})
            w.update({f"prop_{c}": float(v) for c, v in zip(CLASSES, pw, strict=True)})
            windows.append(w)
            for k, acq in enumerate(t["acquired"][r]):
                if not acq:
                    continue
                d = {"window_id": w["window_id"], "t": k, "date": t["dates"][r][k],
                     "flight_id": t["flight_ids"][r][k], "qc_ok": bool(t["qc_ok"][r][k]),
                     "nan_frac": float(t["nan_frac"][r][k])}
                d.update({f"label_{c}": int(v) for c, v in
                          zip(CLASSES, t["labels_date"][r][k], strict=True)})
                d.update({f"prop_{c}": float(v) for c, v in
                          zip(CLASSES, t["props_date"][r][k], strict=True)})
                dates.append(d)
    W, D = pd.DataFrame(windows), pd.DataFrame(dates)
    n_dates = len(t["dates"][0]) if windows else 0
    W.to_csv(Path(out_dir) / "windows.csv", index=False)
    D.to_csv(Path(out_dir) / "dates.csv", index=False)
    write_json(Path(out_dir) / "info.json", {
        "shards": list(shards), "n_windows": len(W), "n_dates_per_window": n_dates,
        "n_acquired": len(D), "n_qc_ok": int(D["qc_ok"].sum()) if len(D) else 0})


def _covariances(root, proc_dir, out_dir, *, estimator, bands, clip, min_pixels, qc):
    """One covariance per eligible (window, date), streamed shard by shard.

    ``covs.npy`` (M, p, p) float64 (identity where the estimate failed),
    ``means.npy``, ``n_obs.npy``, ``logdet.npy``, ``ok.npy`` and ``dates.csv``
    (the eligible rows, ``cov_index`` = their row in these arrays).
    """
    import pyarrow.parquet as pq

    proc_dir, out_dir = Path(proc_dir), Path(out_dir)
    info = json.loads((proc_dir / "info.json").read_text())
    W = pd.read_csv(proc_dir / "windows.csv")
    D = pd.read_csv(proc_dir / "dates.csv", dtype={"date": str, "flight_id": str})
    D = D[D["qc_ok"]] if qc else D
    D = D.reset_index(drop=True)
    D["cov_index"] = np.arange(len(D))
    p = len(bands)
    covs = np.lib.format.open_memmap(out_dir / "covs.npy", mode="w+", dtype=np.float64,
                                     shape=(len(D), p, p))
    means, n_obs = np.zeros((len(D), p)), np.zeros(len(D), np.int64)
    logdet, ok = np.zeros(len(D)), np.zeros(len(D), bool)
    reasons = {"n_lt_min_pixels": 0, "not_spd": 0}
    by_window = dict(list(D.groupby("window_id")))
    n_bands, n_dates = int(info["n_bands"]), int(info["n_dates_per_window"])
    for s, rel in enumerate(info["shards"]):
        f = pq.ParquetFile(Path(root) / rel)
        rows = W[W["shard"] == s]
        for g in sorted(rows["row_group"].unique()):
            tab = f.read_row_group(int(g), columns=["window_id", "cube"])
            cubes = decode_cubes(tab.column("cube"), n_dates, n_bands)
            for r, wid in enumerate(tab.column("window_id").to_pylist()):
                for row in by_window.get(wid, pd.DataFrame()).itertuples(index=False):
                    c = int(row.cov_index)
                    X = _pixels(cubes[r, int(row.t)], bands, clip)
                    n_obs[c] = len(X)
                    covs[c] = np.eye(p)
                    if len(X) < min_pixels:
                        reasons["n_lt_min_pixels"] += 1
                        continue
                    try:
                        C = estimate_covariance(X, estimator)
                    except (InsufficientSamplesError, NotSPDError):
                        reasons["not_spd"] += 1
                        continue
                    sign, ld = np.linalg.slogdet(C)
                    covs[c], means[c], logdet[c], ok[c] = C, X.mean(0), ld, sign > 0
    covs.flush()
    del covs
    np.save(out_dir / "means.npy", means)
    np.save(out_dir / "n_obs.npy", n_obs)
    np.save(out_dir / "logdet.npy", logdet)
    np.save(out_dir / "ok.npy", ok)
    D.to_csv(out_dir / "dates.csv", index=False)
    W.to_csv(out_dir / "windows.csv", index=False)
    info.update(p=p, n_eligible=len(D), n_covariances=int(ok.sum()), excluded=reasons,
                qc=bool(qc), clip=list(clip), min_pixels=int(min_pixels))
    write_json(out_dir / "info.json", info)


def fetch_aviris_ca_ml(data_dir=None, cache_dir=None, keep=None, estimator="ledoit_wolf",
                       force=False, verify="size", *, folds=None, qc=True, band_step=1,
                       clip=CLIP, min_pixels=None, archive=None):
    """Covariances of the valid (window, date) pairs of AVIRIS-C California.

    Parameters
    ----------
    data_dir : path, optional
        Local copy of the release (default ``<root>/HYPERSPECTRAL/AVIRIS-CA-Multilabel``).
    archive : path, optional
        Another folder holding the release (e.g. a Hugging Face snapshot), read in place.
    folds : iterable of int, optional
        Folds to load (default all five); only their shards are read or downloaded.
    qc : bool
        Keep only the dates with ``qc_ok`` (default) or every acquired date.
    band_step : int
        Keep one band in ``band_step`` of the 155 kept bands.
    clip : (float, float)
        Reflectance range the pixels are clipped to before estimation.
    min_pixels : int, optional
        Fewest valid pixels of a date (default and minimum ``2p``).
    estimator, cache_dir, keep, force
        See :func:`spdnet_datasets.core.fetch.run_pipeline`; default Ledoit-Wolf.
    verify : {"size", "sha256", None}
        Check of the local files against the pinned revision.

    Returns
    -------
    Bunch with ``covariances`` (M, p, p) memory-mapped, unscaled (identity where
    ``ok`` is False), ``means``, ``n_obs``, ``logdet``, ``ok``, ``dates`` (one
    row per eligible window-date, ``cov_index``), ``windows`` (one row per
    window), ``bands`` (kept band indices), ``wavelengths`` (nm), ``info``,
    ``root`` and the stage paths.
    """
    if band_step < 1:
        raise ValueError("band_step must be >= 1")
    root = locate_release(data_dir, archive=archive, cache_dir=cache_dir, folds=folds,
                          verify=verify)
    band_table = read_bands(root)
    bands = np.arange(len(band_table))[::band_step]
    p = len(bands)
    min_pixels = 2 * p if min_pixels is None else int(min_pixels)
    if min_pixels < 2 * p:
        raise ValueError(f"min_pixels={min_pixels} < 2p = {2 * p}")
    shards = shard_paths(folds)
    params = {"source": str(Path(root).resolve()), "shards": shards, "version": 1}

    def process(_, out_dir):
        _process(root, shards, out_dir)
        info_path = Path(out_dir) / "info.json"
        info = json.loads(info_path.read_text())
        info["n_bands"] = len(band_table)
        write_json(info_path, info)

    res = run_pipeline(
        "aviris_ca_ml", params=params, process_fn=process,
        covariance_fn=lambda pr, o: _covariances(root, pr, o, estimator=estimator, bands=bands,
                                                 clip=tuple(clip), min_pixels=min_pixels,
                                                 qc=qc),
        covariance_params={"estimator": estimator, "band_step": band_step,
                           "clip": list(clip), "min_pixels": min_pixels, "qc": bool(qc)},
        cache_dir=cache_dir, keep=keep, force=force)
    cdir = Path(res.covariance_dir)
    return Bunch(
        covariances=np.load(cdir / "covs.npy", mmap_mode="r"),
        means=np.load(cdir / "means.npy"), n_obs=np.load(cdir / "n_obs.npy"),
        logdet=np.load(cdir / "logdet.npy"), ok=np.load(cdir / "ok.npy"),
        dates=pd.read_csv(cdir / "dates.csv", dtype={"date": str, "flight_id": str}),
        windows=pd.read_csv(cdir / "windows.csv"),
        info=json.loads((cdir / "info.json").read_text()), bands=bands,
        wavelengths=band_table["wavelength_nm"].to_numpy(float)[bands], root=root,
        shards=shards, covariance_path=cdir, processed_path=res.processed_dir)


def fold_splits(fold, available=range(N_FOLDS)):
    """``fold<k>`` splits (test = k, val = k - 1 mod 5, train = the others) and the
    ``recommended`` = ``train_val_test`` = ``fold4`` alias, as index arrays."""
    fold = np.asarray(fold)
    loaded = sorted(set(np.unique(fold).tolist()) & set(available))
    out = {}
    for k in loaded:
        val_k = (k - 1) % N_FOLDS
        test = np.flatnonzero(fold == k)
        val = np.flatnonzero(fold == val_k)
        train = np.flatnonzero((fold != k) & (fold != val_k))
        if len(train):
            out[f"fold{k}"] = {"train": train, "val": val, "test": test}
    if "fold4" in out:
        out["recommended"] = out["train_val_test"] = out["fold4"]
    return out


@DatasetManager.register_dataset("aviris_ca_ml")
class AVIRISCAMultilabelDataset(HSIDataset):
    """AVIRIS-C California hyperspectral time series, multilabel land cover (CDL, 13 classes).

    Parameters
    ----------
    data_dir : path, optional
        Local copy of the release (see :func:`locate_release`); without one,
        the files are downloaded from Hugging Face (token, private repository).
    mode : {"date", "series"}
        ``"date"`` (default): one ``p x p`` covariance per valid (window, date);
        ``"series"``: one sample per window, ``x = (tokens (14, p, p), mask
        (14,)[, mean (14, p)][, raw (14, 32, 32, p)])``.
    task : {"multilabel", "regression", "fine"}
        13-dim 0/1 float32 vector (default), the 13 CDL proportions, or the
        dominant class.
    label_level : {"date", "window"}, optional
        Labels of the acquisition year (default with ``mode="date"``) or of the
        window (default and only choice with ``mode="series"``).
    split : str
        Split fitting the normalisation: ``"recommended"`` (= ``"fold4"`` =
        ``"train_val_test"``), ``"fold<k>"`` or ``"random"`` (sample-level, leaky).
    subset : {"train", "val", "test"}, optional
        Restrict the dataset to one part of ``split``.
    min_dates : int
        ``mode="series"``: fewest valid dates of a window (default 1).
    folds, qc, band_step, clip, min_pixels, archive, cache_dir, keep, force, verify
        See :func:`fetch_aviris_ca_ml`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options (``return_mean``,
        ``return_raw``, ``estimator`` (default ``"ledoit_wolf"``), ``classes``,
        ``balance``, ``normalize``...).

    With ``normalize=True`` the covariances are multiplied by the geometric scale
    ``s`` of the training covariances (``s = exp(-mean(logdet) / p)``), the mean
    and the raw pixels by ``sqrt(s)``. ``groups`` = 10 km block; ``meta`` has
    ``window_id``, ``fold``, ``zone``, ``lat``, ``lon`` (and ``date``,
    ``flight_id``, ``t`` in date mode).
    """

    def __init__(self, data_dir=None, *, mode="date", task="multilabel", label_level=None,
                 split="recommended", subset=None, folds=None, qc=True, band_step=1,
                 clip=CLIP, min_pixels=None, min_dates=1, cache_dir=None, keep=None,
                 force=False, verify="size", archive=None, estimator="ledoit_wolf",
                 **kwargs):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        label_level = label_level or ("date" if mode == "date" else "window")
        if label_level not in LABEL_LEVELS:
            raise ValueError(f"label_level must be one of {LABEL_LEVELS}")
        if mode == "series" and label_level != "window":
            raise ValueError("mode='series' has window labels only (label_level='window')")
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(data_dir=str(existing_dir(root, cache_dir)), task=task,
                         estimator=estimator, **kwargs)
        b = fetch_aviris_ca_ml(root, cache_dir, keep, self.estimator, force, verify,
                               folds=folds, qc=qc, band_step=band_step, clip=clip,
                               min_pixels=min_pixels, archive=archive)
        self.mode, self.label_level, self.min_dates = mode, label_level, min_dates
        self.fetched = b
        self.data_dir = Path(b.root)
        self.bands, self.wavelengths = b.bands, b.wavelengths
        W = b.windows.set_index("window_id", drop=False)
        D = b.dates[b.ok].reset_index(drop=True)
        if mode == "date":
            table = D.join(W[["fold", "zone", "block10km", "lat", "lon"]], on="window_id")
            src = table if label_level == "date" else W.loc[table["window_id"]]
            self._cov_index = table["cov_index"].to_numpy(np.int64)
        else:
            S = int(b.info["n_dates_per_window"])
            grid = np.full((len(W), S), -1, np.int64)
            pos = {w: i for i, w in enumerate(W.index)}
            grid[D["window_id"].map(pos).to_numpy(), D["t"].to_numpy()] = D["cov_index"]
            keep_w = (grid >= 0).sum(1) >= min_dates
            table = W[keep_w].reset_index(drop=True)
            src = table
            self._series_index = grid[keep_w]
        Y = src[[f"label_{c}" for c in CLASSES]].to_numpy(np.float32)
        P = src[[f"prop_{c}" for c in CLASSES]].to_numpy(np.float64)
        bunch = Bunch(
            covariances=None,  # scale fitted below from the stored log-determinants
            means=None, labels=np.asarray(CLASSES)[P.argmax(1)], targets=P,
            target_names=[f"prop_{c}" for c in CLASSES], multilabels=Y,
            label_names=list(CLASSES), groups=table["block10km"].to_numpy(np.int64),
            meta=table.reset_index(drop=True), info=b.info,
            splits=fold_splits(table["fold"].to_numpy()),
            processed_path=b.processed_path)
        self._setup(bunch, split, None)
        self.scale = self._fit_scale() if self.normalize else 1.0
        if subset is not None:
            self._restrict(subset)
        self.window_ids = self.meta["window_id"].to_numpy(np.int64)
        if self.verbose:
            print(f"AVIRIS-C CA multilabel ({mode}, p={len(self.bands)}): {len(self)} "
                  f"samples, {len(np.unique(self.window_ids))} windows, scale {self.scale:.3e}")

    # -- helpers -----------------------------------------------------------------------
    def _rows(self, part_index):
        """Covariance rows of the samples ``part_index`` (all valid dates in series mode)."""
        k = self._index[np.asarray(part_index, dtype=np.int64)]
        if self.mode == "date":
            return self._cov_index[k]
        rows = self._series_index[k].ravel()
        return rows[rows >= 0]

    def _fit_scale(self):
        train = self.splits[self.split]["train"]
        rows = self._rows(train)
        if not len(rows):
            warnings.warn("Empty training part: covariances are not normalised", UserWarning,
                          stacklevel=3)
            return 1.0
        return float(np.exp(-np.mean(self.fetched.logdet[rows]) / len(self.bands)))

    def _restrict(self, subset):
        parts = self.splits[self.split]
        if subset not in parts:
            raise ValueError(f"subset must be one of {sorted(parts)}")
        sel = np.asarray(parts[subset])
        self._index, self.labels = self._index[sel], self.labels[sel]
        self.groups = self.groups[sel]
        self.meta = self.meta.iloc[sel].reset_index(drop=True)
        self.samples = [self.samples[i] for i in sel]
        self.splits, self.subset = {}, subset

    def _cube(self, k):
        """float16 cube (T, B, H, W) of the window of sample row ``k``."""
        import pyarrow.parquet as pq

        if not hasattr(self, "_window_rows"):
            self._window_rows = self.fetched.windows.set_index("window_id")
        w = self._window_rows.loc[int(self.meta_all["window_id"].iloc[k])]
        key = (int(w["shard"]), int(w["row_group"]))
        if getattr(self, "_group_key", None) != key:
            f = pq.ParquetFile(Path(self.fetched.root) / self.fetched.shards[key[0]])
            col = f.read_row_group(key[1], columns=["cube"]).column("cube")
            self._group = decode_cubes(col, int(self.fetched.info["n_dates_per_window"]),
                                       int(self.fetched.info["n_bands"]))
            self._group_key = key
        return self._group[int(w["row_in_group"])]

    def _raw_date(self, cube, t):
        """Clipped reflectance (H, W, p) float32 of date ``t`` (NaN pixels kept)."""
        x = np.asarray(cube[t], dtype=np.float32)[self.bands]
        return np.moveaxis(np.clip(x, *self.fetched.info["clip"]), 0, -1)

    def _load_raw(self, k):
        cube = self._cube(k)
        if self.mode == "date":
            return self._raw_date(cube, int(self.meta_all["t"].iloc[k]))
        return np.stack([self._raw_date(cube, t) for t in range(cube.shape[0])])

    # -- samples -----------------------------------------------------------------------
    def __getitem__(self, i):
        k = int(self._index[i])
        root = np.sqrt(self.scale)
        covs, means = self.fetched.covariances, self.fetched.means
        raw = None
        if self.return_raw:
            raw = torch.from_numpy((self._load_raw(k) * root).astype(np.float32))
        if self.mode == "date":
            c = int(self._cov_index[k])
            cov = torch.from_numpy(np.asarray(covs[c], dtype=np.float64) * self.scale)
            mean = torch.from_numpy(means[c] * root) if self.return_mean else None
            x = self._pack(cov, mean, raw)
        else:
            rows = self._series_index[k]
            mask = rows >= 0
            p = len(self.bands)
            tokens = np.tile(np.eye(p), (len(rows), 1, 1))
            tokens[mask] = np.asarray(covs[rows[mask]], dtype=np.float64) * self.scale
            x = (torch.from_numpy(tokens), torch.from_numpy(mask.copy()))
            if self.return_mean:
                m = np.zeros((len(rows), p))
                m[mask] = means[rows[mask]] * root
                x += (torch.from_numpy(m),)
            if raw is not None:
                x += (raw,)
        if self.transform is not None:
            x = self.transform(x)
        return x, self._target(i)


__all__ = ["CLASSES", "HF_REPO", "HF_REVISION", "AVIRISCAMultilabelDataset",
           "download_files", "fetch_aviris_ca_ml", "fold_splits", "locate_release"]
