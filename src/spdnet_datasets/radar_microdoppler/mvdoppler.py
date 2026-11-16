"""MVDoppler: gait patterns seen by two mmWave radars (two Doppler views).

Source
------
S. Hor, S. Yang, J. Choi, A. Arbabian, "MVDoppler: Unleashing the Power of
Multi-View Doppler for MicroMotion-Based Gait Classification", NeurIPS 2023
Datasets and Benchmarks, https://github.com/soheilhr/MVDoppler.
54 836 snapshots of 1.28 s (256 frames, step 0.64 s, 50 % overlap) cut from
4 210 recordings of 13 subjects walking with 4 patterns (normal, phone call,
hands in pockets, texting). Each ``Data/<pattern>_<ID>.h5`` holds
``radar_dat`` of shape ``(128 Doppler bins, 256 frames, 2)``.

The last axis holds the **two radar views** (x and y), not I/Q components: all
values are positive amplitudes, the file attributes give one SNR per view
(``snr_db_x``, ``snr_db_y``), ``meta_data.json`` sets
``merge_radar_spectrograms: true``, and the two channels are almost
uncorrelated with different Doppler peaks. The former precompute took
``sqrt(c0**2 + c1**2)`` as an I/Q modulus, which mixes the two views.

Construction (covariance "C2" of the data atlas, 64 x 64)
---------------------------------------------------------
Per view, log amplitude ``log(max(a, 1e-12))``, the 128 Doppler bins averaged
by blocks of 4 (32 bins); the two views are stacked into ``p = 64`` channels
(cross-view correlations included); one covariance over the 256 frames
(``n = 256 = 4p``). "C2" is the name of this joint x-y covariance in the
atlas sheet ``mvdoppler`` (C1 is the 32 x 32 single-view one); the paper
itself defines no "C2" protocol.

Splits and leakage
------------------
Official 4-fold cross-validation of ``design_table.csv`` (column ``fold``),
disjoint in subjects: fold 0 = subjects 4, 6, 11; fold 1 = 2, 8, 10;
fold 2 = 1, 7, 9; fold 3 = 0, 3, 5, 12. ``splits["fold<k>"]``: test = fold
``k``; validation = the ``val_set`` recordings of the other folds (the flag is
constant per recording, so no overlapping snapshot crosses train/val);
train = the rest. ``"official"`` is fold ``fold``. Groups are subjects.
Snapshots of one recording overlap by 50 %: never split them at random.
Files without the HDF5 signature and ``.h5`` files absent from
``design_table.csv`` are skipped and counted.

Download
--------
Access: **manual** for the snapshots, direct for the labels. The data are on the
authors' Google Drive (:data:`DRIVE_FOLDER`, linked from
https://github.com/soheilhr/MVDoppler and https://mvdoppler.github.io/):

- ``Labels_and_metadata/design_table.csv`` (11.5 MB) and ``meta_data.json``
  (1 kB) are downloaded directly; their sha256 (:data:`FILES`) were computed on
  the local copy and match the Drive files;
- ``Data/snapshots.zip`` (about 25 GB, Drive id :data:`SNAPSHOTS_ID`, no
  published checksum) holds the ``<pattern>_<ID>.h5`` snapshots. Drive often
  refuses large downloads (quota): download it from the browser, then pass
  ``archive=``. With ``gdown=True`` the optional ``gdown`` package
  (``pip install spdnet-datasets[gdrive]``) tries the download instead.

Licence: ambiguous, the "License agreement" document of the Drive states both
CC BY 4.0 and CC BY-NC-ND (the code is Apache-2.0). Resolution order:

1. ``data_dir`` (default ``<data root>/RADAR_DOPPLER/MVDoppler``) exists: read in
   place;
2. ``archive=`` given: ``snapshots.zip`` (extracted into
   ``<cache_dir>/mvdoppler/downloads/``, kept) or a folder of ``.h5`` files;
3. ``gdown=True``: ``snapshots.zip`` is downloaded with ``gdown``, then as 2;
4. otherwise the Drive folder is opened and
   :class:`~spdnet_datasets.core.access.ManualDownloadRequired` explains what to
   download.

Example: ``fetch_mvdoppler(archive="~/Downloads/snapshots.zip")``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..audio._common import downloads_dir
from ..core.config import dataset_path
from ..core.fetch import check_local, download, run_pipeline
from ..manager import DatasetManager
from ._common import (
    RadarCovarianceDataset,
    covariance_stage,
    finish_processed,
    load_bunch,
)

DOMAIN, NAME = "RADAR_DOPPLER", "MVDoppler"
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1Mde8sfxKl8L0OwG4UVQR7IE5Tg-bSosR"
SNAPSHOTS_ID = "1slKM0AzsXBTFjRtpjzFea9lV27UDd-mI"  # Data/snapshots.zip
_DRIVE = "https://drive.usercontent.google.com/download?export=download&id="
LABEL_FILES = {  # file name -> Drive id
    "design_table.csv": "1oQBKBxftLbnSSLFvFLnR48mhqAfWH62-",
    "meta_data.json": "1wygIDUoebcdFh-8ewOjYWuM9alSkxggR",
}
FILES = {  # url -> sha256 (local copy, identical to the Drive files on 2026-11-15)
    _DRIVE + LABEL_FILES["design_table.csv"]:
        "a9c5ca209eedd97a0e61ffaecc15bc864377e846c368a4a213fb6dd24992a14f",
    _DRIVE + LABEL_FILES["meta_data.json"]:
        "f6dc2d109478f039f46f29dc9f74b67a0974b5c5b2240d03a757fe59dea1ffbb",
}
CLASSES = ["normal", "phone_call", "pockets", "texting"]
N_FRAMES, N_BINS, POOL = 256, 128, 4
P = 2 * N_BINS // POOL  # 64
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _readable(path):
    try:
        with open(path, "rb") as f:
            return f.read(8) == _HDF5_SIGNATURE
    except OSError:
        return False


def _gdown_snapshots(dest):
    """Download ``snapshots.zip`` with the optional ``gdown`` package."""
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("gdown=True needs the 'gdown' package: "
                          "pip install spdnet-datasets[gdrive]") from exc
    out = Path(dest) / "snapshots.zip"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".part")
        message = ("gdown could not download snapshots.zip (Drive quota?): download it from "
                   f"{DRIVE_FOLDER} and pass archive=")
        try:
            gdown.download(id=SNAPSHOTS_ID, output=str(part), quiet=False, resume=True)
        except Exception as exc:  # gdown raises its own errors (quota, permission...)
            raise OSError(message) from exc
        if not part.is_file():
            raise OSError(message)
        part.rename(out)
    return out


def download_mvdoppler(cache_dir=None, archive=None, gdown=False):
    """Assemble ``<cache_dir>/mvdoppler/downloads/MVDoppler/`` (see "Download").

    ``Labels_and_metadata/`` is downloaded and checked; ``Data`` is a link to the
    folder holding the ``.h5`` snapshots of ``archive`` (extracted ``snapshots.zip``
    or a folder). Returns the ``MVDoppler`` folder.
    """
    from ..core.access import manual_source

    dest = downloads_dir(cache_dir, "mvdoppler")
    root = dest / NAME
    if not (root / "Data").is_dir():
        if archive is None and gdown:
            archive = _gdown_snapshots(dest)
        out = manual_source(NAME, page_url=DRIVE_FOLDER, dest_dir=dest / "snapshots",
                            archive=archive, expected="Data/snapshots.zip (about 25 GB)")
        found = next(Path(out).rglob("*.h5"), None)
        if found is None:
            raise FileNotFoundError(f"No .h5 snapshot in {out}")
        root.mkdir(parents=True, exist_ok=True)
        (root / "Data").symlink_to(found.parent.resolve(), target_is_directory=True)
    for url, sha256 in FILES.items():
        name = next(n for n, i in LABEL_FILES.items() if url.endswith(i))
        download(url, root / "Labels_and_metadata" / name, checksum=sha256)
    return root


def index_mvdoppler(data_dir=None, limit=None):
    """Rows of ``design_table.csv`` whose ``.h5`` file is readable.

    Parameters
    ----------
    limit : int, optional
        Keep at most ``limit`` snapshots per (fold, pattern), the first ones in
        table order (for quick tests).

    Returns
    -------
    (meta, info) : DataFrame of kept snapshots, dict of counts.
    """
    root = dataset_path(DOMAIN, NAME, data_dir)
    design = pd.read_csv(root / "Labels_and_metadata" / "design_table.csv", index_col=0)
    readable = np.array([_readable(root / "Data" / f"{n}.h5") for n in design.fname])
    on_disk = {p.stem for p in (root / "Data").glob("*.h5")}
    info = {"design_rows": len(design), "unreadable": int((~readable).sum()),
            "unreadable_files": design.fname[~readable].tolist(),
            "h5_not_in_design_table": len(on_disk - set(design.fname))}
    meta = design[readable]
    if limit is not None:
        meta = meta.groupby(["fold", "pattern"], sort=False).head(limit)
    meta = meta[["fname", "exp_fname", "pattern", "subject", "fold", "val_set"]].copy()
    meta["val_set"] = meta.val_set.astype(str).str.upper().eq("TRUE")
    meta["label"] = meta.pattern.map(CLASSES.index)
    meta["group"] = meta.subject
    return meta.reset_index(drop=True), info


def snapshot_features(radar_dat):
    """``(128, 256, 2)`` amplitudes -> ``(256, 64)`` pooled log amplitudes (view x, then y)."""
    logamp = np.log(np.clip(np.asarray(radar_dat, dtype=np.float64), 1e-12, None))
    views = [logamp[:, :, v].reshape(N_BINS // POOL, POOL, -1).mean(1) for v in (0, 1)]
    return np.concatenate(views, axis=0).T


def official_folds(meta):
    """``{"fold<k>": {"train", "val", "test"}}`` from the ``fold`` and ``val_set`` columns."""
    fold = meta.fold.to_numpy()
    val = meta.val_set.to_numpy(dtype=bool)
    return {f"fold{k}": {"train": np.flatnonzero((fold != k) & ~val),
                         "val": np.flatnonzero((fold != k) & val),
                         "test": np.flatnonzero(fold == k)}
            for k in range(4)}


def fetch_mvdoppler(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                    verify="size", *, fold=0, limit=None, archive=None, gdown=False):
    """Joint two-view 64 x 64 covariances ("C2") of the MVDoppler snapshots.

    Parameters
    ----------
    data_dir : path, optional
        Read-only copy of ``MVDoppler`` (default ``<data root>/RADAR_DOPPLER/MVDoppler``).
    cache_dir, keep, force
        See :func:`spdnet_datasets.core.run_pipeline`.
    estimator : str
        Covariance estimator (``n/p = 4``: ``ledoit_wolf`` is better conditioned).
    verify : {"size", "md5", None}
        Check of the local copy against ``MANIFEST_NAS.tsv``.
    fold : int
        Official fold used as test set by ``splits["official"]``.
    limit : int, optional
        At most ``limit`` snapshots per (fold, pattern) (quick tests).
    archive : path, optional
        ``snapshots.zip`` or a folder of ``.h5`` files, used when ``data_dir``
        does not exist (see "Download" in the module docstring).
    gdown : bool
        Without ``archive``, try to download ``snapshots.zip`` with ``gdown``.

    Returns
    -------
    Bunch with ``covariances`` (B, 64, 64) unscaled, ``means``, ``labels``,
    ``class_names``, ``groups`` (subject), ``meta``, ``info``, ``splits``,
    ``covariance_path``, ``processed_path`` and ``source_dir``.
    """
    import h5py

    root = dataset_path(DOMAIN, NAME, data_dir)
    if root.is_dir():
        check_local(root, verify=verify, required=("Labels_and_metadata/design_table.csv", "Data"),
                    strict=False)
    else:
        root = download_mvdoppler(cache_dir, archive, gdown)

    def process_fn(_source, out_dir):
        meta, info = index_mvdoppler(root, limit)
        offsets = np.arange(len(meta) + 1, dtype=np.int64) * N_FRAMES
        raw = np.lib.format.open_memmap(out_dir / "raw.npy", mode="w+", dtype=np.float32,
                                        shape=(int(offsets[-1]), P))
        for i, name in enumerate(meta.fname):
            with h5py.File(root / "Data" / f"{name}.h5", "r") as f:
                raw[offsets[i]:offsets[i + 1]] = snapshot_features(f["radar_dat"][...])
        raw.flush()
        del raw
        np.save(out_dir / "offsets.npy", offsets)
        finish_processed(out_dir, meta, info)

    params = {"limit": limit, "source": str(root.resolve()), "version": 1}
    paths = run_pipeline("mvdoppler", params=params, process_fn=process_fn,
                         covariance_fn=covariance_stage(estimator),
                         covariance_params={"estimator": estimator}, cache_dir=cache_dir,
                         keep=keep, force=force)
    bunch = load_bunch(paths, CLASSES)
    bunch.source_dir = root
    bunch.splits = official_folds(bunch.meta)
    bunch.splits["official"] = bunch.splits[f"fold{fold}"]
    return bunch


@DatasetManager.register_dataset("mvdoppler")
class MVDopplerDataset(RadarCovarianceDataset):
    """MVDoppler gait patterns (4 classes), joint x-y 64 x 64 covariances.

    ``dataset[i] -> (x, label)``; ``raw`` is the ``(256, 64)`` pooled log
    amplitude (view x bins then view y bins).

    Parameters
    ----------
    data_dir : path, optional
    fold, limit, cache_dir, keep, force, verify, archive, gdown
        See :func:`fetch_mvdoppler`.
    split : str
        ``"official"`` (fold ``fold``) or ``"fold<k>"``.
    subset : {None, "train", "val", "test"}
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    def __init__(self, data_dir=None, *, fold=0, limit=None, split="official", subset=None,
                 cache_dir=None, keep=None, force=False, verify="size", estimator="scm",
                 archive=None, gdown=False, **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        bunch = fetch_mvdoppler(root, cache_dir, keep, estimator, force, verify, fold=fold,
                                limit=limit, archive=archive, gdown=gdown)
        super().__init__(bunch, split=split, subset=subset, data_dir=bunch.source_dir,
                         estimator=estimator, **kwargs)
