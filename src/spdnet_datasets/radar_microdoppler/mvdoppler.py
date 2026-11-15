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

Download (manual): the data are on the authors' Google Drive, linked from
https://github.com/soheilhr/MVDoppler (no stable direct URL or checksum). Download
``Data/`` and ``Labels_and_metadata/`` and place it under ``data_dir`` (default
``<data root>/RADAR_DOPPLER/MVDoppler``).
"""

import numpy as np
import pandas as pd

from ..core.config import dataset_path
from ..core.fetch import run_pipeline
from ..manager import DatasetManager
from ._common import (
    RadarCovarianceDataset,
    check_local,
    covariance_stage,
    finish_processed,
    load_bunch,
)

DOMAIN, NAME = "RADAR_DOPPLER", "MVDoppler"
FILES = {}  # Google Drive of the authors: manual download (see the dataset page)
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
                    verify="size", *, fold=0, limit=None):
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

    Returns
    -------
    Bunch with ``covariances`` (B, 64, 64) unscaled, ``means``, ``labels``,
    ``class_names``, ``groups`` (subject), ``meta``, ``info``, ``splits``,
    ``covariance_path`` and ``processed_path``.
    """
    import h5py

    root = dataset_path(DOMAIN, NAME, data_dir)
    check_local(root, verify=verify, required=("Labels_and_metadata/design_table.csv", "Data"))

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

    params = {"limit": limit, "estimator": estimator, "version": 1}
    paths = run_pipeline("mvdoppler", params=params, process_fn=process_fn,
                         covariance_fn=covariance_stage(estimator), cache_dir=cache_dir,
                         keep=keep, force=force)
    bunch = load_bunch(paths, CLASSES)
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
    fold, limit, cache_dir, keep, force, verify
        See :func:`fetch_mvdoppler`.
    split : str
        ``"official"`` (fold ``fold``) or ``"fold<k>"``.
    subset : {None, "train", "val", "test"}
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    def __init__(self, data_dir=None, *, fold=0, limit=None, split="official", subset=None,
                 cache_dir=None, keep=None, force=False, verify="size", estimator="scm",
                 **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        bunch = fetch_mvdoppler(root, cache_dir, keep, estimator, force, verify, fold=fold,
                                limit=limit)
        super().__init__(bunch, split=split, subset=subset, data_dir=root,
                         estimator=estimator, **kwargs)
