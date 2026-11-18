"""PyTorch dataset of fMRI connectomes (SPD covariances) from nilearn datasets.

nilearn downloads the data (``nilearn.datasets.fetch_*``) into its own data
directory (``data_dir``, default ``$NILEARN_DATA`` or ``~/nilearn_data``); those
files are never deleted here. This module only caches what it derives from them,
under ``SPDNET_CACHE_DIR/nilearn_<dataset>/`` through
:func:`spdnet_datasets.core.fetch.run_pipeline`::

    processed/<hash>/   series.npy (float32, all subjects stacked), lengths.npy,
                        means.npy, metadata.csv, phenotypic.csv
    covariance/<hash>/  covariances.npy, means.npy, subject_index.npy,
                        n_excluded_short.npy, n_excluded_not_spd.npy,
                        metadata.csv, phenotypic.csv

The covariance stage adds the (effective) estimator and ``allow_small_n`` to the
preprocessing key, so changing the estimator does not extract the time series
again.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.core.config import get_cache_dir
from spdnet_datasets.core.estimators import NotSPDError
from spdnet_datasets.core.fetch import run_pipeline
from spdnet_datasets.core.normalize import geometric_scale
from spdnet_datasets.core.splits import assert_no_leakage
from spdnet_datasets.nilearn.covariances import series_covariance

PIPELINE_VERSION = 1
DATASETS = ("abide", "adhd", "development", "nki", "haxby")
# ABIDE PCP ROI derivatives: short name -> number of regions p
ABIDE_ATLASES = {"tt": 97, "ho": 111, "aal": 116, "ez": 116, "dosenbach160": 161,
                 "cc200": 200, "cc400": 392}
MASKER_ATLASES = ("msdl",)
# Haxby masks shipped with the dataset (ventral temporal cortex, face and house areas)
HAXBY_MASKS = ("vt", "face", "house")
ATLASES = {"abide": tuple(ABIDE_ATLASES), "adhd": MASKER_ATLASES,
           "development": MASKER_ATLASES, "nki": ("destrieux",), "haxby": HAXBY_MASKS}
DEFAULT_ATLAS = {"abide": "tt", "adhd": "msdl", "development": "msdl", "nki": "destrieux",
                 "haxby": "vt"}
HAXBY_CATEGORIES = ["bottle", "cat", "chair", "face", "house", "scissors", "scrambledpix",
                    "shoe"]
CLASSES = {"abide": ["control", "autism"], "adhd": ["control", "adhd"],
           "development": ["adult", "child"], "nki": ["F", "M"], "haxby": HAXBY_CATEGORIES}
HAXBY_TR = 2.5  # seconds
HAXBY_BLOCK = 9  # volumes per stimulus block (22.5 s)
HAXBY_SUBJECTS = 6
META_COLUMNS = ["subject", "site", "label", "age", "sex", "n_timepoints"]


def _plain(v):
    """JSON form of a fetcher argument (NumPy arrays and scalars included) for the cache key."""
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    return v


def _zscore(ts):
    """Z-score each region over time (ddof=1); constant regions are set to 0."""
    ts = np.asarray(ts, dtype=np.float64)
    std = ts.std(axis=0, ddof=1)
    return (ts - ts.mean(axis=0)) / np.where(std > 0, std, 1.0)


def _read_csv(path):
    """Read a cached table, keeping subject identifiers as strings (ADHD IDs start with 0)."""
    return pd.read_csv(path, dtype={"subject": str, "site": str, "Subject": str,
                                    "participant_id": str})


def _extract_timeseries(func, confounds, atlas, data_dir, t_r):
    """Region time series (T, p) of each image with a nilearn masker.

    Only ``atlas="msdl"`` (probabilistic maps, p = 39, ``NiftiMapsMasker``) is
    supported. Confounds are regressed out; standardization is done afterwards
    by :class:`NilearnConnectomes` so that every dataset is treated alike.
    """
    from nilearn import datasets
    from nilearn.maskers import NiftiMapsMasker

    maps = datasets.fetch_atlas_msdl(data_dir=data_dir, verbose=0).maps
    masker = NiftiMapsMasker(maps_img=maps, standardize=False, standardize_confounds=True,
                             t_r=t_r, verbose=0)
    confounds = confounds if confounds is not None else [None] * len(func)
    return [masker.fit_transform(f, confounds=c) for f, c in zip(func, confounds, strict=True)]


def _load_abide(atlas, n_subjects, data_dir, fetcher_kwargs):
    from nilearn import datasets

    derivative = f"rois_{atlas}"
    bunch = datasets.fetch_abide_pcp(data_dir=data_dir, n_subjects=n_subjects,
                                     derivatives=[derivative], verbose=0, **fetcher_kwargs)
    ph = bunch.phenotypic.reset_index(drop=True)
    meta = pd.DataFrame({
        "subject": ph["SUB_ID"].astype(str),
        "site": ph["SITE_ID"].astype(str),
        "label": ph["DX_GROUP"].map({1: "autism", 2: "control"}),
        "age": ph["AGE_AT_SCAN"].astype(float),
        "sex": ph["SEX"].map({1: "M", 2: "F"}),
    })
    return list(bunch[derivative]), meta, ph


def _load_adhd(atlas, n_subjects, data_dir, fetcher_kwargs, confounds):
    from nilearn import datasets

    bunch = datasets.fetch_adhd(data_dir=data_dir, n_subjects=n_subjects, verbose=0,
                                **fetcher_kwargs)
    ph = bunch.phenotypic.reset_index(drop=True)
    meta = pd.DataFrame({
        "subject": ph["Subject"].astype(str),
        # Peking_1/2/3 are cohorts of one acquisition site
        "site": ph["site"].astype(str).str.replace(r"_\d+$", "", regex=True),
        "label": ph["adhd"].map({1: "adhd", 0: "control"}),
        "age": ph["age"].astype(float),
        "sex": ph["sex"].astype(str),
    })
    series = _extract_timeseries(bunch.func, bunch.confounds if confounds else None, atlas,
                                 data_dir, bunch.get("t_r"))
    return series, meta, ph


def _load_development(atlas, n_subjects, data_dir, fetcher_kwargs, confounds):
    from nilearn import datasets

    bunch = datasets.fetch_development_fmri(data_dir=data_dir, n_subjects=n_subjects,
                                            verbose=0, **fetcher_kwargs)
    ph = bunch.phenotypic.reset_index(drop=True)
    meta = pd.DataFrame({
        "subject": ph["participant_id"].astype(str),
        "site": "development",  # single-site study
        "label": ph["Child_Adult"].astype(str),
        "age": ph["Age"].astype(float),
        "sex": ph["Gender"].astype(str),
    })
    series = _extract_timeseries(bunch.func, bunch.confounds if confounds else None, atlas,
                                 data_dir, bunch.get("t_r"))
    return series, meta, ph


def _read_surface(path):
    """Vertex time series (n_vertices, T) of a GIfTI file."""
    import nibabel as nib

    return np.asarray(nib.load(path).agg_data(), dtype=np.float64)


NKI_PHENOTYPIC = "NKI_enhanced_surface_phenotypics.csv"


def _nki_phenotypic(bunch):
    """Phenotypic table of all NKI subjects.

    nilearn (0.14.1) reads ``NKI_enhanced_surface_phenotypics.csv`` with
    ``header=1``, so its first subject (A00028185) becomes the header and is
    lost; the downloaded file is read again here with its real header.
    """
    path = Path(bunch.func_left[0]).parents[1] / NKI_PHENOTYPIC if bunch.func_left else None
    if path is None or not path.exists():
        return bunch.phenotypic
    ph = pd.read_csv(path, header=0, dtype={0: str})
    ph.columns = ["Subject", "Age", "Dominant Hand", "Sex"]
    return ph


def _load_nki(atlas, n_subjects, data_dir, fetcher_kwargs):
    """NKI enhanced resting state on fsaverage5, averaged in the Destrieux regions.

    The regions are the Destrieux labels of each hemisphere except ``Unknown``
    and ``Medial_wall`` (p = 148): the time series of a region is the mean of
    its vertices.
    """
    from nilearn import datasets

    bunch = datasets.fetch_surf_nki_enhanced(n_subjects=n_subjects, data_dir=data_dir,
                                             verbose=0, **fetcher_kwargs)
    destrieux = datasets.fetch_atlas_surf_destrieux(data_dir=data_dir, verbose=0)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in destrieux.labels]
    keep = [k for k, n in enumerate(names) if n not in ("Unknown", "Medial_wall")]
    maps = [np.asarray(destrieux.map_left), np.asarray(destrieux.map_right)]
    regions = [(h, k) for h in (0, 1) for k in keep if (maps[h] == k).any()]
    series = []
    for left, right in zip(bunch.func_left, bunch.func_right, strict=True):
        data = [_read_surface(left), _read_surface(right)]
        series.append(np.stack([data[h][maps[h] == k].mean(axis=0) for h, k in regions],
                               axis=1))
    # phenotypic rows in the order of the fetched files (folder name = subject ID)
    ids = [Path(f).parent.name for f in bunch.func_left]
    ph = _nki_phenotypic(bunch).set_index("Subject")
    missing = sorted(set(ids) - set(ph.index))
    if missing:
        raise ValueError(f"NKI subjects without phenotypic row: {missing[:5]}")
    ph = ph.loc[ids].reset_index()
    meta = pd.DataFrame({
        "subject": ph["Subject"].astype(str),
        "site": "nki",  # single-site study
        "label": ph["Sex"].astype(str).where(ph["Sex"].isin(CLASSES["nki"])),
        "age": ph["Age"].astype(float),
        "sex": ph["Sex"].astype(str),
    })
    return series, meta, ph


def _haxby_run_signals(func, target, mask):
    """Voxel signals (T, n_voxels) of one Haxby subject, detrended and z-scored per run,
    with the block labels and the runs (``chunks``) of ``labels.txt``."""
    from nilearn.maskers import NiftiMasker
    from nilearn.signal import clean

    st = pd.read_csv(target, sep=" ")
    runs = st["chunks"].to_numpy()
    X = NiftiMasker(mask_img=mask, standardize=None).fit_transform(func)
    X = clean(X, runs=runs, detrend=True, standardize="zscore_sample", t_r=HAXBY_TR)
    return X, st["labels"].astype(str).to_numpy(), runs


def _blocks(labels, runs):
    """(start, length, label, run) of each run of identical labels inside a run."""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[i - 1] or runs[i] != runs[i - 1]:
            out.append((start, i - start, labels[start], runs[start]))
            start = i
    return out


def _load_haxby(atlas, n_subjects, data_dir, fetcher_kwargs):
    """One sample per stimulus block: the voxels of the mask as observations of the
    ``HAXBY_BLOCK`` volumes of the block, an array (n_voxels, 9).

    A block lasts 9 volumes, too few to estimate a covariance between regions,
    so the variables are the volumes of the block and the observations the
    voxels of the mask (several hundred): the covariance describes how the
    spatial pattern evolves during the block. Rest periods are left out; a
    block shorter than 9 volumes is dropped.
    """
    from nilearn import datasets

    kw = dict(fetcher_kwargs)
    kw.setdefault("subjects", tuple(range(1, (n_subjects or HAXBY_SUBJECTS) + 1)))
    subjects = [kw["subjects"]] if np.isscalar(kw["subjects"]) else list(kw["subjects"])
    bunch = datasets.fetch_haxby(data_dir=data_dir, verbose=0, **kw)
    series, rows = [], []
    for subj, func, target, mask in zip(subjects, bunch.func, bunch.session_target,
                                        bunch[f"mask_{atlas}"], strict=True):
        X, labels, runs = _haxby_run_signals(func, target, mask)
        for start, length, label, run in _blocks(labels, runs):
            if label == "rest" or length < HAXBY_BLOCK:
                continue
            series.append(X[start:start + HAXBY_BLOCK].T)
            rows.append({"subject": f"subj{int(subj)}", "site": "haxby", "label": label,
                         "age": np.nan, "sex": "", "run": int(run), "block": len(rows)})
    meta = pd.DataFrame(rows)
    return series, meta, meta.copy()


def stratified_split(y, *, stratify=True, val_size=0.1, test_size=0.1, seed=0):
    """Subject-level train/val/test split, stratified on ``y`` if ``stratify``.

    Sites are mixed between train and test, so on multi-site data the score is
    optimistic. ``test_size=0.1`` follows the spdlearn fMRI example. With
    ``stratify``, the test and validation parts hold at least one subject per
    class, so small cohorts get proportionally larger parts; a cohort too small
    for that raises ``ValueError``.
    """
    from sklearn.model_selection import train_test_split

    y = np.asarray(y)
    idx = np.arange(len(y))
    k = len(np.unique(y)) if stratify else 1
    n_test = max(int(np.ceil(test_size * len(y))), k)
    rest, test = train_test_split(idx, test_size=n_test, stratify=y if stratify else None,
                                  random_state=seed)
    n_val = max(int(np.ceil(val_size * len(y))), k)
    train, val = train_test_split(rest, test_size=n_val,
                                  stratify=y[rest] if stratify else None, random_state=seed)
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def site_grouped_split(y, sites, *, stratify=True, n_folds=5, seed=0):
    """Train/val/test split without any acquisition site on two sides.

    ``test`` is the first fold of a :class:`~sklearn.model_selection.StratifiedGroupKFold`
    on sites with ``min(n_folds, n_sites)`` folds; ``val`` is the first fold of the
    same splitter on the remaining sites (empty if a single site remains).
    ``stratify=False`` (regression) groups without stratification.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    sites = np.asarray(sites)
    y = np.asarray(y) if stratify else np.zeros(len(sites), dtype=int)

    def _cut(sub, rs):
        k = min(n_folds, len(np.unique(sites[sub])))
        a, b = next(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=rs)
                    .split(sub, y[sub], sites[sub]))
        return sub[a], sub[b]

    rest, test = _cut(np.arange(len(sites)), seed)
    if len(np.unique(sites[rest])) >= 2:
        train, val = _cut(rest, seed + 1)
    else:
        train, val = rest, rest[:0]
    for a, b in ((train, test), (train, val), (val, test)):
        assert_no_leakage(sites[a], sites[b], what="site")
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


class NilearnConnectomes(BaseDataset):
    """Resting-state fMRI connectomes (one SPD covariance per subject) from nilearn.

    Parameters
    ----------
    dataset : {"abide", "adhd", "development", "nki", "haxby"}
        ``"abide"``: ABIDE PCP ROI time series (``fetch_abide_pcp``), autism vs
        control, sites as groups. ``"adhd"``: ADHD-200 sample of 40 subjects
        (``fetch_adhd``), ADHD vs control, sites as groups. ``"development"``:
        155 children and adults watching a movie (``fetch_development_fmri``),
        child vs adult, single site. ``"nki"``: NKI enhanced resting state of
        102 adults on the fsaverage5 surface (``fetch_surf_nki_enhanced``), sex
        or age, single site. ``"haxby"``: Haxby et al. (2001) visual object
        recognition, 6 subjects (``fetch_haxby``); one sample per stimulus
        block, 8 categories, runs as groups (see below).
    atlas : str, optional
        ``abide``: one of :data:`ABIDE_ATLASES` (``"tt"``, ``"ho"``, ``"aal"``,
        ``"ez"``, ``"dosenbach160"``, ``"cc200"``, ``"cc400"``, the ``rois_*``
        derivatives). ``adhd`` and ``development``: ``"msdl"``. ``nki``:
        ``"destrieux"`` (148 surface regions). ``haxby``: the mask, ``"vt"``
        (ventral temporal cortex), ``"face"`` or ``"house"``. Default:
        :data:`DEFAULT_ATLAS`.
    n_subjects : int, optional
        Passed to the nilearn fetcher (``None``: all subjects). The fetchers take
        the first rows of their phenotypic table, which for ABIDE are sorted by
        site: use ``fetcher_kwargs={"SITE_ID": [...]}`` to pick sites. For
        ``haxby``, the subjects ``1..n_subjects`` (or ``fetcher_kwargs={"subjects":
        (2,)}``).
    task : {"fine", "regression"}
        ``"fine"``: the classes of :data:`CLASSES` (diagnosis, age group, sex
        for ``nki``, object category for ``haxby``). ``"regression"``: age in
        years (float labels; not for ``haxby``).

    Notes
    -----
    ``haxby``: a stimulus block lasts 9 volumes (TR 2.5 s), too few to estimate
    a covariance between regions. Each sample is therefore the block seen as
    ``n_voxels`` observations (the voxels of the mask, detrended and z-scored
    per run) of 9 variables (the volumes of the block): a 9 x 9 covariance of
    how the spatial pattern evolves during the block. ``raw`` is that
    ``(n_voxels, 9)`` array, ``mean`` the mean signal of each volume, and
    ``metadata`` has ``run`` and ``block`` columns (``n_timepoints`` holds the
    number of voxels). Splits: ``run_grouped`` (no run on two sides, the
    within-subject protocol of the original study; recommended),
    ``subject_grouped`` (cross-subject, with at least two subjects) and
    ``stratified``.
    estimator : {"scm", "ledoit_wolf", "tyler", "student"}
        Covariance estimator. ``T >= 2p`` is enforced: subjects with fewer time
        points are excluded and counted (``n_excluded_short``).
    allow_small_n : bool
        Keep subjects with ``T < 2p``: the covariances of all subjects are then
        estimated with Ledoit-Wolf shrinkage (``estimator`` is replaced, with a
        warning). Needed for large atlases such as ABIDE CC200.
    standardize : bool
        Z-score each region time series (ddof=1) before the covariance, so the
        SCM is a correlation matrix. Also applies to ``return_raw``.
    confounds : bool
        Regress the confounds shipped with ``adhd`` and ``development`` out of the
        signals (no effect for ``abide``, whose PCP series are already cleaned).
    fetcher_kwargs : dict, optional
        Extra arguments of the nilearn fetcher, e.g. ``{"pipeline": "cpac",
        "band_pass_filtering": True, "SITE_ID": ["NYU", "UM_1"]}`` for ABIDE or
        ``{"age_group": "child"}`` for ``development``. Part of the cache key.
    keep : {"keep_processed", "keep_covariance", "keep_full"}, optional
        What stays in the cache; ``keep_covariance`` deletes the cached time
        series, so ``return_raw`` is then unavailable. The nilearn files in
        ``data_dir`` are never deleted. Default: ``keep_full``.
    return_mean, return_raw : bool
        Optional outputs: the mean of each region signal before standardization
        (p,), float64, and the time series (T, p), float32. ``T`` varies between
        subjects, so batching ``return_raw`` needs ``batch_size=1`` or a custom
        ``collate_fn``.
    split : str, optional
        Name of a split of :attr:`splits` (default ``"recommended"``); with
        ``normalize=True`` the geometric scale is fitted on its training indices.
    normalize : bool
        Apply the geometric scale ``s`` fitted on ``splits[split]["train"]``:
        covariances times ``s``, means and raw signals times ``sqrt(s)``.
    data_dir : path, optional
        nilearn data directory (default: nilearn's, ``$NILEARN_DATA`` or
        ``~/nilearn_data``).
    cache_dir : path, optional
        Default: ``$SPDNET_CACHE_DIR``.
    force : bool
        Recompute the cached time series and covariances.
    seed : int
        Random state of the splits.

    Attributes
    ----------
    metadata : DataFrame
        One row per kept subject: ``subject``, ``site``, ``label`` (class name),
        ``age``, ``sex``, ``n_timepoints``.
    phenotypic : DataFrame
        The nilearn phenotypic rows of the kept subjects (all original columns).
    labels : ndarray
        Index into :attr:`classes` (``task="fine"``) or age (``"regression"``).
    groups : ndarray of str
        Acquisition site (``abide``, ``adhd``), subject (``development``,
        ``nki``) or ``<subject>_run<k>`` (``haxby``).
    splits : dict
        ``stratified`` (sites mixed: optimistic on multi-site data),
        ``site_grouped`` (no site shared, ``abide`` and ``adhd`` with at least
        two sites) and ``recommended`` (``site_grouped`` when available, else
        ``stratified``). Each is ``{"train", "val", "test"}``.
    scale : float
        Geometric scale applied to the covariances.
    n_excluded : int
        Subjects dropped: ``n_excluded_short`` (``T < 2p``) plus
        ``n_excluded_not_spd`` (covariance not SPD, e.g. a constant region).
    subject_index : ndarray of int
        Index of each kept subject in the cached time series.
    """

    def __init__(self, dataset="abide", *, atlas=None, n_subjects=None, task="fine",
                 estimator="scm", allow_small_n=False, standardize=True, confounds=True,
                 fetcher_kwargs=None, keep=None, return_mean=False, return_raw=False,
                 split="recommended", normalize=True, data_dir=None, cache_dir=None,
                 force=False, seed=0, verbose=False):
        if dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
        atlas = atlas or DEFAULT_ATLAS[dataset]
        allowed = ATLASES[dataset]
        if atlas not in allowed:
            raise ValueError(f"atlas for {dataset!r} must be one of {allowed}, got {atlas!r}")
        if task not in ("fine", "regression"):
            raise ValueError(f"task must be 'fine' or 'regression', got {task!r}")
        if dataset == "haxby" and task == "regression":
            raise ValueError("haxby has no age: task must be 'fine'")
        if return_raw and keep == "keep_covariance":
            raise ValueError("return_raw=True needs the cached time series: use "
                             "keep='keep_processed' or 'keep_full', not 'keep_covariance'.")
        if allow_small_n:
            warnings.warn(
                f"allow_small_n=True: the n >= 2p rule is not enforced and every covariance "
                f"is estimated with Ledoit-Wolf shrinkage (estimator={estimator!r} ignored).",
                UserWarning, stacklevel=2)
            estimator = "ledoit_wolf"
        cache_dir = get_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(cache_dir, seed=seed, verbose=verbose, return_mean=return_mean,
                         return_raw=return_raw, estimator=estimator, task=task,
                         normalize=normalize)
        self.dataset, self.atlas, self.n_subjects = dataset, atlas, n_subjects
        self.allow_small_n, self.standardize, self.confounds = allow_small_n, standardize, confounds
        self.fetcher_kwargs = dict(fetcher_kwargs or {})
        self.nilearn_dir = None if data_dir is None else str(Path(data_dir).expanduser())
        self.keep, self.force, self.split = keep, force, split
        self._load_data()

    # ------------------------------------------------------------ pipeline
    def _params(self):
        return {"dataset": self.dataset, "atlas": self.atlas, "n_subjects": self.n_subjects,
                "standardize": self.standardize,
                "confounds": self.confounds if self.dataset in ("adhd", "development") else None,
                "fetcher_kwargs": _plain(self.fetcher_kwargs),
                "version": PIPELINE_VERSION}

    def _process(self, _source, out):
        if self.dataset in ("abide", "nki", "haxby"):
            loader = {"abide": _load_abide, "nki": _load_nki, "haxby": _load_haxby}[self.dataset]
            series, meta, ph = loader(self.atlas, self.n_subjects, self.nilearn_dir,
                                      self.fetcher_kwargs)
        else:
            loader = _load_adhd if self.dataset == "adhd" else _load_development
            series, meta, ph = loader(self.atlas, self.n_subjects, self.nilearn_dir,
                                      self.fetcher_kwargs, self.confounds)
        key = ["subject", "run", "block"] if self.dataset == "haxby" else ["subject"]
        dup = meta["subject"][meta.duplicated(key)]
        if len(dup):
            raise ValueError(f"Duplicate subjects in the nilearn phenotypic table: "
                             f"{sorted(set(dup))[:5]}")
        if meta["label"].isna().any():
            raise ValueError("Unexpected diagnosis code in the phenotypic table")
        series = [np.asarray(ts, dtype=np.float64) for ts in series]
        meta["n_timepoints"] = [len(ts) for ts in series]
        np.save(out / "means.npy", np.stack([ts.mean(axis=0) for ts in series]))
        if self.standardize:
            series = [_zscore(ts) for ts in series]
        np.save(out / "series.npy", np.concatenate(series).astype(np.float32))
        np.save(out / "lengths.npy", meta["n_timepoints"].to_numpy(np.int64))
        meta.to_csv(out / "metadata.csv", index=False)
        ph.to_csv(out / "phenotypic.csv", index=False)

    def _covariances(self, processed, out):
        """One covariance per subject; subjects with ``T < 2p`` or a non-SPD estimate are dropped."""
        series = self._split_series(np.load(processed / "series.npy", mmap_mode="r"),
                                    np.load(processed / "lengths.npy"))
        covs, kept, n_short, n_not_spd = [], [], 0, 0
        for i, ts in enumerate(series):
            if not self.allow_small_n and len(ts) < 2 * ts.shape[1]:
                n_short += 1
                continue
            try:
                covs.append(series_covariance(np.asarray(ts, dtype=np.float64), self.estimator,
                                              self.allow_small_n))
            except NotSPDError:
                n_not_spd += 1
                continue
            kept.append(i)
        if not kept:
            raise ValueError(f"No subject left: {n_short} with T < 2p, {n_not_spd} not SPD. "
                             "Use a smaller atlas or allow_small_n=True.")
        kept = np.asarray(kept, dtype=np.int64)
        np.save(out / "covariances.npy", np.stack(covs))
        np.save(out / "means.npy", np.load(processed / "means.npy")[kept])
        np.save(out / "subject_index.npy", kept)
        np.save(out / "n_excluded_short.npy", n_short)
        np.save(out / "n_excluded_not_spd.npy", n_not_spd)
        for name in ("metadata.csv", "phenotypic.csv"):
            _read_csv(processed / name).iloc[kept].to_csv(out / name, index=False)

    @staticmethod
    def _split_series(stacked, lengths):
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        return [stacked[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]

    def _load_data(self):
        paths = run_pipeline(f"nilearn_{self.dataset}", params=self._params(),
                             process_fn=self._process, covariance_fn=self._covariances,
                             cache_dir=self.data_dir, keep=self.keep, force=self.force,
                             covariance_params={"estimator": self.estimator,
                                                "allow_small_n": self.allow_small_n})
        self.paths = paths
        d = paths.covariance_dir
        self.covs = np.load(d / "covariances.npy")
        self.means = np.load(d / "means.npy")
        self.subject_index = np.load(d / "subject_index.npy")
        self.n_excluded_short = int(np.load(d / "n_excluded_short.npy"))
        self.n_excluded_not_spd = int(np.load(d / "n_excluded_not_spd.npy"))
        self.n_excluded = self.n_excluded_short + self.n_excluded_not_spd
        self.metadata = _read_csv(d / "metadata.csv")
        self.phenotypic = _read_csv(d / "phenotypic.csv")
        key = ["subject", "run", "block"] if self.dataset == "haxby" else ["subject"]
        if self.metadata.duplicated(key).any():
            raise AssertionError("a subject appears twice")

        if self.task == "regression":
            self.classes, self.class_to_idx = [], {}
            self.labels = self.metadata["age"].to_numpy(np.float32)
            self.samples = [{"class_name": "age"} for _ in self.labels]
            stratify = False
        else:
            self.classes = CLASSES[self.dataset]
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.labels = self.metadata["label"].map(self.class_to_idx).to_numpy(np.int64)
            self.samples = [{"class_name": self.classes[k]} for k in self.labels]
            stratify = True
        self.sites = self.metadata["site"].to_numpy()
        multi_site = self.dataset in ("abide", "adhd")
        subjects = self.metadata["subject"].to_numpy()
        if self.dataset == "haxby":
            self.groups = (self.metadata["subject"] + "_run"
                           + self.metadata["run"].astype(str)).to_numpy()
        else:
            self.groups = self.sites if multi_site else subjects

        self.series = None
        if self.return_raw:
            if paths.processed_dir is None:
                raise ValueError("The cached time series were deleted (keep='keep_covariance');"
                                 " rebuild with force=True and another keep to use return_raw.")
            p = paths.processed_dir
            self.series = self._split_series(np.load(p / "series.npy", mmap_mode="r"),
                                             np.load(p / "lengths.npy"))

        # Each split is built on its own: an infeasible one (too few subjects
        # per class or per site) is left out and does not block the others.
        builders = {"stratified": lambda: stratified_split(self.labels, stratify=stratify,
                                                           seed=self.seed)}
        if multi_site and len(np.unique(self.sites)) >= 2:
            builders["site_grouped"] = lambda: site_grouped_split(
                self.labels, self.sites, stratify=stratify, seed=self.seed)
        if self.dataset == "haxby":  # same splitter, with runs or subjects as groups
            builders["run_grouped"] = lambda: site_grouped_split(
                self.labels, self.groups, stratify=stratify, seed=self.seed)
            if len(np.unique(subjects)) >= 2:
                builders["subject_grouped"] = lambda: site_grouped_split(
                    self.labels, subjects, stratify=stratify, seed=self.seed)
        self.splits, self.split_errors = {}, {}
        for name, build in builders.items():
            try:
                self.splits[name] = build()
            except ValueError as exc:
                self.split_errors[name] = str(exc)
                warnings.warn(f"split {name!r} cannot be built on {len(self.labels)} "
                              f"subjects and is left out: {exc}", UserWarning, stacklevel=3)
        for name in ("run_grouped", "site_grouped", "stratified"):
            if name in self.splits:
                self.splits["recommended"] = self.splits[name]
                break
        if multi_site and self.split == "stratified":
            warnings.warn("split='stratified' mixes acquisition sites between train and test: "
                          "the score is optimistic. Use split='site_grouped'.",
                          UserWarning, stacklevel=3)
        self.scale = 1.0
        if self.split is not None:
            if self.split not in self.splits:
                reason = self.split_errors.get(
                    "site_grouped" if self.split == "recommended" else self.split)
                raise ValueError(f"Split {self.split!r} is not available"
                                 + (f" ({reason})" if reason else "")
                                 + f"; available: {sorted(self.splits)}")
            if self.normalize:
                self.scale = geometric_scale(self.covs[self.splits[self.split]["train"]])

    # ------------------------------------------------------------ access
    @property
    def covariances(self) -> np.ndarray:
        """All covariances (n_samples, p, p), scaled by :attr:`scale`."""
        return self.covs * self.scale

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        # covariance scaled by s, mean and raw by sqrt(s): the three stay consistent
        root = float(np.sqrt(self.scale))
        cov = torch.from_numpy(self.covs[idx] * self.scale)
        mean = torch.from_numpy(self.means[idx] * root) if self.return_mean else None
        raw = (torch.from_numpy(np.array(self.series[self.subject_index[idx]], dtype=np.float32)
                                * np.float32(root))
               if self.return_raw else None)
        label = self.labels[idx]
        return self._pack(cov, mean, raw), (float(label) if self.task == "regression"
                                            else int(label))
