"""PyTorch dataset of SPD covariances built from a MOABB dataset and paradigm.

MOABB downloads, reads and epochs the recordings (``paradigm.get_data``); its
raw files stay in the MNE data directory (``MNE_DATA``) and are never touched
here. This module only caches what it derives from the epochs, under
``SPDNET_CACHE_DIR/moabb_<dataset code>/`` through
:func:`spdnet_datasets.core.fetch.run_pipeline`::

    processed/<hash>/   epochs.npy (float32), labels.npy, metadata.csv,
                        channels.npy, excluded_subjects.npy
    covariance/<hash>/  covariances.npy, means.npy, labels.npy, metadata.csv,
                        trial_index.npy, n_excluded.npy, channels.npy,
                        excluded_subjects.npy

The covariance stage is keyed on the estimator only on top of the epochs key,
so changing the estimator does not call ``paradigm.get_data`` again.
"""

import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ..base import BaseDataset
from ..core.config import get_cache_dir
from ..core.estimators import InsufficientSamplesError, NotSPDError
from ..core.fetch import run_pipeline
from ..core.normalize import geometric_scale
from ..manager import DatasetManager
from .covariances import epochs_to_covariances

SPLIT_KINDS = ("within_session", "cross_session", "cross_subject")


_MAX_DEPTH = 6


def _key_value(v, depth=0):
    """JSON-friendly, process-stable fingerprint of ``v`` for the cache key.

    Objects are fingerprinted by their configuration (``get_params()`` for
    scikit-learn-like objects, else their public attributes), never by their
    ``repr``: two objects of one class with different settings get different
    keys. A value that cannot be fingerprinted raises ``TypeError`` instead of
    risking a silent cache collision.
    """
    if depth > _MAX_DEPTH:
        raise TypeError(f"Cannot fingerprint {type(v).__name__} for the cache key "
                        "(nested too deeply)")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (bool, int, float, str, type(None))):
        return v
    if isinstance(v, (list, tuple, set, frozenset)):
        items = [_key_value(x, depth + 1) for x in v]
        return sorted(items, key=repr) if isinstance(v, (set, frozenset)) else items
    if isinstance(v, dict):
        return {str(k): _key_value(x, depth + 1)
                for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, np.ndarray):
        return _key_value(v.tolist(), depth + 1)
    if isinstance(v, type) or callable(v) and hasattr(v, "__qualname__"):
        return f"{v.__module__}.{v.__qualname__}"
    name = f"{type(v).__module__}.{type(v).__qualname__}"
    if hasattr(v, "get_params"):
        return {"class": name, "params": _key_value(v.get_params(deep=False), depth + 1)}
    if hasattr(v, "__dict__"):
        return {"class": name, "attrs": _key_attributes(v, depth + 1)}
    raise TypeError(f"Cannot fingerprint {name} for the cache key; pass plain values "
                    "(numbers, strings, lists, dicts or objects with public attributes)")


def _key_attributes(obj, depth=0) -> dict:
    """Public attributes of ``obj`` (dataset or paradigm options) for the cache key."""
    return {k: _key_value(v, depth) for k, v in sorted(vars(obj).items())
            if not k.startswith("_")}


def _read_metadata(path):
    return pd.read_csv(path, dtype={"session": str, "run": str})


def moabb_splits(y, metadata, n_folds=5, seed=42) -> dict:
    """Train/test indices of the three MOABB evaluation protocols.

    The folds come from MOABB's own splitters
    (:class:`moabb.evaluations.splitters.WithinSessionSplitter`,
    ``CrossSessionSplitter`` and ``CrossSubjectSplitter``), so they match
    ``WithinSessionEvaluation``, ``CrossSessionEvaluation`` and
    ``CrossSubjectEvaluation``.

    Parameters
    ----------
    y : array of shape (n_trials,)
    metadata : DataFrame with ``subject`` and ``session`` columns and a
        positional ``RangeIndex``.
    n_folds : int
        Folds of the within-session protocol (MOABB default: 5).
    seed : int
        Random state of the within-session folds.

    Returns
    -------
    dict
        ``"<protocol>_<k>" -> {"train", "val", "test"}`` (``val`` is empty).
        Each protocol is built on its own: ``cross_session`` needs at least
        one subject with two sessions (MOABB skips single-session subjects),
        ``cross_subject`` at least two subjects, and ``within_session`` enough
        trials per class and session for ``n_folds`` folds. A protocol that
        cannot be built is absent (with a ``UserWarning``) and does not prevent
        the others.
    """
    from moabb.evaluations.splitters import (
        CrossSessionSplitter,
        CrossSubjectSplitter,
        WithinSessionSplitter,
    )

    y = np.asarray(y)
    metadata = metadata.reset_index(drop=True)
    splitters = {"within_session": WithinSessionSplitter(n_folds=n_folds, shuffle=True,
                                                         random_state=seed)}
    if metadata.groupby("subject")["session"].nunique().max() >= 2:
        splitters["cross_session"] = CrossSessionSplitter()
    if metadata["subject"].nunique() >= 2:
        splitters["cross_subject"] = CrossSubjectSplitter()
    empty = np.array([], dtype=np.int64)
    out = {}
    for kind, splitter in splitters.items():
        try:
            folds = [(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
                     for tr, te in splitter.split(y, metadata)]
        except ValueError as exc:  # e.g. fewer trials per class than folds
            warnings.warn(f"MOABB protocol {kind!r} skipped: {exc}", UserWarning,
                          stacklevel=2)
            continue
        for k, (train, test) in enumerate(folds):
            out[f"{kind}_{k}"] = {"train": train, "val": empty, "test": test}
    return out


def stratified_split(y, val_size=0.1, test_size=0.15, seed=42) -> dict:
    """Trial-level train/val/test split, stratified by class over all trials.

    The test set is drawn first, then the validation set from the rest, so both
    sizes are fractions of all trials. Every subject keeps trials in the three
    parts: this is a within-subject protocol. Unlike the MOABB protocols, it
    has a validation set.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    train, val = train_test_split(rest, test_size=val_size / (1.0 - test_size),
                                  random_state=seed, stratify=y[rest])
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def _channel_mismatch(exc) -> bool:
    """Whether ``exc`` is the MNE error of concatenating epochs of other channels."""
    return (isinstance(exc, ValueError)
            and re.search(r"info\['(ch_names|nchan)'\]", str(exc)) is not None)


class MOABBCovariances(BaseDataset):
    """SPD covariances of the epochs of a MOABB dataset, as a PyTorch dataset.

    Parameters
    ----------
    dataset : moabb.datasets.base.BaseDataset
        Any MOABB dataset, e.g. ``BNCI2014_001()``.
    paradigm : moabb.paradigms.base.BaseParadigm
        Any MOABB paradigm compatible with ``dataset``, e.g. ``LeftRightImagery()``.
    subjects : list of int, optional
        Subjects to load (default: ``dataset.subject_list``).
    estimator : {"scm", "ledoit_wolf", "tyler", "student"}
        Covariance estimator; ``n_times >= 2 * n_channels`` is enforced.
    keep : {"keep_processed", "keep_covariance", "keep_full"}, optional
        What stays in the cache. ``keep_covariance`` deletes the cached epochs,
        so ``return_raw`` is then unavailable. The MNE/MOABB raw files are never
        deleted. Default: ``keep_full`` (epochs and covariances kept).
    return_mean, return_raw : bool
        Optional outputs (:mod:`spdnet_datasets.core.outputs`): the channel mean
        (n_channels,) and the epoch (n_times, n_channels).
    split : str, optional
        Name of a split of :attr:`splits`; with ``normalize=True`` the geometric
        scale is fitted on its training indices. Without a split no scale is
        applied (there is no training set to fit it on).
    normalize : bool
        Apply the geometric scale ``s`` fitted on ``splits[split]["train"]``:
        covariances times ``s``, means and raw signals times ``sqrt(s)``.
    n_folds : int
        Folds of the within-session protocol.
    holdout : (float, float), optional
        ``(val_size, test_size)`` of an extra ``stratified`` split
        (:func:`stratified_split`). Default: no such split.
    cache_dir : path, optional
        Default: ``$SPDNET_CACHE_DIR``.
    force : bool
        Recompute the cached epochs and covariances.
    seed : int
        Random state of the within-session folds.

    Attributes
    ----------
    metadata : DataFrame
        MOABB metadata (``subject``, ``session``, ``run``, ...), one row per trial.
    labels : ndarray of int
        Index into :attr:`classes`.
    groups : ndarray of str
        ``"<subject>-<session>"`` of each trial.
    subjects, sessions : ndarray
        Subject and session of each trial.
    splits : dict
        ``within_session_<k>``, ``cross_session_<k>``, ``cross_subject_<k>``
        (see :func:`moabb_splits`), ``recommended``, an alias of the first
        fold of the strictest protocol available (cross-subject, then
        cross-session, then within-session), and ``stratified`` when
        ``holdout`` is set.
    scale : float
        Geometric scale applied to the covariances.
    channels : list of str
        Channel names, the same for every kept trial.
    excluded_subjects : list
        Subjects left out because their channels (names and order) differ from
        the most common channel list, or differ between their own sessions.
    n_excluded : int
        Trials dropped because their covariance could not be estimated
        (too few time samples, or not SPD).
    trial_index : ndarray of int
        Index of each kept trial in the cached epochs.
    """

    def __init__(self, dataset, paradigm, subjects=None, *, estimator="scm", keep=None,
                 return_mean=False, return_raw=False, split=None, normalize=True,
                 n_folds=5, holdout=None, cache_dir=None, force=False, seed=42,
                 verbose=False):
        cache_dir = get_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(cache_dir, seed=seed, verbose=verbose, return_mean=return_mean,
                         return_raw=return_raw, estimator=estimator, normalize=normalize)
        if return_raw and keep == "keep_covariance":
            raise ValueError("return_raw=True needs the cached epochs: use "
                             "keep='keep_processed' or 'keep_full', not 'keep_covariance'.")
        self.dataset, self.paradigm = dataset, paradigm
        self.subjects_requested = sorted(subjects or dataset.subject_list)
        self.keep, self.force, self.n_folds = keep, force, n_folds
        self.holdout = None if holdout is None else tuple(holdout)
        self.split = split
        self._load_data()

    def _params(self):
        return {"dataset": self.dataset.code, "subjects": self.subjects_requested,
                "dataset_params": _key_attributes(self.dataset),
                "paradigm": type(self.paradigm).__name__,
                "paradigm_params": _key_attributes(self.paradigm),
                "version": 2}  # 2: per-subject loading, channel names checked

    def _get_epochs(self, subject):
        """Epochs (float32), labels, metadata and channel names of one subject.

        The epochs are scaled by the unit factor of the dataset, as in the
        arrays returned by MOABB.
        """
        epochs, y, metadata = self.paradigm.get_data(dataset=self.dataset,
                                                     subjects=[subject], return_epochs=True)
        X = (epochs.get_data(copy=False) * self.dataset.unit_factor).astype(np.float32)
        if len(X) != len(y):
            raise ValueError(f"{self.dataset.code}: {len(X)} epochs for {len(y)} labels; "
                             "paradigms with several filters are not supported")
        return X, y, metadata, list(epochs.ch_names)

    def _process(self, _source, out):
        """Epochs of the subjects whose channels are the most common list.

        Subjects are loaded one by one. A subject is excluded when its channel
        list (names and order) differs from the most common one (ties go to the
        longer list), or when its own sessions have different channels.
        """
        parts, excluded = {}, []
        for s in self.subjects_requested:
            try:
                parts[s] = self._get_epochs(s)
            except ValueError as exc:
                if not _channel_mismatch(exc):
                    raise
                excluded.append(s)  # its sessions have different channels
        if not parts:
            raise ValueError(f"{self.dataset.code}: no subject left, the sessions of "
                             f"subjects {excluded} have different channels")
        counts = Counter(tuple(part[3]) for part in parts.values())
        channels = max(counts, key=lambda c: (counts[c], len(c)))
        kept = [s for s, part in parts.items() if tuple(part[3]) == channels]
        excluded = sorted(excluded + [s for s in parts if s not in kept])
        np.save(out / "channels.npy", np.asarray(channels))
        np.save(out / "excluded_subjects.npy", np.asarray(excluded))
        np.save(out / "epochs.npy", np.concatenate([parts[s][0] for s in kept]))
        np.save(out / "labels.npy", np.concatenate([parts[s][1] for s in kept]).astype(str))
        metadata = pd.concat([parts[s][2] for s in kept], ignore_index=True)
        metadata.to_csv(out / "metadata.csv", index=False)

    def _covariances(self, processed, out):
        """Covariances of the epochs; trials whose estimate fails are excluded.

        A trial raising :class:`InsufficientSamplesError` or :class:`NotSPDError`
        is dropped from every array (``trial_index.npy`` maps the kept trials to
        the cached epochs). If no trial is left, the last error is raised.
        """
        X = np.load(processed / "epochs.npy")
        covs, kept, error = [], [], None
        for i, epoch in enumerate(X):
            try:
                covs.append(epochs_to_covariances(epoch[None], self.estimator)[0])
            except (InsufficientSamplesError, NotSPDError) as exc:
                error = exc
                continue
            kept.append(i)
        if not kept:
            raise error
        kept = np.asarray(kept, dtype=np.int64)
        np.save(out / "covariances.npy", np.stack(covs))
        # channel means first, then the kept rows: no copy of the epochs
        np.save(out / "means.npy", X.mean(axis=2, dtype=np.float64)[kept])
        np.save(out / "labels.npy", np.load(processed / "labels.npy")[kept])
        np.save(out / "trial_index.npy", kept)
        np.save(out / "n_excluded.npy", len(X) - len(kept))
        for name in ("channels.npy", "excluded_subjects.npy"):
            np.save(out / name, np.load(processed / name))
        metadata = _read_metadata(processed / "metadata.csv").iloc[kept]
        metadata.reset_index(drop=True).to_csv(out / "metadata.csv", index=False)

    def _load_data(self):
        paths = run_pipeline(f"moabb_{self.dataset.code}", params=self._params(),
                             process_fn=self._process, covariance_fn=self._covariances,
                             cache_dir=self.data_dir, keep=self.keep, force=self.force,
                             covariance_params={"estimator": self.estimator})
        self.paths = paths
        d = paths.covariance_dir
        self.covs = np.load(d / "covariances.npy")
        self.trial_index = np.load(d / "trial_index.npy")
        self.n_excluded = int(np.load(d / "n_excluded.npy"))
        self.channels = np.load(d / "channels.npy").tolist()
        self.excluded_subjects = np.load(d / "excluded_subjects.npy").tolist()
        if self.excluded_subjects:
            warnings.warn(f"{self.dataset.code}: subjects {self.excluded_subjects} excluded, "
                          f"their channels differ from the {len(self.channels)} kept",
                          UserWarning, stacklevel=4)
        self.means = np.load(d / "means.npy")
        self.metadata = _read_metadata(d / "metadata.csv")
        self.classes, self.labels = np.unique(np.load(d / "labels.npy"), return_inverse=True)
        self.classes = self.classes.tolist()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # one entry per trial, as read by base.create_dataloaders (verbose split report)
        self.samples = [{"class_name": self.classes[k]} for k in self.labels]
        self.subjects = self.metadata["subject"].to_numpy()
        self.sessions = self.metadata["session"].to_numpy()
        self.groups = (self.metadata["subject"].astype(str) + "-"
                       + self.metadata["session"].astype(str)).to_numpy()
        self.epochs = None
        if self.return_raw:
            if paths.processed_dir is None:
                raise ValueError("The cached epochs were deleted (keep='keep_covariance'); "
                                 "rebuild with force=True and another keep to use return_raw.")
            self.epochs = np.load(paths.processed_dir / "epochs.npy", mmap_mode="r")

        self.splits = moabb_splits(self.labels, self.metadata, self.n_folds, self.seed)
        for kind in ("cross_subject", "cross_session", "within_session"):
            if f"{kind}_0" in self.splits:
                self.splits["recommended"] = self.splits[f"{kind}_0"]
                break
        if self.holdout is not None:
            self.splits["stratified"] = stratified_split(self.labels, *self.holdout,
                                                         seed=self.seed)
        self.scale = 1.0
        if self.split is not None:
            if self.split not in self.splits:
                raise ValueError(f"Unknown split {self.split!r}; available: {sorted(self.splits)}")
            if self.normalize:
                self.scale = geometric_scale(self.covs[self.splits[self.split]["train"]])

    @property
    def covariances(self) -> np.ndarray:
        """All covariances (n_trials, n_channels, n_channels), scaled by :attr:`scale`."""
        return self.covs * self.scale

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        # covariance scaled by s, mean and raw by sqrt(s): the three stay consistent
        root = float(np.sqrt(self.scale))
        cov = torch.from_numpy(self.covs[idx] * self.scale)
        mean = torch.from_numpy(self.means[idx] * root) if self.return_mean else None
        raw = (torch.from_numpy(np.array(self.epochs[self.trial_index[idx]].T, dtype=np.float32) * np.float32(root))
               if self.return_raw else None)
        return self._pack(cov, mean, raw), int(self.labels[idx])


@DatasetManager.register_dataset("moabb")
class MOABBDataset(MOABBCovariances):
    """MOABB covariances built from class names, for configuration files.

    Registered as ``"moabb"`` in :class:`~spdnet_datasets.manager.DatasetManager`,
    so a Hydra ``dataset`` config of spdnet-training can select it::

        name: moabb
        dataset: BNCI2014_001
        paradigm: MotorImagery
        paradigm_params: {n_classes: 4, fmin: 8, fmax: 32, resample: 128}
        split_strategy: stratified

    Parameters
    ----------
    dataset, paradigm : str
        Class names in :mod:`moabb.datasets` and :mod:`moabb.paradigms`.
    dataset_params, paradigm_params : dict, optional
        Keyword arguments of these classes (``sessions``, ``events``, ``tmin``...).
    subjects : list of int, optional
        Default: ``dataset.subject_list``.
    exclude_subjects : list of int
        Removed from ``subjects``.
    holdout : (float, float)
        ``(val_size, test_size)`` of the ``stratified`` split. The manager's
        ``val_ratio`` and ``test_ratio`` do not apply to named splits.
    split : str, optional
        Split the geometric scale is fitted on. The manager sets it to
        ``split_strategy``.
    input_dim : int, optional
        Expected number of channels, checked after loading.
    data_dir : path, optional
        Alias of ``cache_dir`` (the manager sets it from the config key ``path``).
    **kwargs
        Other arguments of :class:`MOABBCovariances`.
    """

    def __init__(self, dataset, paradigm="MotorImagery", *, dataset_params=None,
                 paradigm_params=None, subjects=None, exclude_subjects=(),
                 holdout=(0.1, 0.15), split=None, input_dim=None, data_dir=None,
                 **kwargs):
        try:
            import moabb.datasets
            import moabb.paradigms
        except ImportError as exc:
            raise ImportError("MOABBDataset needs moabb: "
                              "pip install 'spdnet-datasets[moabb]'") from exc

        for module, name in ((moabb.datasets, dataset), (moabb.paradigms, paradigm)):
            if not hasattr(module, name):
                raise ValueError(f"{module.__name__} has no {name!r}")
        ds = getattr(moabb.datasets, dataset)(**(dataset_params or {}))
        pa = getattr(moabb.paradigms, paradigm)(**(paradigm_params or {}))
        excluded = set(exclude_subjects)
        subjects = ds.subject_list if subjects is None else subjects
        subjects = [s for s in subjects if s not in excluded]
        if not subjects:
            raise ValueError(f"{dataset}: no subject left after exclude_subjects")
        if data_dir is not None:
            kwargs.setdefault("cache_dir", data_dir)
        super().__init__(ds, pa, subjects, holdout=holdout, split=split, **kwargs)
        n_channels = self.covs.shape[-1]
        if input_dim is not None and input_dim != n_channels:
            raise ValueError(f"input_dim={input_dim}, but {dataset} gives "
                             f"{n_channels} channels")
