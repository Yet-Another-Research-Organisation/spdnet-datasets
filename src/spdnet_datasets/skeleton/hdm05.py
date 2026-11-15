"""HDM05: motion capture actions (Müller et al., 2007), as used by SPDNet.

The local copy only holds the 2 086 covariance matrices 93 x 93 (31 joints x 3
coordinates) of the SPDNet benchmark (Huang & Van Gool, 2017), one per motion
sequence, in 117 action classes: files ``{id}_{frames}_{class}.npy``. The
skeletons, the estimator used and the actor / take identifiers are absent.

Covariances
-----------
- ``features="c93"`` (default): the provided matrices, ``p = 93``. ``n`` is
  the number of frames of the sequence (94 to 901, median 251); the 604
  sequences with fewer than ``2p = 186`` frames are excluded and listed in
  ``excluded`` (the matrices cannot be re-estimated).
- ``features="c31"``: 31 x 31 joint covariance, sum of the three per-coordinate
  blocks of C93 (variables ordered coordinate-major, as inferred by the atlas):
  ``C31[j, k] = sum_d C93[31 d + j, 31 d + k]``; ``2p = 62 <= 94`` keeps every
  sequence.

Only covariances are available (no raw data, no mean) and only ``"scm"`` (the
matrices are given).

Splits
------
``"stratified"`` (= ``"recommended"``): 5 stratified folds (seed 0); test =
fold ``fold``, validation = the next fold, train = the others.

.. warning::
   **Actor leakage.** HDM05 has 5 actors, but the copy has no actor or take
   identifier: segments of one take and one actor fall in train and test,
   which inflates the scores compared with a cross-actor protocol. The SPDNet
   literature uses 10 random 50/50 splits, which has the same bias.

Download (manual): the motion database is published at
http://resources.mpi-inf.mpg.de/HDM05/ (free for research, C3D/AMC files). The
``{id}_{frames}_{class}.npy`` covariances read here are a preprocessed export whose
construction is not published, so this package cannot rebuild them from the original
files; copy the export and place it under ``data_dir`` (default ``<data
root>/SKELETON/HDM05``).
"""

import re
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from ..audio._common import (
    CSV,
    SequenceCovarianceDataset,
    evenly_spaced,
    exclusion,
    folds_to_split,
    kfold_assignment,
    load_result,
    save_covariances,
)
from ..core import (
    InsufficientSamplesError,
    NotSPDError,
    check_local,
    check_sample_size,
    check_spd,
    dataset_path,
    run_pipeline,
)
from ..manager import DatasetManager

DOMAIN, NAME = "SKELETON", "HDM05"
FILES = {}  # local copy only (the original database needs a new download)
N_FOLDS = 5
FEATURES = ("c93", "c31")
PATTERN = re.compile(r"^(\d+)_(\d+)_(\d+)\.npy$")
ACTOR_WARNING = ("HDM05 has no actor identifier: sequences of one actor and one take fall "
                 "in train and test, so scores are optimistic.")


def _index(root):
    rows = []
    for path in sorted(Path(root).glob("*.npy")):
        m = PATTERN.match(path.name)
        if m:
            seq, frames, cls = m.groups()
            rows.append({"id": path.stem, "label": f"{int(cls):03d}", "group": path.stem,
                         "seq": int(seq), "n_obs": int(frames), "file": path.name})
    if not rows:
        raise FileNotFoundError(f"No {{id}}_{{frames}}_{{class}}.npy file in {root}")
    return pd.DataFrame(rows).sort_values("seq", kind="stable").reset_index(drop=True)


def c31(C):
    """Sum of the three per-coordinate blocks of a coordinate-major 93 x 93 matrix."""
    q = C.shape[0] // 3
    return sum(C[d * q:(d + 1) * q, d * q:(d + 1) * q] for d in range(3))


def _process(source, out_dir, *, root, limit):
    index = _index(root)
    index.iloc[evenly_spaced(len(index), limit)].to_csv(Path(out_dir) / "index.csv",
                                                        index=False)


def _covariances(proc_dir, out_dir, *, root, features):
    index = pd.read_csv(Path(proc_dir) / "index.csv",
                        dtype={"id": str, "label": str, "group": str}, **CSV)
    covs, rows, excluded = [], [], []
    for i, rec in enumerate(index.itertuples(index=False)):
        C = np.load(Path(root) / rec.file).astype(np.float64)
        if features == "c31":
            C = c31(C)
        try:
            check_sample_size(rec.n_obs, C.shape[0])
            check_spd(C)
        except (InsufficientSamplesError, NotSPDError) as exc:
            excluded.append(exclusion(rec.id, rec.label, exc))
            continue
        covs.append((C + C.T) / 2)
        rows.append(i)
    p = covs[0].shape[0] if covs else 0
    save_covariances(out_dir, covs, [np.zeros(p)] * len(covs), index.iloc[rows], excluded, p)


def hdm05_splits(meta, labels, groups, fold=1, seed=0):
    """Stratified 5-fold split (no actor information; see module docstring)."""
    parts = folds_to_split(kfold_assignment(labels, None, N_FOLDS, seed), fold, N_FOLDS)
    return {"stratified": parts, "recommended": parts}


def fetch_hdm05(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                verify="size", features="c93", limit=None, fold=1):
    """Covariances of the HDM05 sequences.

    Parameters
    ----------
    data_dir : path, optional
        Read-only copy (default ``<data root>/SKELETON/HDM05``).
    estimator : {"scm"}
        The matrices are provided; any other value raises ``ValueError``.
    features : {"c93", "c31"}
        See the module docstring.
    limit : int, optional
        At most ``limit`` sequences, evenly spread.
    fold : int
        Test fold (1..5) of the returned ``splits``.
    cache_dir, keep, force, verify
        See :func:`spdnet_datasets.core.fetch.run_pipeline`.

    Returns
    -------
    Bunch
        ``covariances`` (B, p, p), ``labels`` (``"000"`` to ``"116"``),
        ``groups`` (sequence id), ``meta``, ``excluded``, ``splits``.
    """
    if estimator != "scm":
        raise ValueError("HDM05 covariances are provided: only estimator='scm' is available")
    if features not in FEATURES:
        raise ValueError(f"features must be one of {FEATURES}, got {features!r}")
    root = dataset_path(DOMAIN, NAME, data_dir)
    check_local(root, verify=verify)
    params = {"data_dir": str(root.resolve()), "features": features, "limit": limit}
    run = run_pipeline(
        "hdm05", params=params,
        process_fn=partial(_process, root=root, limit=limit),
        covariance_fn=partial(_covariances, root=root, features=features),
        cache_dir=cache_dir, keep=keep, force=force)
    return load_result(run, hdm05_splits, fold=fold)


@DatasetManager.register_dataset("hdm05")
class HDM05Dataset(SequenceCovarianceDataset):
    """HDM05 covariances (see :mod:`spdnet_datasets.skeleton.hdm05`).

    Parameters
    ----------
    features : {"c93", "c31"}
    **kwargs
        :class:`~spdnet_datasets.audio._common.SequenceCovarianceDataset`
        arguments; ``classes``, ``max_classes`` and ``max_samples_per_class``
        are honoured.

    Emits a ``UserWarning`` about the actor leakage.
    """

    DOMAIN, NAME = DOMAIN, NAME
    OUTPUTS = ("cov",)

    def __init__(self, data_dir=None, *, features="c93", **kwargs):
        self.features = features
        warnings.warn(ACTOR_WARNING, UserWarning, stacklevel=2)
        super().__init__(data_dir, **kwargs)

    def _fetch(self):
        return fetch_hdm05(self.data_dir, features=self.features, fold=self.fold,
                           **self.fetch_kwargs)

    def _make_splits(self, meta, labels, groups):
        return hdm05_splits(meta, labels, groups, fold=self.fold, seed=0)


__all__ = ["HDM05Dataset", "c31", "fetch_hdm05", "hdm05_splits"]
