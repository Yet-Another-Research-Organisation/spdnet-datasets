"""GTZAN: music genre classification (Tzanetakis & Cook, 2002).

10 genres x 100 excerpts of 30 s at 22 050 Hz. The local copy (Hugging Face
export, parquet files with columns ``file``, ``audio``, ``genre``) holds 999
excerpts: ``jazz.00054`` is missing (known corrupted file). The Hugging Face
metadata announce 32 kHz; the real rate (22 050 Hz) is read from each WAV file.

Known defects (Sturm, 2013): repetitions, mislabelings, distortions. The 14
pairs of exact duplicates measured by the atlas (:data:`DUPLICATES`) are
removed (the second file of each pair), which leaves 985 excerpts.

Covariances
-----------
One covariance per excerpt over its non-silent frames (``n`` around 2 580):
``p = 20`` MFCC (default) or 64 log-mel bands.

Splits
------
There is no official split and the copy has no artist metadata, so the
artist-filtered split of Kereliuk, Sturm & Larsen (2015) cannot be rebuilt.
``"stratified"`` (= ``"recommended"``): 5 stratified folds (seed 0); test =
fold ``fold``, validation = the next fold, train = the others. Excerpts of one
artist (and repeated songs) fall on both sides: scores are optimistic, and the
dataset warns about it.
"""

import warnings
from functools import partial
from pathlib import PurePosixPath

from ..core import dataset_path, run_pipeline
from ..manager import DatasetManager
from ._common import (
    SequenceCovarianceDataset,
    check_local,
    estimate_stage,
    evenly_spaced,
    folds_to_split,
    kfold_assignment,
    load_result,
)
from ._features import iter_parquet_rows, process_recordings

DOMAIN, NAME = "AUDIO", "GTZAN"
FILES = {}  # local copy; download registry filled by the fetch phase
N_FOLDS = 5
GENRES = ("blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop",
          "reggae", "rock")  # ClassLabel order of the Hugging Face export
DUPLICATES = (  # (kept, removed): identical signals measured by the atlas
    ("pop.00054.wav", "pop.00060.wav"), ("pop.00067.wav", "pop.00071.wav"),
    ("metal.00034.wav", "metal.00094.wav"), ("metal.00042.wav", "metal.00063.wav"),
    ("disco.00098.wav", "disco.00099.wav"), ("metal.00043.wav", "metal.00064.wav"),
    ("metal.00040.wav", "metal.00061.wav"), ("reggae.00081.wav", "reggae.00082.wav"),
    ("metal.00058.wav", "rock.00016.wav"), ("metal.00044.wav", "metal.00065.wav"),
    ("metal.00045.wav", "metal.00066.wav"), ("hiphop.00039.wav", "hiphop.00045.wav"),
    ("metal.00041.wav", "metal.00062.wav"), ("hiphop.00076.wav", "hiphop.00078.wav"),
)
ARTIST_WARNING = ("GTZAN has no artist metadata: the stratified folds put excerpts of one "
                  "artist (and repeated songs) in train and test, so scores are optimistic.")


def _files(root):
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet file in {root}")
    return files


def _index(root):
    """``(name, genre)`` of every excerpt of the copy, duplicates included."""
    return [(PurePosixPath(row["file"]).name, GENRES[row["genre"]])
            for _, row in iter_parquet_rows(_files(root), ["file", "genre"])]


def _process(source, out_dir, *, root, limit, features, n_mfcc, n_mels):
    index = _index(root)
    removed = {b for _, b in DUPLICATES}
    candidates = [i for i, (name, _) in enumerate(index) if name not in removed]
    keep = [candidates[k] for k in evenly_spaced(len(candidates), limit)]

    def records():
        for pos, row in iter_parquet_rows(_files(root), ["audio"], keep):
            name, genre = index[pos]
            yield {"id": name, "label": genre, "group": name, "blob": row["audio"]["bytes"]}

    process_recordings(records(), out_dir, features=features, n_mfcc=n_mfcc, n_mels=n_mels)


def gtzan_splits(meta, labels, groups, fold=1, seed=0):
    """Stratified 5-fold split (no artist information; see module docstring)."""
    parts = folds_to_split(kfold_assignment(labels, None, N_FOLDS, seed), fold, N_FOLDS)
    return {"stratified": parts, "recommended": parts}


def fetch_gtzan(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                verify="size", features="mfcc", n_mfcc=20, n_mels=64, limit=None, fold=1):
    """Covariances of the GTZAN excerpts (exact duplicates removed).

    Parameters are those of :func:`spdnet_datasets.audio.esc50.fetch_esc50`
    (``data_dir`` default ``<data root>/AUDIO/GTZAN``; ``fold`` selects the test
    fold of the stratified split).

    Returns
    -------
    Bunch
        ``covariances``, ``means``, ``labels`` (genre names), ``groups``
        (excerpt name), ``meta``, ``excluded``, ``splits``, ``processed_path``,
        ``covariance_path``.
    """
    root = dataset_path(DOMAIN, NAME, data_dir)
    check_local(root, verify)
    params = {"data_dir": str(root.resolve()), "features": features, "n_mfcc": n_mfcc,
                  "n_mels": n_mels, "limit": limit, "estimator": estimator}
    run = run_pipeline(
        "gtzan", params=params,
        process_fn=partial(_process, root=root, limit=limit, features=features,
                           n_mfcc=n_mfcc, n_mels=n_mels),
        covariance_fn=partial(estimate_stage, estimator=estimator),
        cache_dir=cache_dir, keep=keep, force=force)
    return load_result(run, gtzan_splits, fold=fold)


@DatasetManager.register_dataset("gtzan")
class GTZANDataset(SequenceCovarianceDataset):
    """GTZAN covariances (see :mod:`spdnet_datasets.audio.gtzan`).

    Emits a ``UserWarning`` about the missing artist split.
    """

    DOMAIN, NAME = DOMAIN, NAME

    def __init__(self, data_dir=None, *, features="mfcc", n_mfcc=20, n_mels=64, **kwargs):
        self.feature_kwargs = {"features": features, "n_mfcc": n_mfcc, "n_mels": n_mels}
        warnings.warn(ARTIST_WARNING, UserWarning, stacklevel=2)
        super().__init__(data_dir, **kwargs)

    def _fetch(self):
        return fetch_gtzan(self.data_dir, fold=self.fold, **self.fetch_kwargs,
                           **self.feature_kwargs)

    def _make_splits(self, meta, labels, groups):
        return gtzan_splits(meta, labels, groups, fold=self.fold, seed=0)


__all__ = ["DUPLICATES", "GTZANDataset", "fetch_gtzan", "gtzan_splits"]
