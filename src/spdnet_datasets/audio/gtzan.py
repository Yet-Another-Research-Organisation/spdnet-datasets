"""GTZAN: music genre classification (Tzanetakis & Cook, 2002).

10 genres x 100 excerpts of 30 s at 22 050 Hz. The local copy (Hugging Face
export, parquet files with columns ``file``, ``audio``, ``genre``) holds 999
excerpts: ``jazz.00054`` is missing (known corrupted file). The Hugging Face
metadata announce 32 kHz; the real rate (22 050 Hz) is read from each WAV file.

Known defects (Sturm, 2013): repetitions, mislabelings, distortions. The 14
pairs of exact duplicates measured by the atlas (:data:`DUPLICATES`) are
removed (the second file of each pair), which leaves 985 excerpts.

Download
--------
The local copy is the Hugging Face dataset ``sanchit-gandhi/gtzan`` (three
parquet shards, 1.3 GB; the sha256 of :data:`FILES` are the Hugging Face LFS
identifiers, equal to those of the local copy). When ``data_dir`` does not
exist, the shards of the pinned revision :data:`HF_REVISION` are downloaded
into ``<cache_dir>/gtzan/raw/`` and checked. The excerpts come from commercial
recordings whose redistribution rights were never cleared: research use only.

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
    download_files,
    estimate_stage,
    evenly_spaced,
    folds_to_split,
    kfold_assignment,
    load_result,
)
from ._features import iter_parquet_rows, process_recordings

DOMAIN, NAME = "AUDIO", "GTZAN"
HF_REPO, HF_REVISION = "sanchit-gandhi/gtzan", "4bd857132cb0e731bef3ec68558e7acc0a85f144"
_HF = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/data/"
FILES = {  # url -> sha256 (Hugging Face LFS oid, identical to the local copy)
    _HF + "train-00000-of-00003-abaaa5719027ce5c.parquet":
        "e4030bb79bc745a3255d85a3b5da49d6733be9cb2af4cba030eec1621f75a50a",
    _HF + "train-00001-of-00003-40e2de07ad428882.parquet":
        "7e60e413743fbe7b6d4798161af05c676169a65f043dc15dc2146f63210f1dc8",
    _HF + "train-00002-of-00003-6e2eb838540a06e5.parquet":
        "03876df22ec942651bf0170ecae2ddbe23c465cb2ce63171c3b6de2a4beb5fe2",
}
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
    root = root if source is None else source  # downloaded shards
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
    if root.is_dir():
        check_local(root, verify)
        raw_fn, source = None, str(root.resolve())
    else:
        raw_fn, source = partial(download_files, FILES), f"hf:{HF_REPO}@{HF_REVISION}"
    params = {"data_dir": source, "features": features, "n_mfcc": n_mfcc,
                  "n_mels": n_mels, "limit": limit}
    run = run_pipeline(
        "gtzan", params=params, covariance_params={"estimator": estimator}, raw_fn=raw_fn,
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
