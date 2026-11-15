"""ESC-50: environmental sound classification (Piczak, 2015).

2 000 clips of 5 s (44.1 kHz) from Freesound, 50 balanced classes (40 clips
each) grouped in 5 major categories, with 5 official folds of 400 clips. The
local copy is a Hugging Face export in parquet files (columns ``filename``,
``fold``, ``target``, ``category``, ``esc10``, ``src_file``, ``take``,
``audio``).

Download
--------
The local copy is the Hugging Face dataset ``ashraq/esc50`` (two parquet
shards, 773 MB; the sha256 of :data:`FILES` are the Hugging Face LFS
identifiers, equal to those of the local copy). When ``data_dir`` does not
exist, the two shards of the pinned revision :data:`HF_REVISION` are
downloaded into ``<cache_dir>/esc50/raw/`` and checked. Clips: CC BY-NC 3.0
(Freesound).

Covariances
-----------
One covariance per clip over its non-silent frames (``n`` = number of frames,
427 for a clip without silence): ``p = 20`` MFCC by default
(``features="mfcc"``), or the 64 log-mel bands (``features="logmel"``). Clips
with ``n < 2p`` non-silent frames (digital silence) or a non-SPD estimate are
excluded and listed in ``excluded``.

Splits
------
``"official"`` (= ``"recommended"``): test = official fold ``fold``,
validation = the next fold (cyclically), train = the three others. Groups are
the source Freesound recording (``src_file``); the official folds keep the
takes of one recording together, except 4 recordings (measured by the atlas)
that span two folds.

Tasks: ``task="fine"`` (50 classes) or ``task="coarse"`` (the 5 major
categories: animals, natural soundscapes, human non-speech, domestic, urban).
"""

from functools import partial

from ..core import dataset_path, run_pipeline
from ..manager import DatasetManager
from ._common import (
    SequenceCovarianceDataset,
    check_local,
    download_files,
    estimate_stage,
    evenly_spaced,
    folds_to_split,
    load_result,
)
from ._features import iter_parquet_rows, process_recordings

DOMAIN, NAME = "AUDIO", "ESC-50"
HF_REPO, HF_REVISION = "ashraq/esc50", "e3e2a63ffff66b9a9735524551e3818e96af03ee"
_HF = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/data/"
FILES = {  # url -> sha256 (Hugging Face LFS oid, identical to the local copy)
    _HF + "train-00000-of-00002-2f1ab7b824ec751f.parquet":
        "01fafd618bad0813b1c844466c35d0aea4ee02aeb0ffcde6e012ca48dcd38dbb",
    _HF + "train-00001-of-00002-27425e5c1846b494.parquet":
        "57786179368a444868a98430314ce762876e170bed9636845773762b9e3ad62d",
}
N_FOLDS = 5
COARSE_NAMES = ("animals", "natural_soundscapes", "human_non_speech", "domestic", "urban")
META_COLUMNS = ["filename", "fold", "target", "category", "esc10", "src_file", "take"]


def _files(root):
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet file in {root}")
    return files


def _index(root):
    """Metadata of every clip (no audio decoded)."""
    import pandas as pd

    rows = [row for _, row in iter_parquet_rows(_files(root), META_COLUMNS)]
    return pd.DataFrame(rows)


def _process(source, out_dir, *, root, limit, features, n_mfcc, n_mels):
    root = root if source is None else source  # downloaded shards
    index = _index(root)
    keep = evenly_spaced(len(index), limit)

    def records():
        for pos, row in iter_parquet_rows(_files(root), ["audio"], keep):
            meta = index.iloc[pos]
            yield {"id": meta.filename, "label": meta.category, "group": int(meta.src_file),
                   "fold": int(meta.fold), "target": int(meta.target),
                   "esc10": bool(meta.esc10), "blob": row["audio"]["bytes"]}

    process_recordings(records(), out_dir, features=features, n_mfcc=n_mfcc, n_mels=n_mels)


def esc50_splits(meta, labels, groups, fold=1):
    """Official 5-fold protocol (see module docstring)."""
    parts = folds_to_split(meta["fold"].to_numpy(), fold, N_FOLDS)
    return {"official": parts, "recommended": parts}


def fetch_esc50(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                verify="size", features="mfcc", n_mfcc=20, n_mels=64, limit=None, fold=1):
    """Covariances of the ESC-50 clips.

    Parameters
    ----------
    data_dir : path, optional
        Read-only copy (default ``<data root>/AUDIO/ESC-50``). If the folder does
        not exist, the Hugging Face shards are downloaded (see :data:`FILES`).
    cache_dir : path, optional
        Cache of the processed features and covariances.
    keep : {"keep_processed", "keep_covariance", "keep_full"}, optional
    estimator : {"scm", "ledoit_wolf", "tyler", "student"}
    force : bool
        Recompute every stage.
    verify : {"size", "md5", None}
        Check of the local copy against ``MANIFEST_NAS.tsv``.
    features : {"mfcc", "logmel"}
    n_mfcc, n_mels : int
        Number of MFCC (p for ``mfcc``) and of mel bands (p for ``logmel``).
    limit : int, optional
        Process at most ``limit`` clips, evenly spread (quick tests).
    fold : int
        Official test fold (1..5) of the returned ``splits``.

    Returns
    -------
    Bunch
        ``covariances`` (B, p, p) unscaled, ``means`` (B, p), ``labels``
        (category names), ``groups`` (``src_file``), ``meta`` (DataFrame),
        ``excluded`` (DataFrame), ``splits``, ``processed_path``,
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
        "esc50", params=params, covariance_params={"estimator": estimator}, raw_fn=raw_fn,
        process_fn=partial(_process, root=root, limit=limit, features=features,
                           n_mfcc=n_mfcc, n_mels=n_mels),
        covariance_fn=partial(estimate_stage, estimator=estimator),
        cache_dir=cache_dir, keep=keep, force=force)
    return load_result(run, esc50_splits, fold=fold)


@DatasetManager.register_dataset("esc50")
class ESC50Dataset(SequenceCovarianceDataset):
    """ESC-50 covariances (see :mod:`spdnet_datasets.audio.esc50`).

    Parameters
    ----------
    features : {"mfcc", "logmel"}
    n_mfcc, n_mels : int
    **kwargs
        :class:`~spdnet_datasets.audio._common.SequenceCovarianceDataset`
        arguments (``data_dir``, ``split``, ``subset``, ``fold``, ``task``, ...).
    """

    DOMAIN, NAME = DOMAIN, NAME

    def __init__(self, data_dir=None, *, features="mfcc", n_mfcc=20, n_mels=64, **kwargs):
        self.feature_kwargs = {"features": features, "n_mfcc": n_mfcc, "n_mels": n_mels}
        super().__init__(data_dir, **kwargs)

    def _fetch(self):
        b = fetch_esc50(self.data_dir, fold=self.fold, **self.fetch_kwargs,
                        **self.feature_kwargs)
        # coarse task: major category = target // 10
        self.COARSE_MAP = {c: COARSE_NAMES[t // 10] for c, t in
                           zip(b.meta["label"], b.meta["target"], strict=True)}
        return b

    def _make_splits(self, meta, labels, groups):
        return esc50_splits(meta, labels, groups, fold=self.fold)


__all__ = ["ESC50Dataset", "esc50_splits", "fetch_esc50"]
