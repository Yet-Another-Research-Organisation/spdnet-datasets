"""BirdSet XCM: bird species from focal Xeno-Canto recordings (Rauch et al., 2024).

XCM is the "medium" training set of BirdSet: focal Xeno-Canto recordings
(Vorbis, 32 kHz, variable length, median 32 s) of the species of the 8
soundscape test sets.

.. warning::
   **Partial copy.** Only 21 parquet shards out of at least 341 were downloaded
   (``xcm_00000``, ``xcm_00017``, ..., ``xcm_00340``), i.e. 5 371 recordings;
   the shards are sorted by species, so only 136 of the 411 species are
   present. The official BirdSet test sets (soundscapes) are absent.

Samples: species with at least ``min_recordings`` (10) recordings in the copy
(95 species, 5 185 recordings). ``ebird_code_multilabel`` holds one species
per row, so the task is single-label. The atlas measured 10 pairs of identical
5 s excerpts under two Xeno-Canto identifiers; they are kept.

Covariances
-----------
One covariance per recording over its non-silent frames (whole recording by
default; ``crop_s`` restricts it to an excerpt of ``crop_s`` seconds centred on
the first detected event, as the atlas sheet did with 5 s): ``p = 20`` MFCC
(default) or 64 log-mel bands.

Splits
------
``"group_kfold"`` (= ``"recommended"``): 5-fold ``StratifiedGroupKFold`` on the
species grouped by recordist (seed 0); test = fold ``fold``, validation = the
next fold, train = the others. One recordist often records the same individual
or site, hence the grouping. Empty recordist names form one group ``"NA"``.

Download (manual): Hugging Face ``DBD-research-group/BirdSet`` (CC BY-NC 4.0 card,
per-recording licences), configuration ``XCM`` (branch ``data``, folder ``XCM/``:
audio tar shards and ``XCM_metadata.parquet``). The local ``xcm_<k>.parquet`` shards
are a partial re-export that matches no
published file (the published XCM audio is in tar shards), so no checksum is registered: export the wanted shards with the
``datasets`` library (same columns) and place it under ``data_dir`` (default ``<data
root>/AUDIO/BirdSet_XCM``).
"""

import json
import warnings
from functools import partial
from pathlib import PurePosixPath

import numpy as np

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

DOMAIN, NAME = "AUDIO", "BirdSet_XCM"
FILES = {}  # local copy; download registry filled by the fetch phase
N_FOLDS = 5
PARTIAL_WARNING = ("BirdSet_XCM is a partial copy (21 shards, 136 of 411 species) without "
                   "the official soundscape test sets.")
INDEX_COLUMNS = ["filepath", "ebird_code", "recordist", "length", "detected_events"]


def _files(root):
    files = sorted(root.glob("xcm_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No xcm_*.parquet file in {root}")
    return files


def _species_names(path):
    """eBird codes of the ``ebird_code`` ClassLabel (Hugging Face schema metadata)."""
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(path).schema_arrow.metadata or {}
    try:
        info = json.loads(meta[b"huggingface"])["info"]["features"]["ebird_code"]
        return list(info["names"])
    except (KeyError, ValueError):
        return None


def _index(root, min_recordings=10):
    """Metadata of every recording and the mask of the kept species (no audio)."""
    import pandas as pd

    files = _files(root)
    names = _species_names(files[0])
    rows = []
    for _, row in iter_parquet_rows(files, INDEX_COLUMNS):
        code = row["ebird_code"]
        events = row["detected_events"] or []
        rows.append({"id": PurePosixPath(row["filepath"]).stem,
                     "label": names[code] if names else str(code),
                     "group": row["recordist"] or "NA", "length": row["length"],
                     "event": (events[0][0] + events[0][1]) / 2 if events else np.nan})
    index = pd.DataFrame(rows)
    counts = index["label"].value_counts()
    kept = index["label"].isin(counts.index[counts >= min_recordings]).to_numpy()
    return index, kept


def _process(source, out_dir, *, root, limit, min_recordings, crop_s, features, n_mfcc,
             n_mels):
    index, kept = _index(root, min_recordings)
    candidates = np.flatnonzero(kept)
    keep = candidates[evenly_spaced(len(candidates), limit)]

    def records():
        for pos, row in iter_parquet_rows(_files(root), ["audio"], keep):
            meta = index.iloc[pos]
            rec = {"id": meta.id, "label": meta.label, "group": meta.group,
                   "blob": row["audio"]["bytes"]}
            if crop_s is not None:
                centre = 0.0 if np.isnan(meta.event) else meta.event
                rec["crop"] = (max(0.0, centre - crop_s / 2), crop_s)
            yield rec

    process_recordings(records(), out_dir, features=features, n_mfcc=n_mfcc, n_mels=n_mels)


def birdset_xcm_splits(meta, labels, groups, fold=1, seed=0):
    """``StratifiedGroupKFold`` by recordist (see module docstring)."""
    parts = folds_to_split(kfold_assignment(labels, groups, N_FOLDS, seed), fold, N_FOLDS)
    return {"group_kfold": parts, "recommended": parts}


def fetch_birdset_xcm(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                      verify="size", features="mfcc", n_mfcc=20, n_mels=64, limit=None,
                      fold=1, min_recordings=10, crop_s=None):
    """Covariances of the BirdSet XCM recordings (partial copy, see module docstring).

    Parameters are those of :func:`spdnet_datasets.audio.esc50.fetch_esc50`
    (``data_dir`` default ``<data root>/AUDIO/BirdSet_XCM``), plus:

    min_recordings : int
        Species with fewer recordings in the copy are left out.
    crop_s : float, optional
        Excerpt length (s) centred on the first detected event; ``None`` uses
        the whole recording.

    Returns
    -------
    Bunch
        ``covariances``, ``means``, ``labels`` (eBird codes), ``groups``
        (recordist), ``meta``, ``excluded``, ``splits``, ``processed_path``,
        ``covariance_path``.
    """
    root = dataset_path(DOMAIN, NAME, data_dir)
    check_local(root, verify)
    params = {"data_dir": str(root.resolve()), "features": features, "n_mfcc": n_mfcc,
                  "n_mels": n_mels, "limit": limit, "min_recordings": min_recordings, "crop_s": crop_s,
                  "estimator": estimator}
    run = run_pipeline(
        "birdset_xcm", params=params,
        process_fn=partial(_process, root=root, limit=limit, min_recordings=min_recordings,
                           crop_s=crop_s, features=features, n_mfcc=n_mfcc, n_mels=n_mels),
        covariance_fn=partial(estimate_stage, estimator=estimator),
        cache_dir=cache_dir, keep=keep, force=force)
    return load_result(run, birdset_xcm_splits, fold=fold)


@DatasetManager.register_dataset("birdset_xcm")
class BirdSetXCMDataset(SequenceCovarianceDataset):
    """BirdSet XCM covariances (see :mod:`spdnet_datasets.audio.birdset_xcm`).

    Emits a ``UserWarning`` about the partial copy.
    """

    DOMAIN, NAME = DOMAIN, NAME

    def __init__(self, data_dir=None, *, features="mfcc", n_mfcc=20, n_mels=64,
                 min_recordings=10, crop_s=None, **kwargs):
        self.feature_kwargs = {"features": features, "n_mfcc": n_mfcc, "n_mels": n_mels,
                                   "min_recordings": min_recordings, "crop_s": crop_s}
        warnings.warn(PARTIAL_WARNING, UserWarning, stacklevel=2)
        super().__init__(data_dir, **kwargs)

    def _fetch(self):
        return fetch_birdset_xcm(self.data_dir, fold=self.fold, **self.fetch_kwargs,
                                 **self.feature_kwargs)

    def _make_splits(self, meta, labels, groups):
        return birdset_xcm_splits(meta, labels, groups, fold=self.fold, seed=0)


__all__ = ["BirdSetXCMDataset", "birdset_xcm_splits", "fetch_birdset_xcm"]
