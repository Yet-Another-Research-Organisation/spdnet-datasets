"""Dop-NET: hand gestures seen by a 24 GHz FMCW radar (complex micro-Doppler).

Source
------
M. Ritchie, R. Capraru, F. Fioranelli, "Dop-NET: a micro-Doppler radar data
challenge", Electronics Letters 56(11), 568-570, 2020, doi:10.1049/el.2019.4153,
https://dop-net.com. Complex spectrograms ``(800 Doppler bins, T frames)``,
``T`` variable (43-540), of 4 gestures by 6 persons A-F.

- ``Training Data/Data_Per_PersonData_Training_Person_<X>.mat``:
  ``Data_Training.Doppler_Signals`` is a 1 x 4 cell array, one cell per gesture.
  **The cells are ordered Wave, Pinch, Swipe, Click** (cells 0, 1, 2, 3). The
  former ``cov/`` precompute named them Click, Swipe, Pinch, Wave: its training
  labels are permuted. The order is checked on the exact copies shared by the
  labelled test file and the training files (every copy agrees: Click = cell 3,
  Pinch = 1, Swipe = 2, Wave = 0).
- ``Test Data/Data_For_Test_Random.mat``: ``Data_rand`` (643, 2), the signal
  and a label string such as ``"Click 20 Person F"`` (same six persons).

Construction (48 x 48)
----------------------
1. Exact duplicates (md5 of the complex array) are removed, keeping the first
   copy, training files first: 111 test gestures are copies of training ones.
2. Gestures with ``T < 2p = 96`` frames are rejected (counted).
3. Band: the smallest band centred on zero Doppler (bin 400) holding at least
   ``energy`` of the mean normalised power profile of the training gestures.
4. The band power ``|S|**2`` is averaged into ``p = 48`` contiguous blocks
   (decimation with a box anti-aliasing filter), then ``log10(. + 1e-12)``.
5. One covariance over the ``T`` frames.

Splits and leakage
------------------
``"official"``: training files / test file (after deduplication, so no test
gesture equals a training one; persons are shared: intra-person split).
``"person<k>"``: leave-one-person-out over all gestures (train and test
files); ``"recommended"`` is ``person{fold}``. The band of step 3 is fitted on
the training files, which include the held-out person of ``person<k>``.
``raw`` has a variable length ``T``: use ``batch_size=1`` or a custom collate.
"""

import hashlib

import numpy as np
import pandas as pd

from ..core.config import dataset_path
from ..core.fetch import run_pipeline
from ..manager import DatasetManager
from ._common import (
    RadarCovarianceDataset,
    check_local,
    covariance_stage,
    load_bunch,
    write_processed,
)

DOMAIN, NAME = "RADAR_DOPPLER", "Dop-NET"
FILES = {}  # registration on dop-net.com: manual download (see the dataset page)
CLASSES = ["Click", "Pinch", "Swipe", "Wave"]
CELL_CLASSES = ["Wave", "Pinch", "Swipe", "Click"]  # order of the training cells
PERSONS = ["A", "B", "C", "D", "E", "F"]
TRAIN_GLOB = "Training Data/Data_Per_PersonData_Training_Person_*.mat"
TEST_FILE = "Test Data/Data_For_Test_Random.mat"


def iter_gestures(root):
    """Yield ``(signal, class_name, person, source, index)``, training files then test file."""
    from scipy.io import loadmat

    for path in sorted(root.glob(TRAIN_GLOB)):
        person = path.stem.rsplit("_", 1)[-1]
        cells = loadmat(path)["Data_Training"]["Doppler_Signals"][0, 0]
        for c in range(cells.shape[1]):
            for i, signal in enumerate(cells[0, c].ravel()):
                yield np.asarray(signal), CELL_CLASSES[c], person, "train", i
    data = loadmat(root / TEST_FILE)["Data_rand"]
    for i in range(data.shape[0]):
        signal = np.asarray(data[i, 0])
        if signal.shape == (1, 1):
            signal = np.asarray(signal[0, 0])
        words = str(np.asarray(data[i, 1]).ravel()[0]).split()
        yield signal, words[0], words[-1], "test", i


def signal_md5(signal):
    return hashlib.md5(np.ascontiguousarray(signal).tobytes()).hexdigest()


def energy_band(profile, energy, centre=None, min_width=1):
    """``(start, stop)`` of the smallest band centred on ``centre`` with ``energy`` of the profile."""
    profile = np.asarray(profile, dtype=np.float64)
    profile = profile / profile.sum()
    n = len(profile)
    centre = n // 2 if centre is None else centre
    for half in range(max(1, (min_width + 1) // 2), n // 2 + 1):
        a, b = max(0, centre - half), min(n, centre + half)
        if profile[a:b].sum() >= energy:
            return a, b
    return 0, n


def band_features(signal, start, stop, p):
    """``(F, T)`` complex -> ``(T, p)`` log10 power of the band averaged into ``p`` blocks."""
    power = np.abs(signal[start:stop]) ** 2
    pooled = np.stack([b.mean(0) for b in np.array_split(power, p, axis=0)], axis=1)
    return np.log10(pooled + 1e-12)


def fetch_dopnet(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                 verify="size", *, energy=0.95, p=48, min_frames=None, fold=0):
    """48 x 48 covariances of the Dop-NET gestures over time frames.

    Parameters
    ----------
    data_dir : path, optional
        Read-only copy of ``Dop-NET`` (default ``<data root>/RADAR_DOPPLER/Dop-NET``).
    cache_dir, keep, force
        See :func:`spdnet_datasets.core.run_pipeline`.
    estimator : str
        Covariance estimator.
    verify : {"size", "md5", None}
        Check of the local copy against ``MANIFEST_NAS.tsv``.
    energy : float
        Fraction of the training power kept in the band centred on zero Doppler.
    p : int
        Number of decimated Doppler channels.
    min_frames : int, optional
        Shortest gesture kept (default ``2p``).
    fold : int
        Held-out person of ``splits["recommended"]`` (0 = A).

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means``, ``labels``,
    ``class_names``, ``groups`` (person), ``meta``, ``info`` (duplicates,
    rejections, band), ``splits``, ``covariance_path`` and ``processed_path``.
    """
    min_frames = 2 * p if min_frames is None else min_frames
    if min_frames < 2 * p:
        raise ValueError(f"min_frames={min_frames} < 2p={2 * p}: covariances need n >= 2p")
    root = dataset_path(DOMAIN, NAME, data_dir)
    check_local(root, verify=verify, required=("Training Data", TEST_FILE))

    def process_fn(_source, out_dir):
        rows, seen, profiles = [], set(), []
        for signal, name, person, source, i in iter_gestures(root):
            md5 = signal_md5(signal)
            duplicate = md5 in seen
            seen.add(md5)
            rows.append({"class_name": name, "person": person, "source": source, "index": i,
                         "n_frames": signal.shape[1], "md5": md5, "duplicate": duplicate})
            keep_it = not duplicate and signal.shape[1] >= min_frames
            if keep_it and source == "train":
                power = (np.abs(signal) ** 2).sum(1)
                profiles.append(power / power.sum())
        table = pd.DataFrame(rows)
        start, stop = energy_band(np.mean(profiles, 0), energy, min_width=p)
        kept = ~table.duplicate & (table.n_frames >= min_frames)
        position = {(r.source, r.class_name, r.person, r.index): k
                    for k, r in enumerate(table[kept].itertuples())}
        spectrograms = [None] * len(position)
        for signal, name, person, source, i in iter_gestures(root):
            k = position.get((source, name, person, i))
            if k is not None:
                spectrograms[k] = band_features(signal, start, stop, p).astype(np.float32)
        meta = table[kept].drop(columns="duplicate").reset_index(drop=True)
        meta["label"] = meta.class_name.map(CLASSES.index)
        meta["group"] = meta.person
        dup = table.duplicate
        test_in_train = table.source.eq("test") & table.md5.isin(table.md5[table.source.eq("train")])
        info = {
            "n_gestures": {s: int(table.source.eq(s).sum()) for s in ("train", "test")},
            "duplicates_removed": {s: int((dup & table.source.eq(s)).sum()) for s in ("train", "test")},
            "test_in_train": int(test_in_train.sum()),
            "rejected_short": {s: int((~dup & table.source.eq(s) & (table.n_frames < min_frames)).sum())
                               for s in ("train", "test")},
            "min_frames": min_frames, "energy": energy, "band": [int(start), int(stop)],
            "band_energy": float(np.mean(profiles, 0)[start:stop].sum()), "p": p,
        }
        write_processed(out_dir, spectrograms, meta, info)

    params = {"energy": energy, "p": p, "min_frames": min_frames, "estimator": estimator,
              "version": 1}
    paths = run_pipeline("dopnet", params=params, process_fn=process_fn,
                         covariance_fn=covariance_stage(estimator), cache_dir=cache_dir,
                         keep=keep, force=force)
    bunch = load_bunch(paths, CLASSES)
    source = bunch.meta.source.to_numpy()
    empty = np.array([], dtype=np.int64)
    bunch.splits = {"official": {"train": np.flatnonzero(source == "train"), "val": empty,
                                 "test": np.flatnonzero(source == "test")}}
    persons = bunch.groups
    for k, person in enumerate(PERSONS):
        bunch.splits[f"person{k}"] = {"train": np.flatnonzero(persons != person), "val": empty,
                                      "test": np.flatnonzero(persons == person)}
    bunch.splits["recommended"] = bunch.splits[f"person{fold}"]
    return bunch


@DatasetManager.register_dataset("dopnet")
class DopNetDataset(RadarCovarianceDataset):
    """Dop-NET hand gestures (Click, Pinch, Swipe, Wave), 48 x 48 covariances.

    ``dataset[i] -> (x, label)``; ``raw`` is the ``(T, 48)`` log10 band power
    (variable ``T``).

    Parameters
    ----------
    data_dir : path, optional
    energy, p, min_frames, fold, cache_dir, keep, force, verify
        See :func:`fetch_dopnet`.
    split : str
        ``"official"``, ``"recommended"`` or ``"person<k>"``. For backward
        compatibility, ``"train"`` / ``"test"`` select that part of
        ``"official"`` (``subset``); ``"all"`` is ``"official"``.
    subset : {None, "train", "val", "test"}
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    def __init__(self, data_dir=None, *, energy=0.95, p=48, min_frames=None, fold=0,
                 split="official", subset=None, cache_dir=None, keep=None, force=False,
                 verify="size", estimator="scm", **kwargs):
        if split in ("train", "test"):
            split, subset = "official", split
        elif split == "all":
            split = "official"
        root = dataset_path(DOMAIN, NAME, data_dir)
        bunch = fetch_dopnet(root, cache_dir, keep, estimator, force, verify, energy=energy, p=p,
                             min_frames=min_frames, fold=fold)
        super().__init__(bunch, split=split, subset=subset, data_dir=root,
                         estimator=estimator, **kwargs)
