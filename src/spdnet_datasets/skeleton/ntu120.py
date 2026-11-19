"""NTU RGB+D 120: 3D skeleton action recognition (Liu et al., 2020).

114 480 Kinect v2 skeleton sequences (25 joints, camera coordinates in metres,
30 fps) of 120 actions, 106 subjects, 32 setups, 3 cameras, 2 replications.
The local copy holds the skeletons extracted as ``.npy`` dictionaries
(``skel_body0``: (T, 25, 3), ``nbodys``, ...) under
``npy_skeletons_by_class/A<action>/S<setup>C<camera>P<subject>R<rep>A<action>.npy``.

Preprocessing
-------------
Body 0 only; frames where the body is lost (all coordinates zero) are dropped,
``n`` is the number of remaining frames (15 to 300, median 66). The 356 files
without any body (``nbodys`` = 0) are excluded, as are sequences with ``n < 2p``
(all listed in ``excluded``). The official list of the 535 defective samples is
not available in the copy.

- ``features="parts"`` (default, ``p = 15``): per frame, mean root-centred
  coordinates of five body parts of four joints (trunk, left arm, right arm,
  left leg, right leg; :data:`PARTS`), as the atlas sheet "C15". ``n >= 30``
  keeps about 97 % of the sequences.
- ``features="joints"`` (``p = 75``): the 25 x 3 raw coordinates, as the former
  ``utils/ntu120_covariance_precompute.py``. ``n >= 150`` keeps only about
  2.5 % of the sequences (most are shorter than 5 s).

Splits
------
- ``"xsub"`` (official cross-subject, = ``"recommended"`` and ``"official"``):
  the 53 training subjects of Liu et al. (2020) (:data:`TRAIN_SUBJECTS`) for
  train, the others for test. The validation set is 20 % of the training
  subjects (seeded), never random sequences: one take is filmed by 3 cameras.
- ``"xsetup"`` (official cross-setup): even setups for train, odd setups for
  test; validation = 20 % of the training setups.

Tasks: ``task="fine"`` (120 actions) or ``task="coarse"`` (daily actions,
medical conditions, mutual actions, as in the NTU RGB+D 120 paper). Mutual
actions involve two persons, of which only body 0 is used.

Download
--------
Access: **manual**. Request access on
https://rose1.ntu.edu.sg/dataset/actionRecognition/ (ROSE Lab account, form and
Release Agreement, approval by hand; academic non-commercial use, no
redistribution). Download the 3D skeletons ``nturgbd_skeletons_s001_to_s017.zip``
(NTU RGB+D 60, about 5.8 GB) and ``nturgbd_skeletons_s018_to_s032.zip`` (the
extension to 120, about 4.5 GB); no checksum is published. The copies re-published
elsewhere (figshare, pre-processed pickles) go against the ROSE Lab terms and are
not used.

The zips hold one text file ``S<setup>C<camera>P<subject>R<rep>A<action>.skeleton``
per sequence (format of the official ``read_skeleton`` code of
https://github.com/shahroudy/NTURGB-D: number of frames, then per frame the
number of bodies and, per body, a line of 10 body values, the number of joints
(25) and one line of 12 values per joint, the first three being the camera
coordinates x, y, z in metres). :func:`read_skeleton` turns one file into the
dictionary of the local ``.npy`` copy (``file_name``, ``nframes``, ``njoints``,
``nbodys``, ``skel_body<k>`` of shape (nframes, 25, 3), as the official
``txt2npy.py`` with coordinates only); the ``.skeleton`` files are read
directly, without writing ``.npy`` files. Resolution order:

1. ``data_dir`` (default ``<data root>/SKELETON/NTU_RGBD_120``) exists: the
   ``npy_skeletons_by_class/`` copy (or, without it, the ``.skeleton`` files
   found under ``data_dir``) is read;
2. ``archive=`` given: one zip or a list of zips (or folders) of ``.skeleton``
   files, extracted into ``<cache_dir>/ntu120/downloads/``;
3. otherwise the ROSE Lab page is opened and
   :class:`~spdnet_datasets.core.access.ManualDownloadRequired` explains what to
   download.

Example: ``fetch_ntu120(archive=["~/Downloads/nturgbd_skeletons_s001_to_s017.zip",
"~/Downloads/nturgbd_skeletons_s018_to_s032.zip"])``.
"""

import re
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from ..audio._common import (
    SequenceCovarianceDataset,
    downloads_dir,
    estimate_stage,
    evenly_spaced,
    load_result,
    write_processed,
)
from ..core import check_local, dataset_path, run_pipeline
from ..manager import DatasetManager

DOMAIN, NAME = "SKELETON", "NTU_RGBD_120"
FILES = {}  # licensed dataset (ROSE Lab agreement): manual download only
PAGE = "https://rose1.ntu.edu.sg/dataset/actionRecognition/"
ARCHIVES = ("nturgbd_skeletons_s001_to_s017.zip", "nturgbd_skeletons_s018_to_s032.zip")
MAX_BODIES = 4
SKELETON_DIR = "npy_skeletons_by_class"
FEATURES = ("parts", "joints")
PATTERN = re.compile(r"^S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})$")
TRAIN_SUBJECTS = (1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
                  45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82,
                  83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103)
PARTS = {"trunk": (1, 20, 2, 3), "left arm": (5, 6, 7, 21), "right arm": (9, 10, 11, 23),
         "left leg": (12, 13, 14, 15), "right leg": (16, 17, 18, 19)}
"""0-based joints of the five body parts (root 0, shoulders, hips and thumbs left out)."""


def coarse_class(action):
    """Coarse category of an action ``"A001"``..``"A120"``."""
    a = int(action[1:])
    if 41 <= a <= 49 or 103 <= a <= 105:
        return "medical"
    if 50 <= a <= 60 or a >= 106:
        return "mutual"
    return "daily"


COARSE_MAP = {f"A{a:03d}": coarse_class(f"A{a:03d}") for a in range(1, 121)}


def read_skeleton(path):
    """Read one official ``.skeleton`` text file (see the module docstring).

    Returns the dictionary of the official ``txt2npy.py`` (coordinates only):
    ``file_name``, ``nframes``, ``njoints``, ``nbodys`` (number of bodies of each
    non-empty frame) and ``skel_body<k>`` (nframes, 25, 3) float64 for ``k`` below
    the largest number of bodies; frames without body ``k`` are zeros.
    """
    lines = Path(path).read_text().split("\n")
    nframes, njoints = int(lines[0]), 25
    record = {"file_name": Path(path).stem, "nbodys": [], "njoints": njoints,
              "nframes": nframes}
    skel = np.zeros((MAX_BODIES, nframes, njoints, 3))
    cursor = 1
    for frame in range(nframes):
        nbody = int(lines[cursor])
        cursor += 1
        if nbody:
            record["nbodys"].append(nbody)
        for body in range(nbody):
            cursor += 1  # body line: id, clipped edges, hand states, lean, tracking
            n = int(lines[cursor])
            cursor += 1
            for joint in range(n):
                skel[body, frame, joint] = [float(v) for v in lines[cursor + joint].split()[:3]]
            cursor += n
    for body in range(max(record["nbodys"], default=0)):
        record[f"skel_body{body}"] = skel[body]
    return record


def _load(path):
    """Skeleton record of a local ``.npy`` dictionary or of an official ``.skeleton`` file."""
    if path.suffix == ".skeleton":
        return read_skeleton(path)
    data = np.load(path, allow_pickle=True)
    return data.item() if data.ndim == 0 else {}


def ntu_index(root):
    """Sequences of the copy parsed from the file names (no file is read).

    Reads ``npy_skeletons_by_class/A*/*.npy`` or, without it, the
    ``*.skeleton`` files anywhere under ``root``. Returns a DataFrame with ``id``,
    ``label`` (``"A001"``...), ``group`` (subject), ``setup``, ``camera``,
    ``subject``, ``replication`` and ``path`` (relative to ``root``).
    """
    rows = []
    paths = sorted(Path(root, SKELETON_DIR).glob("A*/*.npy"))
    if not paths:
        paths = sorted(Path(root).rglob("*.skeleton"), key=lambda f: (f.stem[-4:], f.name))
    for path in paths:
        m = PATTERN.match(path.stem)
        if not m:
            continue
        setup, camera, subject, rep, action = map(int, m.groups())
        rows.append({"id": path.stem, "label": f"A{action:03d}", "group": subject,
                     "setup": setup, "camera": camera, "subject": subject,
                     "replication": rep, "path": str(path.relative_to(root))})
    if not rows:
        raise FileNotFoundError(f"No {SKELETON_DIR}/A*/*.npy nor *.skeleton file under {root}")
    return pd.DataFrame(rows)


def skeleton_features(record, features="parts"):
    """Frame features (n, p) of one skeleton record, or ``None`` without a body."""
    nbodys = record.get("nbodys")
    if "skel_body0" not in record or (nbodys is not None and max(nbodys, default=0) == 0):
        return None
    x = np.asarray(record["skel_body0"], dtype=np.float64)  # (T, 25, 3)
    x = x[np.any(x != 0, axis=(1, 2))]
    if features == "joints":
        return x.reshape(len(x), -1)
    centred = x - x[:, :1]
    return np.concatenate([centred[:, list(j)].mean(axis=1) for j in PARTS.values()], axis=1)


def _process(source, out_dir, *, root, limit, features):
    index = ntu_index(root)
    index = index.iloc[evenly_spaced(len(index), limit)]
    p = 15 if features == "parts" else 75
    feats, rows, excluded = [], [], []
    for rec in index.to_dict("records"):
        X = skeleton_features(_load(Path(root, rec.pop("path"))), features)
        reason = ("no body" if X is None else
                  f"n < 2p ({len(X)} frames with a body, 2p = {2 * p})" if len(X) < 2 * p else None)
        if reason:
            excluded.append({"id": rec["id"], "label": rec["label"], "reason": reason})
            continue
        feats.append(X.astype(np.float32))
        rows.append(rec)
    write_processed(out_dir, feats, rows, excluded, p)


def _val_from(train, keys, fraction, seed):
    """Move the samples of ``fraction`` of the ``keys`` values of ``train`` to validation."""
    values = np.unique(keys[train])
    rng = np.random.default_rng(seed)
    held = rng.choice(values, size=int(round(fraction * len(values))), replace=False)
    in_val = np.isin(keys[train], held)
    return train[~in_val], train[in_val]


def ntu120_splits(meta, labels, groups, val_fraction=0.2, seed=0):
    """Official X-Sub and X-Setup splits (see module docstring)."""
    idx = np.arange(len(meta))
    subject, setup = meta["subject"].to_numpy(), meta["setup"].to_numpy()
    splits = {}
    for name, keys, is_train in (("xsub", subject, np.isin(subject, TRAIN_SUBJECTS)),
                                 ("xsetup", setup, setup % 2 == 0)):
        train, val = _val_from(idx[is_train], keys, val_fraction, seed)
        splits[name] = {"train": train, "val": val, "test": idx[~is_train]}
    splits["official"] = splits["recommended"] = splits["xsub"]
    return splits


def fetch_ntu120(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                 verify="size", features="parts", limit=None, archive=None):
    """Covariances of the NTU RGB+D 120 skeleton sequences.

    Parameters
    ----------
    data_dir : path, optional
        Read-only copy (default ``<data root>/SKELETON/NTU_RGBD_120``).
    features : {"parts", "joints"}
        ``p = 15`` or ``p = 75`` (see the module docstring).
    limit : int, optional
        At most ``limit`` sequences, evenly spread over the sorted file list.
    cache_dir, keep, estimator, force, verify
        See :func:`spdnet_datasets.audio.esc50.fetch_esc50`. ``verify`` checks
        the 117 567 files of ``MANIFEST_NAS.tsv``; pass ``None`` to skip it.
    archive : path or list of paths, optional
        Official skeleton zip(s) or folder(s), used when ``data_dir`` does not
        exist (see "Download" in the module docstring).

    Returns
    -------
    Bunch
        ``covariances`` (B, p, p) unscaled, ``means``, ``labels`` (actions
        ``"A001"``...), ``groups`` (subject), ``meta``, ``excluded``,
        ``splits`` (``xsub``, ``xsetup``), ``processed_path``,
        ``covariance_path``.
    """
    if features not in FEATURES:
        raise ValueError(f"features must be one of {FEATURES}, got {features!r}")
    root = dataset_path(DOMAIN, NAME, data_dir)
    if root.is_dir():
        check_local(root, verify=verify,
                    required=(SKELETON_DIR,) if (root / SKELETON_DIR).is_dir() else ())
    else:
        root = ingest_ntu120(archive, cache_dir)
    params = {"data_dir": str(root.resolve()), "features": features, "limit": limit,
                  "estimator": estimator}
    if root == downloads_dir(cache_dir, "ntu120"):  # zips share this folder: name them
        params["archive"] = sorted(Path(a).name for a in
                                   ([archive] if isinstance(archive, (str, Path)) else archive))
    run = run_pipeline(
        "ntu120", params=params,
        process_fn=partial(_process, root=root, limit=limit, features=features),
        covariance_fn=partial(estimate_stage, estimator=estimator),
        cache_dir=cache_dir, keep=keep, force=force)
    return load_result(run, ntu120_splits)


def ingest_ntu120(archive, cache_dir=None):
    """Folder holding the official ``.skeleton`` files of ``archive``.

    ``archive`` is one extracted folder (read in place) or one or several zips,
    each extracted once into ``<cache_dir>/ntu120/downloads/<zip stem>/``.
    Without ``archive`` the ROSE Lab page is opened and
    :class:`~spdnet_datasets.core.access.ManualDownloadRequired` is raised.
    """
    from ..core.access import manual_source

    dest = downloads_dir(cache_dir, "ntu120")
    expected = " and ".join(ARCHIVES) + " (3D skeletons, .skeleton files)"
    if archive is None:
        manual_source(NAME, page_url=PAGE, expected=expected, dest_dir=dest)
    paths = [Path(a).expanduser() for a in
             ([archive] if isinstance(archive, (str, Path)) else archive)]
    if len(paths) == 1 and paths[0].is_dir():
        return paths[0]
    if any(path.is_dir() for path in paths):
        raise ValueError("archive must be one folder, or one or several zip files")
    for path in paths:
        manual_source(NAME, page_url=PAGE, expected=expected,
                      dest_dir=dest / path.name.split(".")[0], archive=path)
    return dest


@DatasetManager.register_dataset("ntu120")
class NTURGBD120Dataset(SequenceCovarianceDataset):
    """NTU RGB+D 120 covariances (see :mod:`spdnet_datasets.skeleton.ntu120`).

    Parameters
    ----------
    features : {"parts", "joints"}
    **kwargs
        :class:`~spdnet_datasets.audio._common.SequenceCovarianceDataset`
        arguments; ``split`` in ``{"xsub", "xsetup", "official",
        "recommended"}``.
    """

    DOMAIN, NAME = DOMAIN, NAME
    COARSE_MAP = COARSE_MAP

    def __init__(self, data_dir=None, *, features="parts", **kwargs):
        self.features = features
        super().__init__(data_dir, **kwargs)

    def _fetch(self):
        return fetch_ntu120(self.data_dir, features=self.features, **self.fetch_kwargs)

    def _make_splits(self, meta, labels, groups):
        return ntu120_splits(meta, labels, groups, seed=0)

    def structured_raw(self, i):
        """Skeleton of sample ``i``, ``(T, 25, 3)`` float64: camera coordinates (m) of body 0,
        frames where the body is lost dropped (the frames of the covariance), read again from
        the copy. Source of :class:`~spdnet_datasets.series.CovarianceSeries`."""
        if not hasattr(self, "_paths"):
            index = ntu_index(self.data_dir)
            self._paths = dict(zip(index["id"], index["path"], strict=True))
        record = _load(Path(self.data_dir, self._paths[self.meta["id"].iloc[i]]))
        x = np.asarray(record["skel_body0"], dtype=np.float64)
        return x[np.any(x != 0, axis=(1, 2))]


__all__ = ["COARSE_MAP", "NTURGBD120Dataset", "TRAIN_SUBJECTS", "fetch_ntu120",
           "ingest_ntu120", "ntu120_splits", "ntu_index", "read_skeleton",
           "skeleton_features"]
