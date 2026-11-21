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

Download
--------
Two sources, chosen with ``source=``:

- ``source="local"`` (default), access **manual**: the 2 086 files
  ``{id}_{frames}_{class}.npy`` read above are a preprocessed export whose
  construction is not published (no public copy, no checksum); the raw motion
  database (C3D / ASF-AMC files, CC BY-SA 3.0) is at
  https://resources.mpi-inf.mpg.de/HDM05/. Resolution order: ``data_dir``
  (default ``<data root>/SKELETON/HDM05``) if it exists; else ``archive=`` (a
  folder, zip or tar of the ``.npy`` files, extracted into
  ``<cache_dir>/hdm05/downloads/``); else the page is opened and
  :class:`~spdnet_datasets.core.access.ManualDownloadRequired` is raised. An
  unofficial ``data.zip`` (Dropbox, cited by SPDNetBN / LieBN code) holds a
  ``data/hdm05.tgz`` that may be the origin of this export; it was not checked.
- ``source="eth"``, access **direct**: the SPDNet matrices of Huang & Van Gool
  linked from https://github.com/zzhiwu/SPDNet, ``HDM05_SPDData.zip`` (603 MB,
  :data:`ETH_URL`; no published checksum, the sha256 of :data:`FILES` was
  computed on the file downloaded on 2026-11-15; licence of the matrices not
  stated). It is downloaded into ``<cache_dir>/hdm05/downloads/``, checked,
  extracted and deleted; ``archive=`` may give the zip (or its extracted
  folder) instead. Content: ``metadata.mat`` (``spd.name``, ``spd.label``,
  ``spd.set``) and ``feature/<action>/<k>[_<s>].mat`` holding one 93 x 93
  single-precision matrix ``Y1``: 19 491 files in 130 action folders.
  ``metadata.mat`` lists 19 344 names (``spd93_half_inter9\\<action>\\<file>``,
  which suggests overlapping half-sequence segments) but only 9 467 of them are
  files of the zip: it does not index this folder. Its ``set`` value (1 or 2) is
  copied as is into ``meta`` (NaN for the other files) and not used.
  **This is not the local export**: other counts (19 491 vs 2 086), classes (130
  vs 117) and conditioning (median 8.4e2 vs 9.6e5), and no local matrix is close
  to any ETH matrix. The ``.mat`` files are read directly (label = action
  folder); ``n`` is unknown, so only the SPD check applies; the variable order is
  unknown, so only ``features="c93"``; groups are ``<action>/<k>`` (segments of
  one sequence) and the split is a stratified group 5-fold.

Example: ``fetch_hdm05(source="eth")`` or ``fetch_hdm05(archive="~/hdm05_npy.zip")``.
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
    downloads_dir,
    evenly_spaced,
    exclusion,
    folds_to_split,
    ingest_archive,
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
    download,
    run_pipeline,
)
from ..manager import DatasetManager

DOMAIN, NAME = "SKELETON", "HDM05"
ETH_URL = "https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/SPDData/HDM05_SPDData.zip"
ETH_SIZE = 603358372
FILES = {  # url -> sha256 (computed on the download of 2026-11-15; none is published)
    ETH_URL: "e516cc983f154304f95b1a37a9858fb60f41bfdef8ab432a8d827a92c3b40a58",
}
PAGE = "https://resources.mpi-inf.mpg.de/HDM05/"
SOURCES = ("local", "eth")
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


def eth_index(root):
    """Matrices of the ETH ``HDM05_SPDData`` folder (``metadata.mat`` + ``feature/``).

    One row per ``feature/<action>/<name>.mat`` file: ``id`` ``<action>/<name>``,
    ``label`` the action, ``group`` ``<action>/<k>`` (``<name>`` is ``<k>`` or
    ``<k>_<s>``), ``set`` from ``metadata.mat`` (NaN if the file is not listed),
    ``n_obs`` NaN (unknown) and ``file`` (relative to ``root``).
    """
    from scipy.io import loadmat

    root = Path(root)
    spd = loadmat(root / "metadata.mat", squeeze_me=True, struct_as_record=False)["spd"]
    listed = {tuple(str(n).split("\\")[-2:]): int(s)
              for n, s in zip(np.atleast_1d(spd.name), np.atleast_1d(spd.set), strict=True)}
    rows = []
    for path in sorted((root / "feature").glob("*/*.mat")):
        action, name = path.parent.name, path.name
        rows.append({"id": f"{action}/{path.stem}", "label": action,
                     "group": f"{action}/{path.stem.split('_')[0]}",
                     "set": listed.get((action, name), np.nan), "n_obs": np.nan,
                     "file": f"feature/{action}/{name}"})
    if not rows:
        raise FileNotFoundError(f"No feature/<action>/*.mat file in {root}")
    return pd.DataFrame(rows)


def download_hdm05_eth(cache_dir=None, archive=None):
    """Folder of ``metadata.mat`` from the ETH zip (downloaded, or ``archive``).

    The zip goes to ``<cache_dir>/hdm05/downloads/``, is checked against
    :data:`FILES`, extracted once into ``HDM05_SPDData/`` and deleted; a
    user-provided ``archive`` is checked and extracted but never deleted.
    """
    from ..core.access import extract_archive, manual_source

    dest = downloads_dir(cache_dir, "hdm05")
    out = dest / "eth"
    if archive is not None:
        root = manual_source(NAME, page_url=ETH_URL, expected="HDM05_SPDData.zip",
                             dest_dir=out, archive=archive, checksum=FILES[ETH_URL],
                             size=ETH_SIZE)
    elif (out / ".extracted").exists():
        root = out
    else:
        zipped = download(ETH_URL, dest / "HDM05_SPDData.zip", checksum=FILES[ETH_URL],
                          size=ETH_SIZE)
        root = extract_archive(zipped, out)
        zipped.unlink()
    found = sorted(Path(root).rglob("metadata.mat"))
    if not found:
        raise FileNotFoundError(f"No metadata.mat in {root}")
    return found[0].parent


def c31(C):
    """Sum of the three per-coordinate blocks of a coordinate-major 93 x 93 matrix."""
    q = C.shape[0] // 3
    return sum(C[d * q:(d + 1) * q, d * q:(d + 1) * q] for d in range(3))


def _process(source, out_dir, *, root, limit, eth=False):
    index = eth_index(root) if eth else _index(root)
    index.iloc[evenly_spaced(len(index), limit)].to_csv(Path(out_dir) / "index.csv",
                                                        index=False)


def _covariances(proc_dir, out_dir, *, root, features):
    index = pd.read_csv(Path(proc_dir) / "index.csv",
                        dtype={"id": str, "label": str, "group": str}, **CSV)
    covs, rows, excluded = [], [], []
    for i, rec in enumerate(index.itertuples(index=False)):
        if rec.file.endswith(".mat"):
            from scipy.io import loadmat

            C = loadmat(Path(root) / rec.file)["Y1"].astype(np.float64)
        else:
            C = np.load(Path(root) / rec.file).astype(np.float64)
        if features == "c31":
            C = c31(C)
        try:
            if not np.isnan(rec.n_obs):
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
    """Stratified 5-fold split (no actor information; see module docstring).

    Grouped by sequence for the ETH matrices (``set`` column in ``meta``).
    """
    grouped = groups if "set" in meta else None
    parts = folds_to_split(kfold_assignment(labels, grouped, N_FOLDS, seed), fold, N_FOLDS)
    return {"stratified": parts, "recommended": parts}


def fetch_hdm05(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                verify="size", features="c93", limit=None, fold=1, source="local",
                archive=None):
    """Covariances of the HDM05 sequences.

    Parameters
    ----------
    data_dir : path, optional
        Read-only copy (default ``<data root>/SKELETON/HDM05``); ignored with
        ``source="eth"``.
    estimator : {"scm"}
        The matrices are provided; any other value raises ``ValueError``.
    features : {"c93", "c31"}
        See the module docstring.
    limit : int, optional
        At most ``limit`` sequences, evenly spread.
    fold : int
        Test fold (1..5) of the returned ``splits``.
    source : {"local", "eth"}
        The local export or the ETH SPDNet matrices (see "Download").
    archive : path, optional
        Downloaded archive or folder (see "Download").
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
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if source == "eth":
        if features != "c93":
            raise ValueError("The variable order of the ETH matrices is unknown: only "
                             "features='c93' is available with source='eth'")
        root = download_hdm05_eth(cache_dir, archive)
    else:
        root = dataset_path(DOMAIN, NAME, data_dir)
        if root.is_dir():
            check_local(root, verify=verify)
        else:
            root = ingest_archive(
                NAME, archive, downloads_dir(cache_dir, "hdm05") / "local", "*_*_*.npy",
                page_url=PAGE, expected="the export of 2 086 files {id}_{frames}_{class}.npy "
                "(not published; or call fetch_hdm05(source='eth') for the SPDNet matrices)")
    params = {"data_dir": str(root.resolve()), "features": features, "limit": limit,
              "source": source}
    run = run_pipeline(
        "hdm05", params=params,
        process_fn=partial(_process, root=root, limit=limit, eth=source == "eth"),
        covariance_fn=partial(_covariances, root=root, features=features),
        cache_dir=cache_dir, keep=keep, force=force)
    return load_result(run, hdm05_splits, fold=fold)


@DatasetManager.register_dataset("hdm05")
class HDM05Dataset(SequenceCovarianceDataset):
    """HDM05 covariances (see :mod:`spdnet_datasets.skeleton.hdm05`).

    Parameters
    ----------
    features : {"c93", "c31"}
    source : {"local", "eth"}
        See :func:`fetch_hdm05`.
    **kwargs
        :class:`~spdnet_datasets.audio._common.SequenceCovarianceDataset`
        arguments; ``classes``, ``max_classes`` and ``max_samples_per_class``
        are honoured.

    Emits a ``UserWarning`` about the actor leakage.
    """

    DOMAIN, NAME = DOMAIN, NAME
    OUTPUTS = ("cov",)

    def __init__(self, data_dir=None, *, features="c93", source="local", **kwargs):
        self.features, self.source = features, source
        self.DOWNLOADABLE = source == "eth"
        warnings.warn(ACTOR_WARNING, UserWarning, stacklevel=2)
        super().__init__(data_dir, **kwargs)

    def _fetch(self):
        return fetch_hdm05(self.data_dir, features=self.features, fold=self.fold,
                           source=self.source, **self.fetch_kwargs)

    series_samples = None  # the copies hold covariances only: nothing to recover

    def structured_raw(self, i):
        """Not available: the local and ETH copies hold covariances only.

        The covariance series need the joint trajectories, i.e. the raw C3D / ASF-AMC files of
        the MPI database (https://resources.mpi-inf.mpg.de/HDM05/), which no loader reads yet.
        """
        raise ValueError(
            "HDM05 covariance series need the raw MPI motion capture files (C3D / ASF-AMC, "
            "https://resources.mpi-inf.mpg.de/HDM05/): the local export and the ETH archive "
            f"(source={self.source!r}) only hold the 93 x 93 covariances, without the "
            "trajectories.")

    def _make_splits(self, meta, labels, groups):
        return hdm05_splits(meta, labels, groups, fold=self.fold, seed=0)


__all__ = ["HDM05Dataset", "c31", "download_hdm05_eth", "eth_index", "fetch_hdm05",
           "hdm05_splits"]
