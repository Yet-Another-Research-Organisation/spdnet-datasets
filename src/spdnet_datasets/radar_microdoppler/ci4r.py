"""CI4R-MULTI3: human activities seen by co-located radars (77 GHz and xethru).

Source
------
Laboratory of Computational Intelligence for Radar (S. Z. Gurbuz),
https://github.com/ci4r/CI4R-Activity-Recognition-datasets. Eleven activities
recorded simultaneously by a TI IWR1443 77 GHz frequency-modulated continuous
wave (FMCW) radar, an Ancortek 24 GHz FMCW radar and a XeThru X4 ultra-wideband
(UWB) radar. Only micro-Doppler spectrograms rendered as jet-coloured PNG
images (no axes, no physical scale) are distributed. Only the 77 GHz and
xethru modes are provided here (the 24 GHz mode is out of scope).

Construction
------------
1. The jet colour map is inverted (nearest colour of the 256-entry jet table,
   value in [0, 1]); the grey margins of the xethru images are cropped.
2. Rows are flipped so that row 0 is the most negative Doppler bin; the
   central ``band`` fraction of the Doppler axis is kept and averaged into
   ``p`` contiguous bins: ``X`` has shape ``(T, p)`` (time columns x bins).
3. One covariance over the ``T`` time columns (``n = T``, 1 475 for 77 GHz,
   1 176 for xethru, so ``n >= 2p`` always holds).

Defaults: 77 GHz ``band=0.5, p=64`` (SCM, all SPD but one sample with a
constant bin, rejected and counted); xethru ``band=0.2, p=32`` with the
Ledoit-Wolf estimator: most xethru pixels sit on a constant background colour,
so many bins never vary and the SCM is singular for 23 % of the recordings
(85 % with a 50 % band). With ``estimator="scm"`` on xethru, the samples with
a constant bin are rejected first (count in ``info["rejected_constant_channel"]``),
but other recordings have varying yet collinear bins and the fetch raises
:class:`~spdnet_datasets.core.NotSPDError`: keep Ledoit-Wolf for xethru.

Splits and leakage
------------------
No official split. The participant code is not documented; it is inferred
from the file name (characters 7-8, code ``00`` merged with ``14``, same date),
as in the data atlas. ``splits["fold<k>"]`` is a ``GroupKFold(n_folds=4)``
over participants (leave-one-subject-out on 77 GHz, which has 4 codes);
``"recommended"`` is fold ``fold``. Validation sets are empty (only 4 to 7
participants). Exact duplicates (``(copy)``/``(another copy)`` files, md5
identical: 7 in 77 GHz, 19 in xethru) are removed. In xethru, codes 05, 11
and 18 come from earlier sessions (February 2019) with few classes:
participant and class are partly confounded.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import dataset_path
from ..core.fetch import run_pipeline
from ..manager import DatasetManager
from ._common import (
    RadarCovarianceDataset,
    check_local,
    covariance_stage,
    group_folds,
    load_bunch,
    write_processed,
)

DOMAIN, NAME = "RADAR_DOPPLER", "CI4R-MULTI3"
FILES = {}  # not downloadable: manual copy (see the dataset page)
MODES = {"77GHz": "77GHz", "xethru": "Xethru"}  # mode -> image sub-directory
DEFAULTS = {
    "77GHz": {"band": 0.5, "p": 64, "estimator": "scm"},
    "xethru": {"band": 0.2, "p": 32, "estimator": "ledoit_wolf"},
}
CLASSES = ["Away", "Bend", "Crawl", "Kneel", "Limp", "Pick", "SStep", "Scissor", "Sit",
           "Toes", "Towards"]
_NAME_RE = re.compile(r"^(\d{8})_(\d+)")

# Piecewise-linear definition of the jet colour map (matplotlib's ``jet``).
_JET = {"r": [(0, 0), (0.35, 0), (0.66, 1), (0.89, 1), (1, 0.5)],
        "g": [(0, 0), (0.125, 0), (0.375, 1), (0.64, 1), (0.91, 0), (1, 0)],
        "b": [(0, 0.5), (0.11, 1), (0.34, 1), (0.65, 0), (1, 0)]}


def _resolve_mode(mode):
    for m in MODES:
        if m.lower() == str(mode).lower():
            return m
    raise ValueError(f"Unknown mode {mode!r}; choose among {tuple(MODES)}")


def jet_to_scalar(rgb, max_distance=40.0):
    """Invert a jet colour map: ``(H, W, 3)`` uint8 -> ``(H', W')`` values in [0, 1].

    Each colour takes the index of the nearest entry of the 256-entry jet
    table. Border rows and columns whose pixels are mostly farther than
    ``max_distance`` from the table (grey margins) are cropped.
    """
    from scipy.spatial import cKDTree

    x = np.linspace(0, 1, 256)
    table = np.stack([np.interp(x, *zip(*_JET[c], strict=True)) for c in "rgb"], 1) * 255
    rgb = np.asarray(rgb)[..., :3].astype(np.int64)
    code = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    colours, inverse = np.unique(code.ravel(), return_inverse=True)
    points = np.stack([(colours >> 16) & 255, (colours >> 8) & 255, colours & 255], 1)
    dist, index = cKDTree(table).query(points.astype(np.float64))
    values = (index / 255.0)[inverse].reshape(code.shape)
    far = (dist > max_distance)[inverse].reshape(code.shape)
    rows = np.flatnonzero(far.mean(1) <= 0.5)
    cols = np.flatnonzero(far.mean(0) <= 0.5)
    return values[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def spectrogram_features(values, band, p):
    """Central ``band`` of the Doppler rows averaged into ``p`` bins -> ``(T, p)``."""
    values = values[::-1]  # image row 0 is the top (highest Doppler)
    height = values.shape[0]
    h = int(round(height * band))
    if h < p:
        raise ValueError(f"band {band} keeps {h} Doppler rows, fewer than p={p}")
    r0 = (height - h) // 2
    rows = values[r0:r0 + h]
    return np.stack([b.mean(0) for b in np.array_split(rows, p, axis=0)], axis=1)


def participant_of(filename):
    """Participant code (characters 7-8 of the file name); ``00`` is merged with ``14``."""
    code = Path(filename).name[6:8]
    return "14" if code == "00" else code


def index_ci4r(data_dir=None, mode="77GHz"):
    """List the images of one mode with labels, participants and md5 duplicates.

    Returns a DataFrame (one row per PNG file, ``duplicate=True`` for exact
    copies of an earlier file: original names are kept before ``copy`` names).
    """
    mode = _resolve_mode(mode)
    root = dataset_path(DOMAIN, NAME, data_dir) / "11_class_activity_data" / MODES[mode]
    if not root.is_dir():
        raise FileNotFoundError(f"CI4R image directory not found: {root}")
    files = sorted(root.glob("*/*.png"), key=lambda f: ("copy" in f.name, f.parent.name, f.name))
    rows, seen = [], set()
    for f in files:
        md5 = hashlib.md5(f.read_bytes()).hexdigest()
        m = _NAME_RE.match(f.name)
        rows.append({
            "path": str(f.relative_to(root.parents[1])), "class_name": f.parent.name,
            "participant": participant_of(f.name), "timestamp": int(m.group(2)[:10]) if m else -1,
            "md5": md5, "duplicate": md5 in seen,
        })
        seen.add(md5)
    meta = pd.DataFrame(rows)
    unknown = set(meta.class_name) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unexpected CI4R class folders: {sorted(unknown)}")
    return meta.sort_values("path").reset_index(drop=True)


def fetch_ci4r(data_dir=None, cache_dir=None, keep=None, estimator=None, force=False,
               verify="size", *, mode="77GHz", band=None, p=None, n_folds=4, fold=0):
    """Covariances of the CI4R-MULTI3 spectrograms (77 GHz or xethru).

    Parameters
    ----------
    data_dir : path, optional
        Read-only copy of ``CI4R-MULTI3`` (default ``<data root>/RADAR_DOPPLER/CI4R-MULTI3``).
    cache_dir, keep, force
        See :func:`spdnet_datasets.core.run_pipeline`.
    estimator : str, optional
        Covariance estimator; default per mode (``DEFAULTS``).
    verify : {"size", "md5", None}
        Check of the local copy against ``MANIFEST_NAS.tsv``.
    mode : {"77GHz", "xethru"}
    band : float, optional
        Central fraction of the Doppler axis kept (default per mode).
    p : int, optional
        Number of Doppler bins, i.e. covariance size (default per mode).
    n_folds, fold : int
        Participant-grouped K-fold; ``splits["recommended"]`` is fold ``fold``.

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means`` (B, p),
    ``labels``, ``class_names``, ``groups`` (participant), ``meta``,
    ``info`` (counts), ``splits``, ``covariance_path`` and ``processed_path``.
    """
    mode = _resolve_mode(mode)
    defaults = DEFAULTS[mode]
    band = defaults["band"] if band is None else band
    p = defaults["p"] if p is None else p
    estimator = estimator or defaults["estimator"]
    root = dataset_path(DOMAIN, NAME, data_dir)
    check_local(root, verify=verify, required=(f"11_class_activity_data/{MODES[mode]}",))

    def process_fn(_source, out_dir):
        from PIL import Image

        meta = index_ci4r(root, mode)
        kept = meta[~meta.duplicate].reset_index(drop=True)
        spectrograms = [
            spectrogram_features(jet_to_scalar(np.asarray(Image.open(root / f).convert("RGB"))),
                                 band, p).astype(np.float32)
            for f in kept.path]
        kept["label"] = kept.class_name.map(CLASSES.index)
        kept["group"] = kept.participant
        info = {"mode": mode, "n_files": len(meta), "duplicates_removed": int(meta.duplicate.sum()),
                "band": band, "p": p}
        write_processed(out_dir, spectrograms, kept, info)

    params = {"mode": mode, "band": band, "p": p, "estimator": estimator, "version": 1}
    paths = run_pipeline(f"ci4r_{mode}", params=params, process_fn=process_fn,
                         covariance_fn=covariance_stage(estimator), cache_dir=cache_dir,
                         keep=keep, force=force)
    bunch = load_bunch(paths, CLASSES)
    bunch.splits = group_folds(bunch.groups, n_folds, bunch.labels)
    bunch.splits["recommended"] = bunch.splits[f"fold{fold}"]
    return bunch


@DatasetManager.register_dataset("ci4r")
class CI4RDataset(RadarCovarianceDataset):
    """CI4R-MULTI3 activity recognition, 77 GHz or xethru (11 classes).

    ``dataset[i] -> (x, label)`` with ``x`` the ``p x p`` covariance (float64),
    or a tuple with the channel mean ``(p,)`` and the preprocessed spectrogram
    ``(T, p)`` (float32) when ``return_mean`` / ``return_raw`` are set.

    Parameters
    ----------
    data_dir : path, optional
    mode : {"77GHz", "xethru"}
        ``modality`` is accepted as a deprecated alias.
    band, p, n_folds, fold, cache_dir, keep, force, verify
        See :func:`fetch_ci4r`.
    split : str
        Split fitting the normalisation (``"recommended"`` or ``"fold<k>"``).
    subset : {None, "train", "val", "test"}
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options (``estimator``,
        ``task``, ``classes``, ``balance``, ``normalize``, ...).
    """

    def __init__(self, data_dir=None, *, mode="77GHz", band=None, p=None, n_folds=4, fold=0,
                 split="recommended", subset=None, cache_dir=None, keep=None, force=False,
                 verify="size", estimator=None, modality=None, **kwargs):
        if modality is not None:
            mode = modality
        mode = _resolve_mode(mode)
        estimator = estimator or DEFAULTS[mode]["estimator"]
        root = dataset_path(DOMAIN, NAME, data_dir)
        self.mode = mode
        bunch = fetch_ci4r(root, cache_dir, keep, estimator, force, verify, mode=mode,
                           band=band, p=p, n_folds=n_folds, fold=fold)
        super().__init__(bunch, split=split, subset=subset, data_dir=root,
                         estimator=estimator, **kwargs)
