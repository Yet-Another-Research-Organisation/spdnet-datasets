"""Rices_90: VNIR cubes of 90 Vietnamese rice varieties, one covariance per grain.

180 ENVI cubes (BIL, uint16 on 12 bits, 336 columns x 921-932 lines x 256
bands, 383-1006 nm): 90 varieties x 2 bundles of about 48 grains laid out on a
grid, acquired over 9 sessions (``index.csv``: variety, bundle, session
folder, file). Each session has a dark reference ``black.raw``; each cube
holds a white reference panel under the grains.

Construction (atlas recipe, as the former ``series_rices``):

1. Dark reference of the session subtracted; relative reflectance against the
   white panel (rows above the panel only).
2. Grains = connected components of an Otsu mask of the mean reflectance
   (after an opening) whose area lies in [0.4, 2] x the median area.
3. Bands averaged by groups of ``band_bin`` (default 4): p = 64. With 256
   bands the median grain has n / p = 1.7 < 2 (QUESTIONS A2); with 64 bands
   the smallest grain (about 200 pixels) still has n >= 2p. Grains with fewer
   than 2p pixels are discarded and counted.

Splits (``groups`` = cube, i.e. variety and bundle):

- ``"bundle"`` = ``"recommended"``: train = grains of bundle 1 (20 % of them,
  drawn at random, as validation: same cubes as train), test = bundle 2;
- ``"bundle_reverse"``: the other way round;
- ``"cross_session"``: as ``"bundle"``, but the test keeps only the varieties
  whose two bundles come from two sessions (17 varieties). For 73 varieties
  both bundles share their session, so variety and session are confounded:
  only this test is not.

Download (manual): no public mirror identified (obtained from the authors). Copy the
``Data-VIS-<session>`` folders and place it under ``data_dir`` (default ``<data
root>/HYPERSPECTRAL/Rices_90``).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import dataset_path
from ..core.fetch import Bunch, run_pipeline, verify_local
from ..manager import DatasetManager
from ._common import (
    HSIDataset,
    covariance_stage,
    load_covariance_dir,
    map_splits,
    write_json,
)

DOMAIN, NAME = "HYPERSPECTRAL", "Rices_90"
FILES = {}  # local copy (no public mirror identified)
N_BANDS = 256


def read_envi(header):
    """ENVI BIL uint16 cube -> (lines, samples, bands) memory-mapped view."""
    fields = {}
    for line in open(header, errors="ignore"):
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip().lower()] = value.strip()
    lines, samples, bands = int(fields["lines"]), int(fields["samples"]), int(fields["bands"])
    if fields.get("interleave") != "bil" or fields.get("data type") != "12":
        raise ValueError(f"Unsupported ENVI layout in {header}")
    raw = np.memmap(str(header)[:-4] + ".raw", dtype="<u2", mode="r",
                    shape=(lines, bands, samples))
    return raw.transpose(0, 2, 1)


def otsu_threshold(values, bins=256):
    """Threshold maximising the between-class variance of the histogram (Otsu, 1979)."""
    hist, edges = np.histogram(values, bins)
    centres = (edges[:-1] + edges[1:]) / 2
    weight, moment = np.cumsum(hist), np.cumsum(hist * centres)
    between = (moment[-1] * weight / weight[-1] - moment) ** 2 / np.maximum(
        weight * (weight[-1] - weight), 1)
    return float(edges[np.argmax(between) + 1])


def segment_grains(cube, dark):
    """(lines, samples, B) cube -> list of (n_pixels, B) relative reflectances of the grains."""
    from scipy import ndimage

    cube = np.asarray(cube, dtype=np.float32) - dark[None]
    profile = np.median(cube.mean(-1), 1)
    rows = np.nonzero(profile > 0.5 * profile.max())[0]  # white reference panel
    first, last = int(rows.min()), int(rows.max())
    white = cube[np.arange(first + 5, max(first + 6, last - 4))].mean(0)
    cube = cube[:max(first - 8, 10)] / np.maximum(white[None], 1.0)
    brightness = cube.mean(-1)
    mask = ndimage.binary_opening(brightness > otsu_threshold(brightness.ravel()), iterations=1)
    labels, n = ndimage.label(mask)
    areas = ndimage.sum(mask, labels, range(1, n + 1))
    median = np.median(areas[areas > 30]) if (areas > 30).any() else 0
    objects = ndimage.find_objects(labels)
    return [cube[objects[i]][labels[objects[i]] == i + 1]
            for i in range(n) if 0.4 * median <= areas[i] <= 2.0 * median]


def fetch_rices90(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                  verify="size", *, band_bin=4, val_fraction=0.2, seed=0):
    """Grain covariances of Rices_90.

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means``, ``labels``
    (variety), ``groups`` (cube), ``meta`` (``cube``, ``bundle``,
    ``session``, ``n_pixels``...), ``info`` and ``splits``.
    """
    if N_BANDS % band_bin:
        raise ValueError(f"band_bin must divide {N_BANDS}")
    root = dataset_path(DOMAIN, NAME, data_dir)
    verify_local(root, verify=verify, required=("index.csv",))
    p = N_BANDS // band_bin
    params = {"band_bin": band_bin, "estimator": estimator, "version": 2}

    def process(_, out_dir):
        index = pd.read_csv(root / "index.csv")
        darks, rows, chunks, short, offset = {}, [], [], 0, 0
        for rec in index.itertuples(index=False):
            variety, bundle, folder, name = rec[0], int(rec[2]), rec[3], rec[4]
            if folder not in darks:
                black = read_envi(root / folder / "black.hdr")
                darks[folder] = np.asarray(black, np.float32).mean(0)
            grains = segment_grains(read_envi(root / folder / f"{name}.hdr"), darks[folder])
            for g, pixels in enumerate(grains):
                if len(pixels) < 2 * p:
                    short += 1
                    continue
                binned = pixels.reshape(len(pixels), p, band_bin).mean(-1).astype(np.float32)
                chunks.append(binned)
                rows.append({"label": variety, "bundle": bundle, "session": folder,
                             "cube": name, "grain": g, "start": offset,
                             "n_pixels": len(binned)})
                offset += len(binned)
        np.save(Path(out_dir) / "pixels.npy", np.concatenate(chunks))
        pd.DataFrame(rows).to_csv(Path(out_dir) / "meta.csv", index=False)
        write_json(Path(out_dir) / "info.json", {
            "n_cubes": len(index), "n_grains": len(rows) + short, "excluded_n_lt_2p": short,
            "p": p})

    def pixels(proc_dir, meta):
        data = np.load(Path(proc_dir) / "pixels.npy", mmap_mode="r")
        for start, n in zip(meta["start"], meta["n_pixels"], strict=True):
            yield np.asarray(data[start:start + n])

    res = run_pipeline(NAME, params=params, process_fn=process,
                       covariance_fn=lambda pr, o: covariance_stage(pr, o, pixels, estimator),
                       cache_dir=cache_dir, keep=keep, force=force)
    out = load_covariance_dir(res.covariance_dir)
    meta = out.meta
    return Bunch(
        covariances=out.covariances, means=out.means, n_obs=out.n_obs,
        labels=meta["label"].to_numpy(), groups=meta["cube"].to_numpy(), meta=meta,
        info=out.info,
        splits=map_splits(bundle_splits(out.meta_all, val_fraction, seed), meta["proc_index"],
                          len(out.meta_all)),
        processed_path=res.processed_dir, covariance_path=res.covariance_dir,
    )


def bundle_splits(meta, val_fraction=0.2, seed=0):
    """Bundle 1 -> 2, 2 -> 1 and cross-session splits (see the module docstring)."""
    rng = np.random.default_rng(seed)
    bundle = meta["bundle"].to_numpy()
    sessions = meta.groupby("label")["session"].nunique()
    cross = meta["label"].map(sessions).to_numpy() > 1

    def one_way(train_bundle):
        train = np.flatnonzero(bundle == train_bundle)
        val = np.sort(rng.choice(train, int(round(val_fraction * len(train))), replace=False))
        return {"train": np.setdiff1d(train, val), "val": val,
                "test": np.flatnonzero(bundle != train_bundle)}

    forward = one_way(1)
    cross_session = dict(forward, test=forward["test"][cross[forward["test"]]])
    return {"bundle": forward, "recommended": forward, "bundle_reverse": one_way(2),
            "cross_session": cross_session}


@DatasetManager.register_dataset("rices90")
class Rices90Dataset(HSIDataset):
    """Rices_90 grains (90 varieties), 64 x 64 covariances, bundle 1 -> 2 split.

    Parameters
    ----------
    band_bin, val_fraction
        See :func:`fetch_rices90`.
    split : str
        ``"recommended"`` (bundle 1 -> 2), ``"bundle_reverse"`` or
        ``"cross_session"``.
    subset
        See :class:`~spdnet_datasets.hyperspectral._common.HSIDataset`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    def __init__(self, data_dir=None, *, band_bin=4, val_fraction=0.2, split="recommended",
                 subset=None, cache_dir=None, keep=None, force=False, verify="size",
                 **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(data_dir=str(root), **kwargs)
        bunch = fetch_rices90(root, cache_dir, keep, self.estimator, force, verify,
                              band_bin=band_bin, val_fraction=val_fraction)
        self._setup(bunch, split, subset)

    def _load_raw(self, k):
        """Binned pixels of the grain, (n_pixels, p)."""
        proc = self.bunch.processed_path
        if proc is None:
            raise ValueError("return_raw needs the processed stage (keep != 'keep_covariance')")
        data = np.load(os.path.join(proc, "pixels.npy"), mmap_mode="r")
        row = self.meta_all.iloc[k]
        return data[int(row["start"]):int(row["start"]) + int(row["n_pixels"])]


__all__ = ["Rices90Dataset", "fetch_rices90", "read_envi", "segment_grains"]
