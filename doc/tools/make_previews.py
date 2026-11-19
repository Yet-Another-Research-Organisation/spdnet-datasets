"""Draw the fixed previews of the datasets: chips, spectrograms, false colours, skeletons.

Run on a machine that holds the data (``$SPDNET_DATA_ROOT`` and the loader
caches of ``$SPDNET_CACHE_DIR``); the documentation build only embeds the PNG
files, so the CI never needs the datasets::

    PYTHONPATH=src python doc/tools/make_previews.py                 # every dataset
    PYTHONPATH=src python doc/tools/make_previews.py esc50 mstar     # some of them

For each dataset, ``doc/_static/previews/<name>/`` receives

- ``samples.png`` and ``samples_dark.png``: up to four examples of distinct
  classes, labelled, for the light and the dark theme (dataset page);
- ``thumb.png``: a 16:9 image without text (tiles and gallery thumbnails).

A dataset that fails (missing data, missing optional dependency) is reported
and skipped; the others are still drawn. The MOABB preview needs ``moabb``
(run it with the interpreter of an environment that has the ``moabb`` extra).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import traceback
from pathlib import Path

import numpy as np

DOC = Path(__file__).resolve().parents[1]
OUT = DOC / "_static" / "previews"
DATA = Path(os.environ.get("SPDNET_DATA_ROOT", "/DATA/DATASET"))
THEMES = {"light": {"surface": "#ffffff", "ink": "#0b0b0b", "muted": "#52514e"},
          "dark": {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7"}}
DIVERGING = ["#184f95", "#f0efec", "#b8282a"]


# ---------------------------------------------------------------------------
# Panels: each preview function returns a list of (kind, array, label)
#   kind "rgb"   array (H, W, 3) in [0, 1]
#   kind "gray"  array (H, W), already scaled to [0, 1]
#   kind "spec"  array (p, T) (low frequencies first), scaled to [0, 1]
#   kind "corr"  array (p, p) correlation in [-1, 1]
#   kind "skel"  array (T, 25, 3) joint coordinates
#   kind "lines" array (k, p) spectra of a few pixels
# ---------------------------------------------------------------------------
def _stretch(a, lo=1, hi=99):
    a = np.asarray(a, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    p0, p1 = np.percentile(finite, [lo, hi])
    return np.clip((a - p0) / (p1 - p0 + 1e-12), 0, 1)


def _false_colour(cube, fractions=(0.42, 0.25, 0.12)):
    """(H, W, p) -> RGB from three bands at the given fractions of the band axis."""
    p = cube.shape[-1]
    idx = [min(p - 1, int(round(f * (p - 1)))) for f in fractions]
    return np.stack([_stretch(cube[..., i], 2, 98) for i in idx], -1)


def _corr(c):
    c = np.asarray(c, dtype=np.float64)
    d = np.sqrt(np.clip(np.diag(c), 1e-30, None))
    return c / np.outer(d, d)


def _loader_samples(name, n=4, **kwargs):
    """``n`` samples of distinct classes from a registered loader: [(x, label)]."""
    import spdnet_datasets.real  # noqa: F401  (registers the loaders)
    from spdnet_datasets import DatasetManager

    cls = DatasetManager.DATASETS[name]
    ds = cls(**{"max_samples_per_class": 1, "max_classes": n, **kwargs})
    names = list(getattr(ds, "classes", []))
    out = []
    for i in range(min(n, len(ds))):
        x, y = ds[i]
        label = names[int(y)] if names and np.ndim(y) == 0 else str(y)
        out.append((x, str(label)))
    return out


def _raw(x):
    return x[-1] if isinstance(x, tuple) else x


# --- audio: log-mel spectrograms decoded from the parquet / wav sources ---------
def _logmel(blob, start_s=None, duration_s=None):
    from spdnet_datasets.audio._features import decode, frame_features

    x, sr = decode(blob, start_s, duration_s)
    return _stretch(frame_features(x, sr, features="logmel").T, 2, 99.5)


def _parquet_rows(files, label_col, n=4, label_map=None):
    import pyarrow.parquet as pq

    out, seen = [], set()
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(columns=["audio", label_col], batch_size=32):
            for row in batch.to_pylist():
                lab = row[label_col]
                lab = label_map(lab) if label_map else lab
                if lab in seen:
                    continue
                seen.add(lab)
                out.append((row["audio"]["bytes"], str(lab)))
                if len(out) == n:
                    return out
    return out


def esc50():
    rows = _parquet_rows(sorted((DATA / "AUDIO/ESC-50").glob("*.parquet")), "category")
    return [("spec", _logmel(b), lab) for b, lab in rows]


def gtzan():
    import pyarrow.parquet as pq

    files = sorted((DATA / "AUDIO/GTZAN").glob("*.parquet"))
    cols = pq.ParquetFile(files[0]).schema_arrow.names
    col = next(c for c in ("genre", "label", "category") if c in cols)
    rows = _parquet_rows(files[::-1], col)
    return [("spec", _logmel(b, 0, 10), lab) for b, lab in rows]


def birdset_xcm():
    files = sorted((DATA / "AUDIO/BirdSet_XCM").glob("*.parquet"))
    rows = _parquet_rows(files[:3], "ebird_code")
    return [("spec", _logmel(b, 0, 8), f"species {lab}") for b, lab in rows]


def urbansound8k():
    import csv

    root = DATA / "AUDIO/UrbanSound8K"
    seen, out = set(), []
    with open(root / "metadata/UrbanSound8K.csv") as f:
        for r in csv.DictReader(f):
            if r["class"] in seen or float(r["end"]) - float(r["start"]) < 3.5:
                continue
            seen.add(r["class"])
            blob = (root / "audio" / f"fold{r['fold']}" / r["slice_file_name"]).read_bytes()
            out.append(("spec", _logmel(blob), r["class"].replace("_", " ")))
            if len(out) == 4:
                break
    return out


# --- skeleton -------------------------------------------------------------------
def ntu120():
    s = _loader_samples("ntu120", subset=None, verify=None, features="joints", return_raw=True)
    return [("skel", np.asarray(_raw(x)).reshape(-1, 25, 3), lab) for x, lab in s]


def hdm05():
    # the provided 93 x 93 covariances, <id>_<frames>_<class>.npy (see the dataset page)
    out, seen = [], set()
    for f in sorted((DATA / "SKELETON/HDM05").glob("*.npy"), key=lambda f: int(f.stem.split("_")[0])):
        cls = f.stem.split("_")[-1]
        if cls not in seen:
            seen.add(cls)
            out.append(("corr", _corr(np.load(f)), f"class {cls}"))
        if len(out) == 4:
            return out
    return out


# --- micro-Doppler radar --------------------------------------------------------
def _spectrograms(name, **kw):
    s = _loader_samples(name, return_raw=True, **kw)
    return [("spec", _stretch(np.asarray(_raw(x)).T, 2, 99.5), lab) for x, lab in s]


def ci4r():
    return _spectrograms("ci4r", mode="77GHz")


def dopnet():
    return _spectrograms("dopnet")


def mvdoppler():
    return _spectrograms("mvdoppler")


# --- satellite image time series -----------------------------------------------
def pastis_r():
    root = DATA / "SITS/PASTIS-R/PASTIS-R"
    names = ["background", "meadow", "soft winter wheat", "corn", "winter barley",
             "winter rapeseed", "spring barley", "sunflower", "grapevine", "beet",
             "winter triticale", "winter durum wheat", "fruits, vegetables, flowers",
             "potatoes", "leguminous fodder", "soybeans", "orchard", "mixed cereal",
             "sorghum", "void"]
    out = []
    for pid in sorted(p.stem.split("_")[1] for p in (root / "DATA_S2").glob("S2_*.npy"))[:40:10]:
        s2 = np.load(root / "DATA_S2" / f"S2_{pid}.npy", mmap_mode="r")      # (T, 10, H, W)
        blue = np.asarray(s2[:, 0]).reshape(s2.shape[0], -1).mean(1)
        t = int(np.argmin(np.abs(blue - np.median(blue))))  # a typical date: no cloud, no gap
        rgb = np.stack([_stretch(s2[t, b], 2, 98) for b in (2, 1, 0)], -1)
        sem = np.load(root / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0]
        edges = (np.diff(sem, axis=0, prepend=sem[:1]) != 0) | (np.diff(sem, axis=1, prepend=sem[:, :1]) != 0)
        rgb[edges] = [1.0, 0.85, 0.2]
        cls = np.bincount(sem[(sem > 0) & (sem < 19)].ravel(), minlength=20).argmax()
        out.append(("rgb", rgb, f"patch {pid}: {names[cls]}"))
    return out


# --- hyperspectral ---------------------------------------------------------------
def _hsi_windows(name, **kw):
    s = _loader_samples(name, return_raw=True, **kw)
    return [("rgb", _false_colour(np.asarray(_raw(x))), lab) for x, lab in s]


def chikusei():
    return _hsi_windows("chikusei", verbose=False)


def gsoff():
    return _hsi_windows("gsoff", verbose=False)


def uav_hsi_crop():
    return _hsi_windows("uav", verbose=False)


def kaggle_wheat():
    return _hsi_windows("kaggle_wheat", verbose=False)


def placenta():
    return _hsi_windows("placenta", verbose=False)


def hyperleaf():
    # mode="image": the leaf restricted to bands 81, 51, 21 (about 640 / 550 / 460 nm)
    s = _loader_samples("hyperleaf", verbose=False, mode="image", bands=(81, 51, 21))
    return [("rgb", np.stack([_stretch(np.asarray(x)[c], 2, 99) for c in range(3)], -1), lab)
            for x, lab in s]


def rices90():
    s = _loader_samples("rices90", verbose=False, return_raw=True)
    out = []
    for x, lab in s:
        px = np.asarray(_raw(x))
        out.append(("lines", px[np.linspace(0, len(px) - 1, 12).astype(int)], f"variety {lab}"))
    return out


def deephsfruit():
    s = _loader_samples("deephsfruit", verbose=False)
    return [("corr", _corr(_raw(x)), lab) for x, lab in s]


def hyperpistachio():
    import spectral

    root = DATA / "HYPERSPECTRAL/HyperPistachio"
    out = []
    for level in ("Level 01", "Level 09", "Level 17", "Level 26"):
        hdr = sorted((root / level).glob("*.hdr"))[0]
        cube = np.asarray(spectral.open_image(str(hdr)).load(), dtype=np.float32)
        out.append(("rgb", _false_colour(cube, (0.43, 0.27, 0.14)), f"contamination {level.lower()}"))
    return out


# --- SAR ----------------------------------------------------------------------------
def _chips(name, channel=0, log=True, **kw):
    s = _loader_samples(name, return_raw=True, **kw)
    out = []
    for x, lab in s:
        a = np.abs(np.asarray(_raw(x), dtype=np.float64))
        a = a[..., channel] if a.ndim == 3 else a
        k = -(-max(a.shape) // 96)  # at most 96 px a side: speckle does not compress
        if k > 1:
            a = a[: a.shape[0] // k * k, : a.shape[1] // k * k]
            a = a.reshape(a.shape[0] // k, k, a.shape[1] // k, k).mean((1, 3))
        out.append(("gray", _stretch(np.log1p(a) if log else a, 1, 99.5), lab))
    return out


def mstar():
    return _chips("mstar")


def fusar_ship():
    return _chips("fusar_ship", log=False)


def atrnet_star():
    return _chips("atrnet_star")


def s1slc_cvdl():
    return _chips("s1slc_cvdl", features="amplitude")


def _polsar_rgb(name, **kw):
    s = _loader_samples(name, return_raw=True, **kw)
    out = []
    for x, lab in s:
        a = np.asarray(_raw(x), dtype=np.float64)
        if a.ndim != 3:
            raise ValueError(f"{name}: raw output is not an image grid {a.shape}")
        out.append(("rgb", np.stack([_stretch(a[..., c], 2, 98) for c in (0, 1, 2)], -1), lab))
    return out


def pol_insar_island():
    try:  # the grid layout has its own argument in recent versions
        return _polsar_rgb("pol_insar_island", raw_layout="grid")
    except TypeError:
        return _polsar_rgb("pol_insar_island")


def air_polsar_seg():
    return _polsar_rgb("air_polsar_seg")


def cpazmal():
    s = _loader_samples("cpazmal", return_raw=True)
    out = []
    for x, lab in s:
        px = np.asarray(_raw(x), dtype=np.float64)                 # (pixels, 40): HH then HV
        px = px[np.linspace(0, len(px) - 1, min(len(px), 160)).astype(int), :20]
        out.append(("spec", _stretch(px[np.argsort(px.mean(1))].T, 2, 98), lab))
    return out


# --- external and synthetic -----------------------------------------------------------
def nilearn():
    cache = Path(os.environ.get("SPDNET_CACHE_DIR", "~/.cache/spdnet_datasets")).expanduser()
    proc = sorted((cache / "nilearn_abide" / "processed").glob("*/series.npy"))[0].parent
    series, lengths = np.load(proc / "series.npy"), np.load(proc / "lengths.npy")
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    return [("corr", np.corrcoef(series[s:s + n].T), f"ABIDE subject {k + 1}")
            for k, (s, n) in enumerate(zip(starts[:4], lengths[:4], strict=True))]


def moabb():
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    X, y, _ = MotorImagery(n_classes=4).get_data(BNCI2014_001(), subjects=[1])
    classes = sorted(set(y))[:4]
    corr = {c: _corr(np.mean([t @ t.T / t.shape[1] for t in X[y == c]], 0)) for c in classes}
    mean = np.mean(list(corr.values()), 0)          # EEG channels are all correlated:
    scale = max(np.abs(v - mean).max() for v in corr.values())   # show the class contrast
    return [("corr", (corr[c] - mean) / scale, f"{c.replace('_', ' ')} − mean") for c in classes]


def synthetic():
    from spdnet_datasets.synthetic.data_generator import WishartGenerator

    gen = WishartGenerator(matrix_size=16, n_classes=4, seed=0)
    data, labels, _ = gen.generate_data(n_samples_per_class=1, conditioning=50.0, df=40,
                                        discriminant_position="large")
    return [("corr", _corr(c), f"Wishart, class {int(k)}") for c, k in zip(data, labels, strict=True)]


PREVIEWS = {f.__name__: f for f in (
    esc50, gtzan, birdset_xcm, urbansound8k, hdm05, ntu120, ci4r, dopnet, mvdoppler, pastis_r,
    chikusei, gsoff, uav_hsi_crop, kaggle_wheat, placenta, rices90, hyperleaf, deephsfruit,
    hyperpistachio, pol_insar_island, cpazmal, s1slc_cvdl, air_polsar_seg, atrnet_star,
    fusar_ship, mstar, moabb, nilearn, synthetic)}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
NTU_BONES = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
             (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
             (18, 17), (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]


def _draw(ax, kind, a, theme):
    from matplotlib.colors import LinearSegmentedColormap

    t = THEMES[theme]
    ax.set_facecolor(t["surface"])
    if kind == "rgb":
        ax.imshow(a, interpolation="nearest" if a.shape[0] < 64 else "antialiased")
    elif kind == "gray":
        ax.imshow(a, cmap="gray", interpolation="nearest")
    elif kind == "spec":
        k = -(-a.shape[1] // 160)  # at most 160 columns: block means (smaller PNG)
        if k > 1:
            a = a[:, : a.shape[1] // k * k].reshape(a.shape[0], -1, k).mean(-1)
        ax.imshow(a, cmap="magma", origin="lower", aspect="auto", interpolation="nearest")
    elif kind == "corr":
        mid = "#f0efec" if theme == "light" else "#383835"
        cmap = LinearSegmentedColormap.from_list("div", [DIVERGING[0], mid, DIVERGING[2]])
        ax.imshow(a, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    elif kind == "lines":
        col = "#2a78d6" if theme == "light" else "#3987e5"
        for row in a:
            ax.plot(row, color=col, alpha=0.55, linewidth=1)
        ax.set_aspect("auto")
    elif kind == "skel":
        col = "#eb6834" if theme == "light" else "#f0995e"
        frames = np.linspace(0, len(a) - 1, 5).astype(int)
        span = np.ptp(a[..., 0][a[..., 0] != 0]) if np.any(a[..., 0]) else 1
        for k, f in enumerate(frames):
            j = a[f]
            dx = k * span * 0.9
            for u, v in NTU_BONES:
                ax.plot([j[u - 1, 0] + dx, j[v - 1, 0] + dx], [j[u - 1, 1], j[v - 1, 1]],
                        color=col, linewidth=1.6, alpha=0.35 + 0.65 * (k + 1) / len(frames),
                        solid_capstyle="round")
        ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


COLOURS = {"pastis_r": 48}  # natural images: fewer colours keep the PNG small


def _png(fig, path, facecolor, colors=128):
    import matplotlib.pyplot as plt
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=facecolor)
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                                            dither=Image.Dither.NONE).save(path, optimize=True)


def draw(name, panels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    panels = panels[:4]
    wide = all(k == "rgb" and a.shape[1] > 3 * a.shape[0] for k, a, _ in panels)
    for theme in ("light", "dark"):
        t = THEMES[theme]
        if wide:  # elongated images (HyperLeaf leaves): one row each, label on the left
            fig, axes = plt.subplots(len(panels), 1, figsize=(10.4, 0.75 * len(panels) + 0.2),
                                     dpi=100, squeeze=False)
            axes = axes.T
        else:
            fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.75), dpi=100,
                                     squeeze=False)
        for ax, (kind, a, label) in zip(axes[0], panels, strict=True):
            _draw(ax, kind, a, theme)
            label = label if len(label) <= 28 else label[:27] + "…"
            if wide:
                ax.set_ylabel(label, fontsize=9, color=t["ink"], rotation=0, ha="right", va="center")
            else:
                ax.set_title(label, fontsize=9, color=t["ink"], pad=5)
        if wide:
            fig.subplots_adjust(left=0.14, right=0.99, bottom=0.03, top=0.97, hspace=0.12)
        else:
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.88, wspace=0.08)
        _png(fig, out / ("samples.png" if theme == "light" else "samples_dark.png"), t["surface"],
             COLOURS.get(name, 128))
    # thumbnail: 16:9, no text, on the dark surface (the tiles show it in both themes)
    n = len(panels)
    fig, axes = plt.subplots(n if wide else 1, 1 if wide else n, figsize=(4.8, 2.7), dpi=100,
                             squeeze=False)
    for ax, (kind, a, _) in zip(axes.ravel(), panels, strict=True):
        _draw(ax, kind, a, "dark")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.96, wspace=0.05, hspace=0.1)
    _png(fig, out / "thumb.png", THEMES["dark"]["surface"], COLOURS.get(name, 128))
    return {p.name: p.stat().st_size for p in sorted(out.glob("*.png"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help=f"datasets (default: all): {', '.join(PREVIEWS)}")
    args = ap.parse_args()
    report = {}
    for name in args.names or list(PREVIEWS):
        try:
            report[name] = draw(name, PREVIEWS[name]())
        except Exception as exc:  # one dataset must not stop the others
            traceback.print_exc()
            report[name] = f"FAILED: {type(exc).__name__}: {exc}"
        print(name, report[name], flush=True)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
