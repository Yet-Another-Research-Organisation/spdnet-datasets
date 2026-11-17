"""Build the figures of a raw- or mean-only dataset page, and page-specific extras.

``make_figures.py`` needs a covariance output. The SAR datasets of the mission
that only give raw chips or channel means (``OUTPUTS = ("raw", "mean")``) are
drawn here with the same palette, layout and output convention::

    python doc/tools/make_figures_raw.py mstar --split official
    python doc/tools/make_figures_raw.py atrnet_star --split official --fifth protocol_overlap

Figures written to ``doc/_static/figures/<page>/`` (fixed PNG or interactive JSON,
see ``figures_io.py``), with ``summary.json``:

- ``raw_samples``: one chip per class for four classes (first channel, log10
  scale or as returned with ``--raw-scale linear``, 1-99 % clipping, at most
  128 px a side);
- ``class_hist``: samples per class and split part;
- ``class_means``: mean of each channel per class (box plot of the chip means
  when there is a single channel);
- ``split_schema``: split sizes and samples per group (the split unit);
- fifth figure (``--fifth``): ``amplitude_hist`` (distribution of the raw
  values per channel) or ``protocol_overlap`` (ATRNet-STAR: chips shared by the
  train / test parts of every pair of protocols).

Extras (``--extra``) read loader caches or metadata for one page:

- ``pastis_dates``: S2-S1 gap distribution and acquisition timeline per tile
  (PASTIS-R processed cache, ``gap_histogram.csv`` and ``date_gaps.csv``);
- ``split_map``: map of the samples coloured by split part, from two columns
  of ``dataset.meta`` (``--xy easting,northing``);
- ``parcel_pixels``: ``raw_samples`` for datasets whose raw output is an
  irregular set of pixels ``(n, p)`` (overwrites the one of ``make_figures.py``,
  which draws every pixel);
- ``cpazmal_dates``: the HH / HV acquisition dates retained for each orbit;
- ``image_mean``: ``raw_samples`` as the channel-mean image, for loaders whose
  raw output is a set of pixels (``--kwargs`` selects the image layout).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from figures_io import write_figure
from make_figures import GRID, OUT, SEQUENTIAL, SERIES, _layout  # noqa: F401

PART_COLORS = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}


def _write(out, figs):
    for kind, f in figs.items():
        write_figure(f, out, kind)


def _labels(ds):
    for attr in ("targets", "labels"):
        v = getattr(ds, attr, None)
        if v is not None and len(v) == len(ds):
            return np.asarray(v).astype(int)
    raise SystemExit("dataset exposes neither `targets` nor `labels`")


def _parts(ds, split):
    splits = getattr(ds, "splits", None) or {}
    if split not in splits:
        raise SystemExit(f"split {split!r} not in dataset.splits: {sorted(splits)}")
    return {k: np.asarray(v, dtype=int) for k, v in splits[split].items()}


def _sample(y, max_samples, rng, contiguous=0):
    """Indices stratified by class (or blocks of ``contiguous`` consecutive items)."""
    if contiguous:
        starts = rng.choice(np.arange(0, len(y), contiguous),
                            size=max(1, max_samples // contiguous), replace=False)
        return np.sort(np.concatenate([np.arange(s, min(s + contiguous, len(y))) for s in starts]))
    classes = np.unique(y)
    per = max(3, int(np.ceil(max_samples / len(classes))))
    idx = [rng.choice(np.flatnonzero(y == c), min(per, int((y == c).sum())), replace=False)
           for c in classes]
    return np.sort(np.concatenate(idx))


def _get(ds, i, what):
    ds.return_raw, ds.return_mean = what == "raw", what == "mean"
    x, _ = ds[int(i)]
    return np.asarray(x, dtype=np.float64)


def build_raw(name, kwargs, split, page, max_samples=400, seed=0, fifth="amplitude_hist",
              channel_names=None, contiguous=0, raw_scale="log10"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from spdnet_datasets import DatasetManager

    cls = DatasetManager.DATASETS[name]
    if "cov" in cls.OUTPUTS:
        raise SystemExit(f"{name} has a covariance output: use make_figures.py")
    ds = cls(**{"return_raw": True, **kwargs})
    y_all = _labels(ds)
    parts = _parts(ds, split)
    names = list(ds.classes)
    rng = np.random.default_rng(seed)
    idx = _sample(y_all, max_samples, rng, contiguous)
    means = np.stack([_get(ds, i, "mean") for i in idx])
    ys = y_all[idx]
    p = means.shape[1]
    ch = channel_names or [f"channel {k}" for k in range(p)]
    figs = {}

    # raw_samples: one chip of each of four classes (first channel), at most 128 px a side
    shown = [int(idx[np.flatnonzero(ys == c)[0]]) for c in np.unique(ys)[:4]]
    raws = [_get(ds, i, "raw") for i in shown]
    fig = make_subplots(rows=1, cols=len(shown), subplot_titles=[names[y_all[i]] for i in shown])
    for j, r in enumerate(raws):
        step = int(np.ceil(max(r.shape[:2]) / 128))
        img = r[::step, ::step, 0]
        if raw_scale == "log10":  # floor at 1e-3 of the maximum: zeros stay visible
            img = np.log10(np.abs(img) + 1e-3 * np.abs(img).max() + 1e-12)
        lo, hi = np.percentile(img, [1, 99])
        fig.add_trace(go.Heatmap(z=np.round(img, 2).astype(np.float32), zmin=lo, zmax=hi,
                                 colorscale=SEQUENTIAL, showscale=False,
                                 hovertemplate="%{z:.2f}<extra></extra>"), 1, j + 1)
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{j + 1 if j else ''}",
                         constrain="domain", row=1, col=j + 1)
        fig.update_xaxes(constrain="domain", row=1, col=j + 1)
    shape = "×".join(str(s) for s in raws[0].shape)
    scale_txt = "log10 scale" if raw_scale == "log10" else "as returned"
    _layout(fig, f"Raw chips ({shape}), {ch[0]}, {scale_txt}")
    figs["raw_samples"] = fig

    # class_hist
    fig = go.Figure()
    counts = {}
    for s, ii in parts.items():
        cnt = np.bincount(y_all[ii], minlength=len(names))
        counts[s] = cnt.tolist()
        fig.add_bar(x=names, y=cnt, name=f"{s} ({len(ii)})", marker_color=PART_COLORS.get(s))
    _layout(fig, f"Samples per class ({split})", barmode="group", yaxis_title="samples")
    figs["class_hist"] = fig

    # class_means
    fig = go.Figure()
    if p == 1:
        for c in np.unique(ys):
            fig.add_box(y=means[ys == c, 0], name=names[c], marker_color=SERIES[0],
                        boxpoints=False, showlegend=False)
        _layout(fig, f"Chip mean per class ({len(idx)} sampled chips)",
                yaxis_title=f"mean of {ch[0]}")
    else:
        for c, color in zip(np.unique(ys)[:8], SERIES, strict=False):
            fig.add_scatter(x=ch, y=means[ys == c].mean(0), mode="lines+markers", name=names[c],
                            line={"width": 2, "color": color}, marker={"size": 8})
        _layout(fig, f"Mean channel value per class ({len(idx)} sampled chips, first 8 classes)",
                yaxis_title="mean")
    figs["class_means"] = fig

    # split_schema
    fig = make_subplots(rows=1, cols=2, subplot_titles=["samples per split part",
                                                        "samples per group"])
    fig.add_bar(x=list(parts), y=[len(v) for v in parts.values()],
                marker_color=[PART_COLORS.get(s, SERIES[3]) for s in parts], showlegend=False,
                row=1, col=1)
    if getattr(ds, "groups", None) is not None:
        _, sizes = np.unique(np.asarray(ds.groups), return_counts=True)
        fig.add_histogram(x=sizes, marker_color=SERIES[1], nbinsx=40, showlegend=False,
                          row=1, col=2)
        fig.update_xaxes(title_text="samples in the group", row=1, col=2)
        _layout(fig, f"Split sizes and group sizes ({len(sizes)} groups, never across parts)")
    else:  # no group (e.g. MSTAR: the parts are fixed by the depression angle)
        sizes = np.zeros(0)
        fig.add_annotation(text="no groups: the parts are given by the source", showarrow=False,
                           xref="x2 domain", yref="y2 domain", x=0.5, y=0.5)
        _layout(fig, "Split sizes (no groups)")
    figs["split_schema"] = fig

    extra = {}
    if fifth == "amplitude_hist":
        vals = np.concatenate([(r := _get(ds, i, "raw")).reshape(-1, r.shape[-1])
                               for i in idx[:: max(1, len(idx) // 60)]])
        fig = go.Figure()
        rch = channel_names if channel_names and len(channel_names) == vals.shape[1] else \
            [f"channel {k}" for k in range(vals.shape[1])]
        for k, color in zip(range(min(vals.shape[1], 8)), SERIES, strict=False):
            v = vals[:, k]
            h, e = np.histogram(v, bins=80)
            fig.add_scatter(x=(e[1:] + e[:-1]) / 2, y=h / h.sum(), mode="lines", name=rch[k],
                            line={"width": 2, "color": color})
        _layout(fig, "Distribution of the raw values per channel (sampled chips)",
                xaxis_title="raw value", yaxis_title="fraction of pixels", yaxis_type="log")
        figs["amplitude_hist"] = fig
    elif fifth == "protocol_overlap":
        from spdnet_datasets.sar.atrnet_star import protocol_overlap

        prots = ["SOC_40classes", "EOC_scene", "EOC_depression", "EOC_azimuth", "EOC_band"]
        rep = kwargs.get("representation", "Slant_Range")
        labels_ = [f"{q} {part}" for q in prots for part in ("train", "test")]
        z = np.zeros((len(labels_), len(labels_)), int)
        for a_i, a in enumerate(prots):
            for b_i, b in enumerate(prots):
                if b_i < a_i:
                    continue
                ov = protocol_overlap(ds.data_dir, a, b, representation=rep)
                for (pa, pb), v in ov.items():
                    r_, c_ = 2 * a_i + (pa == "test"), 2 * b_i + (pb == "test")
                    z[r_, c_] = z[c_, r_] = v
        extra["protocol_overlap"] = {"labels": labels_, "counts": z.tolist()}
        fig = go.Figure(go.Heatmap(z=np.log10(z + 1), x=labels_, y=labels_, customdata=z,
                                   colorscale=SEQUENTIAL,
                                   colorbar={"title": "log10(1 + chips)"},
                                   hovertemplate="%{y} ∩ %{x}: %{customdata} chips"
                                                 "<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        _layout(fig, f"Chips shared between protocol parts ({rep}, same class and file name)")
        fig.update_layout(margin={"l": 150, "r": 20, "t": 50, "b": 130}, height=560)
        figs["protocol_overlap"] = fig

    out = OUT / page
    _write(out, figs)
    summary = {"name": name, "kwargs": kwargs, "split": split, "n_samples": int(len(ds)),
               "n_groups": int(len(sizes)) if len(sizes) else None, "raw_shape": list(raws[0].shape), "mean_dim": p,
               "classes": names, "counts": counts, **extra}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# ------------------------------------------------------------------------ extras
def pastis_dates(page, cache_dir):
    """S2-S1 gap distribution and acquisition timeline per tile (PASTIS-R)."""
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    proc = None
    for d in sorted((Path(cache_dir) / "pastis_r" / "processed").iterdir()):
        try:
            prm = json.loads((d / "report.json").read_text())["params"]
        except (OSError, KeyError, ValueError):
            continue
        if prm.get("version") == 2 and prm.get("patches") is None and prm["scenario"] == "C":
            proc = d
    if proc is None:
        raise SystemExit("no full scenario-C processed cache (version 2) found")
    hist = pd.read_csv(proc / "gap_histogram.csv")
    gaps = pd.read_csv(proc / "date_gaps.csv")
    figs = {}

    fig = go.Figure()
    for sensor, color in (("S1A", SERIES[0]), ("S1D", SERIES[1])):
        h = hist[hist.sensor == sensor]
        fig.add_bar(x=h.abs_gap_days, y=h.n_patch_dates / h.n_patch_dates.sum(),
                    name=f"{sensor} (ascending)" if sensor == "S1A" else f"{sensor} (descending)",
                    marker_color=color, customdata=h.n_patch_dates,
                    hovertemplate="%{x} d: %{customdata} patch-dates<extra></extra>")
    fig.add_vline(x=3.5, line_dash="dash", line_color=SERIES[4],
                  annotation_text="max_gap_days = 3", annotation_position="top right")
    _layout(fig, "Gap between each S2 date and the nearest S1 acquisition (all patches, all dates)",
            barmode="group", xaxis_title="|S1 − S2| (days)", yaxis_title="fraction of patch-dates")
    figs["date_gaps"] = fig

    tiles = sorted(gaps.tile.unique())
    fig = make_subplots(rows=len(tiles), cols=1, shared_xaxes=True, subplot_titles=tiles,
                        vertical_spacing=0.06)
    for r, t in enumerate(tiles, 1):
        g = gaps[gaps.tile == t]
        for col, yv, nm, color in (("s2_date", 2, "S2", SERIES[2]), ("s1a_date", 1, "nearest S1A",
                                   SERIES[0]), ("s1d_date", 0, "nearest S1D", SERIES[1])):
            fig.add_scatter(x=pd.to_datetime(g[col]), y=[yv] * len(g), mode="markers", name=nm,
                            marker={"size": 8, "color": color, "symbol": "line-ns-open",
                                    "line": {"width": 2, "color": color}},
                            showlegend=r == 1, legendgroup=nm,
                            customdata=np.c_[g.s2_date, g.gap_s1a, g.gap_s1d],
                            hovertemplate="S2 %{customdata[0]}<br>S1A gap %{customdata[1]} d, "
                                          "S1D gap %{customdata[2]} d<extra></extra>", row=r, col=1)
        fig.add_vrect(x0="2019-06-30", x1="2019-07-30", fillcolor=SERIES[3], opacity=0.15,
                      line_width=0, row=r, col=1)
        fig.update_yaxes(tickvals=[0, 1, 2], ticktext=["S1D", "S1A", "S2"], range=[-0.6, 2.6],
                         row=r, col=1)
    _layout(fig, "S2 acquisitions and their nearest S1 dates per tile "
                 "(shaded: ref_date 2019-07-15 ± 15 days)", height=620)
    figs["date_timeline"] = fig
    _write(OUT / page, figs)
    q = json.loads((proc / "report.json").read_text())["gap_quantiles_all_s2_dates"]
    return {"cache": proc.name, "quantiles": q, "n_tile_dates": int(len(gaps))}


def split_map(name, kwargs, split, page, xy, scale=1.0, unit=""):
    """Scatter of the samples at ``meta[xy]``, coloured by split part."""
    import plotly.graph_objects as go

    from spdnet_datasets import DatasetManager

    cls = DatasetManager.DATASETS[name]
    extra = {} if "cov" in cls.OUTPUTS else {"return_mean": True}
    ds = cls(**{**extra, **kwargs})
    parts = _parts(ds, split)
    meta = ds.meta
    xs, ys_ = (meta[c].to_numpy(float) * scale for c in xy)
    fig = go.Figure()
    used = np.zeros(len(meta), bool)
    for s, ii in parts.items():
        used[ii] = True
        fig.add_scattergl(x=xs[ii], y=ys_[ii], mode="markers", name=f"{s} ({len(ii)})",
                          marker={"size": 4, "color": PART_COLORS.get(s, SERIES[3])})
    if (~used).any():
        fig.add_scattergl(x=xs[~used], y=ys_[~used], mode="markers",
                          name=f"removed by the guard band ({int((~used).sum())})",
                          marker={"size": 4, "color": GRID})
    _layout(fig, f"Samples by split part ({split})", xaxis_title=f"{xy[0]} {unit}".strip(),
            yaxis_title=f"{xy[1]} {unit}".strip())
    fig.update_yaxes(scaleanchor="x")
    _write(OUT / page, {"split_map": fig})
    return {k: int(len(v)) for k, v in parts.items()} | {"unused": int((~used).sum())}


def parcel_pixels(name, kwargs, page, max_pixels=300, seed=0):
    """``raw_samples`` for irregular samples (raw = (n, p) pixels): 4 classes, n subsampled."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from spdnet_datasets import DatasetManager

    ds = DatasetManager.DATASETS[name](**{**kwargs, "return_raw": True})
    y = _labels(ds)
    rng = np.random.default_rng(seed)
    shown = [int(rng.choice(np.flatnonzero(y == c))) for c in np.unique(y)[:4]]
    fig = make_subplots(rows=1, cols=len(shown), subplot_titles=[ds.classes[y[i]] for i in shown])
    n_px = []
    for j, i in enumerate(shown):
        x, _ = ds[i]
        r = np.asarray(x[-1], dtype=np.float32)
        n_px.append(len(r))
        keep = np.sort(rng.choice(len(r), min(max_pixels, len(r)), replace=False))
        fig.add_trace(go.Heatmap(z=np.round(r[keep].T, 3), colorscale=SEQUENTIAL,
                                 showscale=False,
                                 hovertemplate="channel %{y}, pixel %{x}: %{z:.3g}<extra></extra>"),
                      1, j + 1)
        fig.update_yaxes(autorange="reversed", row=1, col=j + 1)
    _layout(fig, f"Pixels of four samples (channels × pixels, ≤ {max_pixels} pixels shown; "
                 f"n = {', '.join(map(str, n_px))})")
    _write(OUT / page, {"raw_samples": fig})
    return {"n_pixels": n_px}


def image_mean(name, kwargs, page, channel_axis=-1, seed=0):
    """``raw_samples`` as the channel-mean image of four samples of distinct classes.

    For loaders whose default raw output is a set of pixels: pass the
    constructor arguments that give the image (HyperLeaf ``mode="image"`` with
    every band, Pol-InSAR-Island ``raw_layout="grid"``).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from spdnet_datasets import DatasetManager

    ds = DatasetManager.DATASETS[name](**kwargs)
    y = _labels(ds)
    rng = np.random.default_rng(seed)
    shown = [int(rng.choice(np.flatnonzero(y == c))) for c in np.unique(y)[:4]]
    imgs = []
    for i in shown:
        x, _ = ds[i]
        img = np.asarray(x[-1] if isinstance(x, tuple) else x, dtype=np.float32)
        imgs.append(np.round(img.mean(axis=channel_axis), 3))
    wide = imgs[0].shape[1] > 3 * imgs[0].shape[0]  # elongated images (leaves): 2 x 2 grid
    rows, cols = (2, 2) if wide else (1, len(imgs))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[ds.classes[y[i]] for i in shown],
                        vertical_spacing=0.12)
    for j, z in enumerate(imgs):
        r, c = (j // 2 + 1, j % 2 + 1) if wide else (1, j + 1)
        fig.add_trace(go.Heatmap(z=z, colorscale=SEQUENTIAL, showscale=False,
                                 hovertemplate="value %{z:.3g}<extra></extra>"), r, c)
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{j + 1 if j else ''}", row=r, col=c)
    _layout(fig, "Channel-mean image of four samples")
    _write(OUT / page, {"raw_samples": fig})
    return {"shown": shown, "shape": list(img.shape)}


def cpazmal_dates(page, kwargs):
    """The ``n_dates`` acquisition dates (HH and HV) retained for each orbit."""
    import pandas as pd
    import plotly.graph_objects as go

    from spdnet_datasets import DatasetManager

    figs, info = {}, {}
    fig = go.Figure()
    for row, orbit in enumerate(("ASC", "DSC")):
        ds = DatasetManager.DATASETS["cpazmal"](**{**kwargs, "orbit": orbit})
        kept = pd.to_datetime(pd.Series([str(d) for d in ds.dates]))
        info[orbit] = {"n_samples": len(ds), "dates": [str(d) for d in ds.dates],
                       "excluded": sorted(map(str, ds.excluded))}
        fig.add_scatter(x=kept, y=[orbit] * len(kept), mode="markers", name=f"{orbit} retained",
                        marker={"size": 10, "color": SERIES[row], "symbol": "line-ns-open",
                                "line": {"width": 2, "color": SERIES[row]}})
    _layout(fig, "Retained acquisition dates (HH and HV) per orbit", xaxis_title="date")
    figs["dates"] = fig
    _write(OUT / page, figs)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", nargs="?", help="registered dataset name")
    ap.add_argument("--kwargs", default="{}", help="JSON constructor arguments")
    ap.add_argument("--split", default="official")
    ap.add_argument("--page", default=None, help="output folder (default: name)")
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--fifth", default="amplitude_hist",
                    choices=["amplitude_hist", "protocol_overlap", "none"])
    ap.add_argument("--channels", default=None, help="comma-separated channel names")
    ap.add_argument("--contiguous", type=int, default=0,
                    help="sample blocks of this many consecutive items (chunked stores)")
    ap.add_argument("--raw-scale", default="log10", choices=["log10", "linear"],
                    help="display of raw_samples (linear for already log-scaled raw outputs)")
    ap.add_argument("--extra", choices=["pastis_dates", "split_map", "cpazmal_dates", "parcel_pixels",
                                        "image_mean"])
    ap.add_argument("--channel-axis", type=int, default=-1, help="image_mean: channel axis")
    ap.add_argument("--xy", default="easting,northing", help="split_map: meta columns")
    ap.add_argument("--scale", type=float, default=1.0, help="split_map: coordinate factor")
    ap.add_argument("--unit", default="", help="split_map: axis unit")
    ap.add_argument("--cache-dir", default=None, help="pastis_dates: SPDNET_CACHE_DIR")
    a = ap.parse_args()
    kw = json.loads(a.kwargs)
    page = a.page or a.name
    if a.extra == "pastis_dates":
        import os

        res = pastis_dates(page or "pastis_r", a.cache_dir or os.environ["SPDNET_CACHE_DIR"])
    elif a.extra == "split_map":
        res = split_map(a.name, kw, a.split, page, a.xy.split(","), a.scale, a.unit)
    elif a.extra == "parcel_pixels":
        res = parcel_pixels(a.name, kw, page)
    elif a.extra == "image_mean":
        res = image_mean(a.name, kw, page, a.channel_axis)
    elif a.extra == "cpazmal_dates":
        res = cpazmal_dates(page or "cpazmal", kw)
    else:
        res = build_raw(a.name, kw, a.split, page, a.max_samples, fifth=a.fifth,
                        channel_names=a.channels.split(",") if a.channels else None,
                        contiguous=a.contiguous, raw_scale=a.raw_scale)
    print(json.dumps(res, indent=1, default=str))
