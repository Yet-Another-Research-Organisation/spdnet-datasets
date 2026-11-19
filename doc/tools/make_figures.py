"""Build the figures of a dataset page (Plotly figures, written by figures_io.py).

Run on a machine that holds the data (BERTHA); the documentation build only
embeds the resulting files, so the CI never needs the datasets::

    python doc/tools/make_figures.py pastis_r --kwargs '{"scenario": "A"}' --split official

Writes ``doc/_static/figures/<page>/<kind>.{png,json}`` (fixed or interactive,
see ``figures_io.py``) and ``summary.json`` (class
counts per split, p, n). The five figure kinds follow the dataset-atlas contract,
plus the eigenvalue spectrum:

- ``raw_samples``: examples of the preprocessed data;
- ``class_hist``: samples per class and split;
- ``cov_samples``: a few covariance matrices (as correlations, diverging scale);
- ``class_means``: mean channel profile per class;
- ``split_schema``: split sizes and samples per group (the split unit);
- ``eigen_spectrum``: median and 10-90 % band of the log10 eigenvalues.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from figures_io import write_figure

# Reference categorical palette (dataviz skill), light mode, fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]
DIVERGING = [[0.0, "#184f95"], [0.5, "#f0efec"], [1.0, "#b8282a"]]
SEQUENTIAL = [[0.0, "#cde2fb"], [1.0, "#0d366b"]]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
OUT = Path(__file__).resolve().parents[1] / "_static" / "figures"


def _layout(fig, title, **kw):
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15, "color": INK}},
        font={"family": "system-ui, sans-serif", "size": 12, "color": MUTED},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 50, "r": 20, "t": 50, "b": 45}, hovermode="closest",
        legend={"orientation": "h", "y": -0.2}, **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _samples(ds, idx):
    """Return covs, means, raws, labels for the given indices."""
    covs, means, raws, ys = [], [], [], []
    for i in idx:
        x, y = ds[int(i)]
        parts = x if isinstance(x, tuple) else (x,)
        covs.append(np.asarray(parts[0]))
        if ds.return_mean:
            means.append(np.asarray(parts[1]))
        if ds.return_raw:
            raws.append(np.asarray(parts[-1]))
        ys.append(int(y) if np.ndim(y) == 0 else -2)
    return covs, means, raws, np.asarray(ys)


def build(name, kwargs, split, page, max_samples=600, seed=0):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from spdnet_datasets import DatasetManager

    cls = DatasetManager.DATASETS[name]
    if "cov" not in cls.OUTPUTS:
        raise SystemExit(f"{name} has no covariance output; use make_figures_raw.py")
    # request the extra outputs at construction: some loaders open the raw data there
    has_raw = "raw" in cls.OUTPUTS
    wanted = {"return_mean": "mean" in cls.OUTPUTS, "return_raw": has_raw}
    ds = cls(**{**wanted, **kwargs})
    # raw data are then read for the four displayed samples only (memory)
    ds.return_raw = False
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(max_samples, len(ds)), replace=False)
    covs, means, _, ys = _samples(ds, idx)
    raws = []
    if has_raw:
        ds.return_raw = True
        raws = _samples(ds, idx[:4])[2]
        ds.return_raw = False
    covs = np.stack(covs)
    p = covs.shape[-1]
    names = list(getattr(ds, "classes", [])) or [str(c) for c in np.unique(ys)]
    out = OUT / page
    out.mkdir(parents=True, exist_ok=True)

    figs = {}
    # raw_samples
    if raws and np.asarray(raws[0]).ndim >= 2:
        k = min(4, len(raws))
        fig = make_subplots(rows=1, cols=k, subplot_titles=[names[ys[j]] if ys[j] >= 0 else "?"
                                                            for j in range(k)])
        for j in range(k):
            r = np.asarray(raws[j], dtype=float)
            img = r.mean(-1) if r.ndim == 3 else r.T  # band-mean image or (p, T) spectrogram
            img = img[:, ::-(-img.shape[1] // 400)]  # at most 400 columns (file size)
            fig.add_trace(go.Heatmap(z=img.astype(np.float32), colorscale=SEQUENTIAL,
                                     showscale=False, hovertemplate="value %{z:.3g}<extra></extra>"), 1, j + 1)
        _layout(fig, "Examples of the preprocessed data (band mean / spectrogram)")
        figs["raw_samples"] = fig

    # class_means (8 classes at most: fixed categorical order, no cycling)
    fig = go.Figure()
    # without a mean output (provided matrices), the profile is the log10 variance
    M = np.stack(means) if means else np.log10(np.diagonal(covs, axis1=1, axis2=2))
    for c, color in zip(np.unique(ys[ys >= 0])[:8], SERIES, strict=False):
        fig.add_scatter(y=M[ys == c].mean(0), mode="lines", name=names[c],
                        line={"width": 2, "color": color})
    what = "profile" if means else "log10 variance"
    _layout(fig, f"Mean channel {what} per class (first 8 classes)", xaxis_title="channel",
            yaxis_title="mean" if means else "log10 variance")
    figs["class_means"] = fig

    # class_hist
    fig = go.Figure()
    all_y = _all_labels(ds)
    splits = ds.splits.get(split) if getattr(ds, "splits", None) else None
    parts = splits or {"all": np.arange(len(all_y))}
    counts = {}
    for (s, ii), color in zip(parts.items(), SERIES, strict=False):
        ii = np.asarray(ii, dtype=int)
        cnt = np.bincount(all_y[ii][all_y[ii] >= 0], minlength=len(names))
        counts[s] = cnt.tolist()
        fig.add_bar(x=names, y=cnt, name=f"{s} ({len(ii)})", marker_color=color)
    _layout(fig, f"Samples per class ({split or 'all'})", barmode="group",
            yaxis_title="samples")
    figs["class_hist"] = fig

    # cov_samples
    k = min(4 if p <= 128 else 2, len(covs))  # fewer panels for large p (file size)
    fig = make_subplots(rows=1, cols=k, subplot_titles=[names[ys[j]] if ys[j] >= 0 else "?"
                                                        for j in range(k)])
    for j in range(k):
        d = np.sqrt(np.diag(covs[j]))
        corr = (covs[j] / np.outer(d, d)).astype(np.float32)
        fig.add_trace(go.Heatmap(z=corr, zmin=-1, zmax=1, colorscale=DIVERGING,
                                 showscale=j == k - 1,
                                 hovertemplate="(%{y}, %{x}) %{z:.2f}<extra></extra>"), 1, j + 1)
        fig.update_yaxes(autorange="reversed", row=1, col=j + 1)
    _layout(fig, f"Covariance matrices shown as correlations ({p}×{p})")
    figs["cov_samples"] = fig

    # eigen_spectrum
    logeig = np.log10(np.linalg.eigvalsh(covs)[:, ::-1])
    q10, q50, q90 = np.nanpercentile(logeig, [10, 50, 90], axis=0)  # NaN: non-SPD
    x = np.arange(1, p + 1)
    fig = go.Figure()
    fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[q90, q10[::-1]], fill="toself",
                    fillcolor="rgba(42,120,214,0.18)", line={"width": 0},
                    name="10–90 %", hoverinfo="skip")
    fig.add_scatter(x=x, y=q50, mode="lines", name="median", line={"width": 2,
                                                                  "color": SERIES[0]})
    _layout(fig, "Eigenvalue spectrum (log10, sorted)", xaxis_title="rank",
            yaxis_title="log10 eigenvalue")
    figs["eigen_spectrum"] = fig

    # split_schema: split sizes, and samples per group (the unit kept together)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["samples per split",
                                                        "samples per group"])
    fig.add_bar(x=list(parts), y=[len(v) for v in parts.values()], marker_color=SERIES[0],
                showlegend=False, row=1, col=1)
    if getattr(ds, "groups", None) is not None:
        _, sizes = np.unique(np.asarray(ds.groups), return_counts=True)
        fig.add_histogram(x=sizes, marker_color=SERIES[1], nbinsx=40, showlegend=False,
                          row=1, col=2)
    _layout(fig, "Split sizes and group sizes (groups never cross splits)")
    figs["split_schema"] = fig

    for kind, f in figs.items():
        write_figure(f, out, kind)
    summary = {"name": name, "kwargs": kwargs, "split": split, "p": int(p),
               "n_samples": int(len(ds)), "classes": names, "counts": counts}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _all_labels(ds):
    for attr in ("labels", "targets", "y"):
        v = getattr(ds, attr, None)
        if v is not None and len(v) == len(ds) and np.ndim(v) == 1:
            return np.asarray(v).astype(int)
    return np.asarray([int(ds[i][1]) for i in range(len(ds))])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="registered dataset name")
    ap.add_argument("--kwargs", default="{}", help="JSON constructor arguments")
    ap.add_argument("--split", default="official")
    ap.add_argument("--page", default=None, help="output folder (default: name)")
    ap.add_argument("--max-samples", type=int, default=600)
    a = ap.parse_args()
    print(json.dumps(build(a.name, json.loads(a.kwargs), a.split, a.page or a.name,
                           a.max_samples), indent=1))
