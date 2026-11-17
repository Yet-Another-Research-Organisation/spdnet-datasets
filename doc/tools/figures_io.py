"""Write the figures of a dataset page: fixed images or light interactive data.

The figure scripts (``make_figures.py``, ``make_figures_raw.py``) build Plotly
figures and hand them to :func:`write_figure`, which decides how each kind is
published:

- **interactive** kinds (:data:`INTERACTIVE`), where hovering over a value
  helps (covariance matrices, eigenvalue spectra, overlap matrices), are written
  as compact Plotly JSON ``<kind>.json`` (no template, float32 arrays). The
  ``dataset-figure`` directive (``doc/_ext/dataset_figures.py``) embeds them as
  ``<div>`` elements that ``_static/plotly-figures.js`` fills lazily, with a
  single plotly.js per page;
- every other kind (class counts, split schemas, mean profiles, examples of the
  raw data, maps, dates) is drawn with matplotlib into two small PNG files,
  ``<kind>.png`` (light) and ``<kind>_dark.png`` (dark), from the same Plotly
  figure, so the two outputs never disagree.

The former HTML files can be converted without the data::

    python doc/tools/figures_io.py --migrate doc/_static/figures

reads the figure embedded in every ``<kind>.html`` (``Plotly.newPlot(...)``),
writes the new outputs and removes the HTML file.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np

INTERACTIVE = {"cov_samples", "eigen_spectrum", "protocol_overlap"}

# Reference categorical palette (dataviz skill): light steps and their dark steps.
LIGHT_TO_DARK = {
    "#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70",
    "#eda100": "#c98500", "#e87ba4": "#d55181", "#008300": "#0ca30c",
    "#6250d6": "#9085e9", "#e34948": "#e66767",
}
THEMES = {
    "light": {"surface": "#ffffff", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e7e6e2",
              "mid": "#f0efec"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#34342f",
             "mid": "#383835"},
}
_DTYPES = {"f8": np.float64, "f4": np.float32, "i1": np.int8, "u1": np.uint8, "i2": np.int16,
           "u2": np.uint16, "i4": np.int32, "u4": np.uint32}


# ---------------------------------------------------------------------------
# Plotly JSON helpers
# ---------------------------------------------------------------------------
def decode(v):
    """Plotly typed array ``{"dtype", "bdata", "shape"}`` -> numpy array."""
    if isinstance(v, dict) and "bdata" in v:
        a = np.frombuffer(base64.b64decode(v["bdata"]), dtype=_DTYPES[v["dtype"]])
        if "shape" in v:
            a = a.reshape([int(s) for s in str(v["shape"]).split(",")])
        return a
    return v


def _encode_f4(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    out = {"dtype": "f4", "bdata": base64.b64encode(a.tobytes()).decode()}
    if a.ndim > 1:
        out["shape"] = ", ".join(map(str, a.shape))
    return out


def _compact(obj):
    """float64 typed arrays -> float32 (half the size); recursive."""
    if isinstance(obj, dict):
        if "bdata" in obj and obj.get("dtype") == "f8":
            return _encode_f4(decode(obj))
        return {k: _compact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and all(isinstance(x, float) for x in obj):
            return [float(f"{x:.4g}") if np.isfinite(x) else None for x in obj]
        return [_compact(x) for x in obj]
    return obj


def figure_dict(fig) -> dict:
    """A Plotly figure (object or dict) as a plain dict without its template."""
    d = fig if isinstance(fig, dict) else json.loads(fig.to_json())
    d = {"data": d.get("data", []), "layout": dict(d.get("layout", {}))}
    d["layout"].pop("template", None)
    return d


def extract_html(path: Path) -> dict:
    """The figure passed to ``Plotly.newPlot`` in a file written by ``to_html``."""
    s = path.read_text()
    j = s.index("Plotly.newPlot(") + len("Plotly.newPlot(")
    dec, args = json.JSONDecoder(), []
    for _ in range(3):
        while s[j].isspace() or s[j] == ",":
            j += 1
        v, j = dec.raw_decode(s, j)
        args.append(v)
    return figure_dict({"data": args[1], "layout": args[2]})


# ---------------------------------------------------------------------------
# Static rendering (matplotlib) of the trace types used by the figure scripts
# ---------------------------------------------------------------------------
def _color(c, theme, default):
    if not isinstance(c, str):
        return default
    c = c.lower()
    if theme == "dark":
        c = LIGHT_TO_DARK.get(c, c)
    if c.startswith("rgba("):
        r, g, b, a = (float(x) for x in c[5:-1].split(","))
        if theme == "dark":  # same hue, from the dark steps when it is a palette colour
            hexc = "#%02x%02x%02x" % (int(r), int(g), int(b))
            hexc = LIGHT_TO_DARK.get(hexc, hexc)
            r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
        return (r / 255, g / 255, b / 255, a)
    return c


def _cmap(scale, theme):
    from matplotlib.colors import LinearSegmentedColormap

    stops = [(float(p), THEMES[theme]["mid"] if (c.lower() == "#f0efec") else c)
             for p, c in scale]
    return LinearSegmentedColormap.from_list("plotly", stops)


def _axes_layout(layout):
    """{"x": (x0, x1, y0, y1), "x2": ...} from the axis domains of the layout."""
    boxes = {}
    for key, ax in layout.items():
        if not key.startswith("xaxis"):
            continue
        suffix = key[5:]
        ydom = layout.get(f"yaxis{suffix}", {}).get("domain", [0, 1])
        boxes[suffix] = (*ax.get("domain", [0, 1]), *ydom)
    return boxes or {"": (0, 1, 0, 1)}


def render_static(fig: dict, path: Path, theme: str, width=9.6, height=4.4, dpi=100) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    layout, data = fig["layout"], fig["data"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": t["ink"],
        "axes.labelcolor": t["muted"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["grid"], "axes.facecolor": t["surface"],
        "figure.facecolor": t["surface"], "axes.grid": True, "grid.color": t["grid"],
        "grid.linewidth": 0.7, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })
    f = plt.figure(figsize=(width, height), dpi=dpi)
    title = (layout.get("title") or {}).get("text", "")
    top = 0.86 if title else 0.94
    has_legend = sum(1 for tr in data if tr.get("showlegend", True) is not False and tr.get("name")
                     and tr.get("type") not in ("heatmap",)) > 1
    top = top - 0.05 if has_legend else top
    rotated = any(tr.get("type") == "bar" and tr.get("x") is not None
                  and 8 < len(decode(tr["x"])) <= 60 for tr in data) \
        or sum(tr.get("type") == "box" for tr in data) > 6
    bottom = 0.30 if rotated else 0.12
    boxes = _axes_layout(layout)
    many_panels = len(boxes) > 1
    axes = {}
    for suffix, (x0, x1, y0, y1) in boxes.items():
        L, R = 0.07 + x0 * 0.9, 0.07 + x1 * 0.9
        B, T = bottom + y0 * (top - bottom), bottom + y1 * (top - bottom)
        ax = f.add_axes([L, B, R - L - (0.01 if many_panels else 0), T - B - (0.06 if many_panels else 0)])
        xa, ya = layout.get(f"xaxis{suffix}", {}), layout.get(f"yaxis{suffix}", {})
        if (xa.get("title") or {}).get("text"):
            ax.set_xlabel(xa["title"]["text"])
        if (ya.get("title") or {}).get("text"):
            ax.set_ylabel(ya["title"]["text"])
        if ya.get("type") == "log":
            ax.set_yscale("log")
        if ya.get("tickvals") is not None and ya.get("ticktext") is not None:
            ax.set_yticks(decode(ya["tickvals"]), ya["ticktext"])
            ax.set_ylim(min(decode(ya["tickvals"])) - 0.6, max(decode(ya["tickvals"])) + 0.6)
        axes[suffix] = ax

    groups = [tr for tr in data if tr.get("type") == "bar"]
    nbar = {}
    for tr in groups:
        nbar.setdefault(tr.get("xaxis", "x"), []).append(id(tr))
    seq = {}
    for tr in data:
        ax = axes.get(tr.get("xaxis", "x")[1:], next(iter(axes.values())))
        typ = tr.get("type", "scatter")
        default = "#2a78d6" if theme == "light" else "#3987e5"
        raw_color = (tr.get("marker") or {}).get("color") or (tr.get("line") or {}).get("color")
        if isinstance(raw_color, list):  # one colour per bar
            color = [_color(c, theme, default) for c in raw_color]
        else:
            color = _color(raw_color, theme, default)
        label = tr.get("name") if tr.get("showlegend", True) is not False else None
        if typ == "heatmap":
            z = np.asarray(decode(tr["z"]), dtype=float)
            ax.imshow(z, aspect="auto", interpolation="nearest", origin="lower",
                      cmap=_cmap(tr.get("colorscale", [[0, "#cde2fb"], [1, "#0d366b"]]), theme),
                      vmin=tr.get("zmin"), vmax=tr.get("zmax"))
            ax.grid(False)
            if z.shape[0] > 1 and z.shape[1] / z.shape[0] > 6:
                ax.set_aspect("auto")
            ax.set_xticks([]), ax.set_yticks([])
        elif typ == "bar":
            x = decode(tr.get("x"))
            y = np.asarray(decode(tr.get("y")), dtype=float)
            ids = nbar[tr.get("xaxis", "x")]
            k, m = ids.index(id(tr)), len(ids)
            grouped = layout.get("barmode", "group") == "group" and m > 1
            w = 0.8 / m if grouped else 0.8
            pos = np.arange(len(y)) + ((k - (m - 1) / 2) * w if grouped else 0)
            ax.bar(pos, y, width=w * 0.92, color=color, label=label, linewidth=0)
            cats = [str(c) for c in x] if x is not None else []
            ax.set_xticks(np.arange(len(y)))
            if len(cats) <= 60:
                ax.set_xticklabels(cats, rotation=90 if len(cats) > 8 or max(map(len, cats or [""])) > 8 else 0,
                                   fontsize=7 if len(cats) > 25 else 8)
            else:
                ax.set_xticklabels([])
                ax.set_xlabel(f"{len(cats)} classes")
            ax.grid(axis="x", visible=False)
        elif typ == "histogram":
            x = np.asarray(decode(tr.get("x")), dtype=float)
            x = x[np.isfinite(x)]
            integer = x.size and np.all(x == np.round(x)) and x.max() - x.min() <= 80
            bins = np.arange(x.min(), x.max() + 2) - 0.5 if integer else (tr.get("nbinsx") or 30)
            ax.hist(x, bins=bins, color=color, rwidth=0.85, label=label)
            ax.set_xlabel(ax.get_xlabel() or "samples per group")
        elif typ == "box":
            y = np.asarray(decode(tr.get("y")), dtype=float)
            names = seq.setdefault(("box", id(ax)), [])
            bp = ax.boxplot(y[np.isfinite(y)], positions=[len(names)], widths=0.6,
                            patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set_facecolor(color), b.set_edgecolor(t["muted"])
            for part in ("whiskers", "caps", "medians"):
                for line in bp[part]:
                    line.set_color(t["muted"])
            names.append(str(tr.get("name", "")))
            ax.set_xticks(range(len(names)), names, rotation=45 if len(names) > 6 else 0,
                          ha="right" if len(names) > 6 else "center", fontsize=7)
        else:  # scatter, scattergl
            x = decode(tr.get("x"))
            y = decode(tr.get("y"))
            if isinstance(y, list) and y and isinstance(y[0], str):  # categorical y axis
                cats = seq.setdefault(("cats", id(ax)), [])
                cats += [c for c in dict.fromkeys(y) if c not in cats]
                y = np.asarray([cats.index(c) for c in y], dtype=float)
                ax.set_yticks(range(len(cats)), cats)
                ax.set_ylim(-0.6, len(cats) - 0.4)
            y = np.asarray(y, dtype=float)
            x = np.arange(1, len(y) + 1) if x is None else x
            if tr.get("fill") == "toself":
                ax.fill(np.asarray(x, dtype=float), y, color=_color(tr.get("fillcolor"), theme, color),
                        linewidth=0, label=label)
                continue
            mode = tr.get("mode", "lines")
            if isinstance(x, list) and x and isinstance(x[0], str) and len(x[0]) >= 8 and x[0][4] == "-":
                x = np.asarray(x, dtype="datetime64[D]")
            ms = (tr.get("marker") or {}).get("size", 6)
            if mode == "markers":
                ax.scatter(x, y, s=max(2, ms) ** 1.2, color=color, label=label, linewidths=0,
                           alpha=(tr.get("marker") or {}).get("opacity", 1))
            else:
                ax.plot(x, y, color=color, linewidth=(tr.get("line") or {}).get("width", 2),
                        marker="o" if "markers" in mode else None, markersize=3, label=label)
    for suffix, ax in axes.items():
        if layout.get(f"yaxis{suffix}", {}).get("autorange") == "reversed":
            ax.invert_yaxis()
        if not ax.has_data():  # empty panel (e.g. no groups): keep only its annotation
            ax.set_axis_off()
    if title:
        f.text(0.07, 0.95, title, fontsize=12, color=t["ink"], ha="left", va="top", fontweight="bold")
    for an in layout.get("annotations", []):
        ref = str(an.get("xref", ""))
        if ref.endswith(" domain") and an.get("text"):  # text inside one panel
            ax = axes.get(ref.split()[0][1:], next(iter(axes.values())))
            ax.text(an.get("x", 0.5), an.get("y", 0.5), an["text"], transform=ax.transAxes,
                    ha="center", va="center", color=t["muted"])
        elif ref == "paper" and an.get("text"):
            x = 0.07 + an.get("x", 0.5) * 0.9
            y = bottom + an.get("y", 1) * (top - bottom)
            f.text(x, y - 0.045 if many_panels else y, an["text"], ha="center", va="bottom",
                   fontsize=9, color=t["ink"])
    if has_legend:
        handles, labels = [], []
        for ax in axes.values():
            h, lab = ax.get_legend_handles_labels()
            for hh, ll in zip(h, lab, strict=True):
                if ll not in labels:
                    handles.append(hh), labels.append(ll)
        f.legend(handles, labels, loc="upper right", ncol=min(len(labels), 6), fontsize=8,
                 bbox_to_anchor=(0.97, 0.90))
    buf = io.BytesIO()
    f.savefig(buf, format="png", facecolor=t["surface"])
    plt.close(f)
    _save_png(buf, path)


def _save_png(buf, path):
    from PIL import Image

    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        path, optimize=True)


# ---------------------------------------------------------------------------
# Entry point used by the figure scripts
# ---------------------------------------------------------------------------
def _decimate(d: dict, max_side=128) -> dict:
    """Keep one row and column in ``k`` of heatmaps larger than ``max_side``.

    The axes keep the original indices, so hovering still reads the true band.
    """
    for tr in d["data"]:
        if tr.get("type") != "heatmap":
            continue
        z = np.asarray(decode(tr["z"]), dtype=np.float32)
        if z.ndim == 2 and max(z.shape) > max_side:
            k = -(-max(z.shape) // max_side)
            tr["z"] = _encode_f4(np.round(z[::k, ::k], 3))
            tr["x"], tr["y"] = list(range(0, z.shape[1], k)), list(range(0, z.shape[0], k))
        else:
            tr["z"] = _encode_f4(np.round(z, 3))
    return d


def write_figure(fig, out: Path, kind: str) -> list[Path]:
    """Write ``kind`` into ``out``: JSON when interactive, light/dark PNG otherwise."""
    out.mkdir(parents=True, exist_ok=True)
    d = figure_dict(fig)
    if kind in INTERACTIVE:
        path = out / f"{kind}.json"
        path.write_text(json.dumps(_compact(_decimate(d)), separators=(",", ":"), allow_nan=False))
        return [path]
    paths = [out / f"{kind}.png", out / f"{kind}_dark.png"]
    render_static(d, paths[0], "light")
    render_static(d, paths[1], "dark")
    return paths


def migrate(root: Path) -> None:
    for html in sorted(root.rglob("*.html")):
        written = write_figure(extract_html(html), html.parent, html.stem)
        print(html.relative_to(root), html.stat().st_size, "->",
              {p.name: p.stat().st_size for p in written})
        html.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--migrate", type=Path, required=True,
                    help="figures folder whose <kind>.html files are converted")
    migrate(ap.parse_args().migrate)
