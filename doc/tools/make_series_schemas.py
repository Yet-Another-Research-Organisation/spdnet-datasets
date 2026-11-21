"""Draw the covariance-series diagrams of the dataset pages (no data needed).

For every dataset and mode of :data:`spdnet_datasets.series.SERIES_MODES`, the mode cuts a
zero sample of its ``example_shape`` and the figure shows

1. the raw sample with its axes (a rectangle for ``T x F``, a box for three axes), the token
   windows coloured in series order (one hue per outer window and lighter steps for the inner
   windows of a product mode, a single-hue ramp otherwise; token 1 filled, token S numbered);
2. one token: ``n_t`` observations of ``p`` variables, and its ``p x p`` covariance;
3. the resulting ``S x p x p`` series.

Output: ``doc/_static/figures/<page>/series_<mode>.png`` and ``series_<mode>_dark.png``,
embedded by the ``dataset-figure`` directive. Colours: the palette and surfaces of
``figures_io.py`` (same as the other figures)::

    PYTHONPATH=src python doc/tools/make_series_schemas.py            # every dataset
    PYTHONPATH=src python doc/tools/make_series_schemas.py esc50       # some of them
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_io import LIGHT_TO_DARK, THEMES  # noqa: E402

DOC = Path(__file__).resolve().parents[1]
OUT = DOC / "_static" / "figures"
PAGES = {"uav": "uav_hsi_crop"}
PALETTE = list(LIGHT_TO_DARK)  # categorical order of the dataviz palette (light steps)
AXIS_NAMES = {
    "audio": {"T": "frames", "F": "MFCC / log-mel bands"},
    "radar": {"T": "time", "F": "Doppler bins"},
    "hsi": {"H": "rows", "W": "columns", "B": "bands"},
    "skeleton": {"T": "frames", "J": "joints", "C": "x, y, z"},
    "sits": {"T": "dates / periods", "N": "parcel pixels", "B": "variables"},
    "sar": {"H": "rows", "W": "columns", "F": "log-powers"},
    "sar_grid": {"H": "rows", "W": "columns", "F": "features"},
    "sar_dates": {"T": "dates", "N": "block pixels", "P": "polarisations"},
    "eeg": {"T": "time samples", "C": "channels", "f": "frequency, Hz"},
    "fmri": {"T": "volumes", "R": "regions"},
}
FAMILY_OF = {"esc50": "audio", "gtzan": "audio", "birdset_xcm": "audio", "ci4r": "radar",
             "mvdoppler": "radar", "dopnet": "radar", "ntu120": "skeleton",
             "hdm05": "skeleton", "pastis_r": "sits", "pol_insar_island": "sar",
             "cpazmal": "sar_dates", "s1slc_cvdl": "sar_grid", "atrnet_star": "sar_grid",
             "air_polsar_seg": "sar_grid", "mstar": "sar_grid", "moabb": "eeg",
             "nilearn": "fmri"}
# drawing axes of a 3-axis sample: (horizontal, vertical, depth)
LAYOUT3 = {"hsi": (1, 0, 2), "sar": (1, 0, 2), "sar_grid": (1, 0, 2), "sits": (0, 1, 2),
           "sar_dates": (0, 1, 2), "skeleton": (0, 1, 2)}


def _rgb(c):
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _mix(c, surface, t):
    """Colour ``c`` moved a fraction ``t`` towards the surface (lighter steps)."""
    return tuple((1 - t) * _rgb(c) + t * _rgb(surface))


def _intervals(pos):
    """Per axis: (start, stop), or a list of indices (a set of joints)."""
    return [p if isinstance(p, list) else tuple(p) for p in pos]


def _lanes(positions, colours, axis):
    """Distinct windows along ``axis`` in series order, with the colour of their first token;
    empty when the axis is not cut."""
    seen = {}
    for p, col in zip(positions, colours, strict=True):
        v = p[axis]
        if isinstance(v, list):
            return []
        seen.setdefault(tuple(v), col)
    return list(seen.items()) if len(seen) > 1 else []


def token_colours(positions, theme):
    """Colour of each token: hue per outer window, lighter steps for the inner ones."""
    surface = THEMES[theme]["surface"]
    pal = [LIGHT_TO_DARK[c] if theme == "dark" else c for c in PALETTE]
    pos = [_intervals(p) for p in positions]
    varying = [a for a in range(len(pos[0])) if len({str(p[a]) for p in pos}) > 1]
    if len(pos) > 1 and len(varying) > 1:  # product mode: the axis that changes first is inner
        inner = next(a for a in varying if str(pos[1][a]) != str(pos[0][a]))
        outer = [a for a in varying if a != inner]
        keys = [tuple(str(p[a]) for a in outer) for p in pos]
        groups = list(dict.fromkeys(keys))
        n_inner = max(keys.count(k) for k in groups)
        colours, seen = [], {}
        for k in keys:
            j = seen.get(k, 0)
            seen[k] = j + 1
            colours.append(_mix(pal[groups.index(k) % len(pal)], surface,
                                0.45 * j / max(n_inner - 1, 1)))
        return colours
    n = len(pos)
    return [_mix(pal[0], surface, 0.45 * (1 - k / max(n - 1, 1))) for k in range(n)]


def _spans(v):
    """``(start, stop)`` spans of one axis position (one per index for a set of indices)."""
    return [(i, i + 1) for i in v] if isinstance(v, list) else [tuple(v)]


def _draw_2d(ax, mode, shape, positions, colours, t, names, box):
    from matplotlib.patches import Rectangle

    x0, y0, W, H = box
    nT, nF = shape[:2]
    a0, a1 = mode.axes[:2]
    # variables after pooling: the size the positions refer to
    pooled_F = nF if isinstance(positions[0][1], list) else max(p[1][1] for p in positions)
    ax.add_patch(Rectangle((x0, y0), W, H, facecolor=t["mid"], edgecolor=t["muted"], lw=0.8))

    def rects(p):
        (a, b) = _spans(p[0])[0]
        return [(x0 + W * a / nT, y0 + H * (1 - d / pooled_F), W * (b - a) / nT,
                 H * (d - c) / pooled_F) for c, d in _spans(p[1])]

    for k, (p, col) in enumerate(zip(positions, colours, strict=True)):
        for x, y, w, h in rects(p):
            ax.add_patch(Rectangle((x, y), w, h, facecolor=(*col, 0.55 if k == 0 else 0.16),
                                   edgecolor=col, lw=1.0 if k == 0 else 0.6))
    for k in (0, len(positions) - 1):
        x, y, w, h = rects(positions[k])[0]
        ax.text(x + w / 2, y + h / 2, "1" if k == 0 else f"S={len(positions)}", ha="center",
                va="center", fontsize=8, color=t["ink"], fontweight="bold")
    for k, ((a, b), col) in enumerate(_lanes(positions, colours, 0)):  # windows along axis 0
        y = y0 - 0.12 - 0.13 * (k % 2)
        ax.plot([x0 + W * a / nT, x0 + W * b / nT], [y, y], color=col, lw=2.2,
                solid_capstyle="butt")
    for k, ((a, b), col) in enumerate(_lanes(positions, colours, 1)):  # windows along axis 1
        x = x0 - 0.12 - 0.13 * (k % 2)
        ax.plot([x, x], [y0 + H * (1 - a / pooled_F), y0 + H * (1 - b / pooled_F)], color=col,
                lw=2.2, solid_capstyle="butt")
    ax.annotate("", (x0 + W, y0 - 0.5), (x0, y0 - 0.5),
                arrowprops={"arrowstyle": "->", "color": t["muted"], "lw": 0.8})
    ax.text(x0 + W / 2, y0 - 0.57, f"{names[a0]} ({a0} = {nT})", ha="center", va="top",
            fontsize=8, color=t["muted"])
    label = f"{names[a1]} ({a1} = {nF}" + (f" → {pooled_F})" if pooled_F != nF else ")")
    ax.text(x0 - 0.5, y0 + H / 2, label, ha="right", va="center", rotation=90, fontsize=8,
            color=t["muted"])


def _draw_3d(ax, mode, shape, positions, colours, t, names, box, layout):
    from matplotlib.patches import Polygon

    x0, y0, W, H = box
    ih, iv, idp = layout
    dx, dy = 1.1, 0.75  # depth direction
    # depth = variables, possibly pooled: the size the positions refer to
    depth = (shape[idp] if isinstance(positions[0][idp], list)
             else max(p[idp][1] for p in positions))
    sizes = [shape[ih], shape[iv], depth]

    def pt(h, v, d):  # vertical axis drawn top-down (row 0 at the top)
        return (x0 + W * h / sizes[0] + dx * d / sizes[2],
                y0 + H * (1 - v / sizes[1]) + dy * d / sizes[2])

    def faces(h0, h1, v0, v1, d0, d1):
        return [[pt(h0, v0, d0), pt(h1, v0, d0), pt(h1, v1, d0), pt(h0, v1, d0)],   # front
                [pt(h0, v0, d0), pt(h1, v0, d0), pt(h1, v0, d1), pt(h0, v0, d1)],   # top
                [pt(h1, v0, d0), pt(h1, v1, d0), pt(h1, v1, d1), pt(h1, v0, d1)]]   # side

    for f in faces(0, sizes[0], 0, sizes[1], 0, sizes[2]):
        ax.add_patch(Polygon(f, closed=True, facecolor=t["mid"], edgecolor=t["muted"], lw=0.8))

    def spans(p, axis):
        v = p[axis]
        if isinstance(v, list):  # set of indices: one span per index
            return [(i, i + 1) for i in v]
        return [tuple(v)]

    for k, (p, col) in enumerate(zip(positions, colours, strict=True)):
        for (h0, h1) in spans(p, ih):
            for (v0, v1) in spans(p, iv):
                d0, d1 = spans(p, idp)[0]
                for f in faces(h0, h1, v0, v1, d0, d1):
                    ax.add_patch(Polygon(f, closed=True,
                                         facecolor=(*col, 0.6 if k == 0 else 0.13),
                                         edgecolor=col, lw=0.9 if k == 0 else 0.5))
    for k, ((a, b), col) in enumerate(_lanes(positions, colours, ih)):
        (xa, ya), (xb, _) = pt(a, sizes[1], 0), pt(b, sizes[1], 0)
        y = ya - 0.12 - 0.13 * (k % 2)
        ax.plot([xa, xb], [y, y], color=col, lw=2.2, solid_capstyle="butt")
    for k, ((a, b), col) in enumerate(_lanes(positions, colours, iv)):
        (xa, ya), (_, yb) = pt(0, a, 0), pt(0, b, 0)
        x = xa - 0.12 - 0.13 * (k % 2)
        ax.plot([x, x], [ya, yb], color=col, lw=2.2, solid_capstyle="butt")
    for k, ((a, b), col) in enumerate(_lanes(positions, colours, idp)):
        (xa, ya), (xb, yb) = pt(sizes[0], sizes[1], a), pt(sizes[0], sizes[1], b)
        off = 0.14 + 0.13 * (k % 2)
        ax.plot([xa + off, xb + off], [ya - off * 0.4, yb - off * 0.4], color=col, lw=2.2,
                solid_capstyle="butt")
    first = positions[0]
    (h0, h1), (v0, v1) = spans(first, ih)[0], spans(first, iv)[0]
    cx, cy = pt((h0 + h1) / 2, (v0 + v1) / 2, spans(first, idp)[0][0])
    ax.text(cx, cy, "1", ha="center", va="center", fontsize=8, color=t["ink"],
            fontweight="bold")
    a = mode.axes
    ax.text(x0 + W / 2, y0 - 0.42, f"{names[a[ih]]} ({a[ih]} = {shape[ih]})", ha="center",
            va="top", fontsize=8, color=t["muted"])
    ax.text(x0 - 0.45, y0 + H / 2, f"{names[a[iv]]} ({a[iv]} = {shape[iv]})", ha="right",
            va="center", rotation=90, fontsize=8, color=t["muted"])
    ex, ey = pt(sizes[0], 0, sizes[2] / 2)
    ax.text(ex + 0.5, ey + 0.05, f"{names[a[idp]]}\n({a[idp]} = {shape[idp]})", ha="left",
            va="center", fontsize=8, color=t["muted"])


def _draw_series(ax, S, p, n_t, colours, t, box):
    from matplotlib.patches import Rectangle

    x0, y0 = box[:2]
    # one token: n_t x p observations -> p x p
    ax.add_patch(Rectangle((x0, y0 + 0.2), 0.45, 2.0, facecolor=(*colours[0], 0.55),
                           edgecolor=colours[0], lw=0.9))
    ax.text(x0 + 0.225, y0 + 0.05, f"n_t = {n_t:,}".replace(",", " "), ha="center", va="top",
            fontsize=8, color=t["ink"])
    ax.text(x0 + 0.225, y0 + 2.3, f"p = {p}", ha="center", va="bottom", fontsize=8,
            color=t["ink"])
    ax.annotate("", (x0 + 1.25, y0 + 1.2), (x0 + 0.6, y0 + 1.2),
                arrowprops={"arrowstyle": "->", "color": t["muted"], "lw": 0.9})
    ax.text(x0 + 0.92, y0 + 1.3, "cov", ha="center", va="bottom", fontsize=8, color=t["muted"])
    # series: stack of S matrices (at most 8 drawn)
    shown = list(range(min(S, 7))) + ([S - 1] if S > 7 else [])
    side, off = 1.25, 0.2
    sx, sy = x0 + 1.4, y0 + 0.55
    for j, k in enumerate(shown):
        ax.add_patch(Rectangle((sx + off * j, sy + off * j * 0.6), side, side,
                               facecolor=(*colours[k], 0.9), edgecolor=t["surface"], lw=1.2))
    end = sx + off * (len(shown) - 1) + side
    ax.text((sx + end) / 2, y0 + 0.3, f"S × p × p = {S} × {p} × {p}", ha="center", va="top",
            fontsize=8, color=t["ink"])
    if S > 7:
        ax.text(end + 0.1, sy + off * len(shown) * 0.6 + side / 2, "⋯", fontsize=10,
                color=t["muted"], va="center")


def draw(name, mode, path, theme):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    t = THEMES[theme]
    x = np.zeros(mode.example_shape, dtype=np.float32)
    if mode.prepare is not None:
        x = mode.prepare(x)
    obs, positions = mode.tokenize(x, **mode.defaults)
    S, p, n_t = len(obs), obs[0].shape[1], min(len(o) for o in obs)
    colours = token_colours(positions, theme)
    kind = FAMILY_OF.get(name, "hsi")
    names = AXIS_NAMES[kind]

    fig = plt.figure(figsize=(9.6, 3.1), dpi=110)
    fig.patch.set_facecolor(t["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")
    order = {"temporal": "time series", "vision": "vision tokens"}[mode.family]
    ax.text(0.3, 4.95, f"{name} · {mode.name}  ({order})", fontsize=10, color=t["ink"],
            fontweight="bold", va="top")
    ax.text(0.3, 4.55, mode.description, fontsize=8, color=t["muted"], va="top", wrap=True)
    if mode.name == "filter_bank":  # tokens = frequency bands of the whole epoch
        from dataclasses import replace

        bands = mode.defaults["bands"]
        positions = [[(0, x.shape[0]), (int(lo), int(hi))] for lo, hi in bands]
        top = max(b[1] for b in bands)
        _draw_2d(ax, replace(mode, axes=("T", "f")), (x.shape[0], top), positions, colours,
                 t, names, (1.2, 0.9, 6.0, 2.9))
    elif kind == "skeleton":  # frames x joints; the 3 coordinates are always all used
        positions = [list(p[:2]) for p in positions]
        _draw_2d(ax, mode, x.shape[:2], positions, colours, t, names, (1.2, 0.9, 6.0, 2.9))
    elif x.ndim == 2:
        _draw_2d(ax, mode, x.shape, positions, colours, t, names, (1.2, 0.9, 6.0, 2.9))
    else:
        layout = LAYOUT3.get(kind, (1, 0, 2))
        _draw_3d(ax, mode, x.shape, positions, colours, t, names, (1.2, 0.7, 4.8, 2.4), layout)
    _draw_series(ax, S, p, n_t, colours, t, (9.0, 0.7))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=t["surface"])
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").quantize(colors=96, method=Image.Quantize.MEDIANCUT,
                                            dither=Image.Dither.NONE).save(path, optimize=True)


def main(argv=None):
    from spdnet_datasets.series import SERIES_MODES

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("datasets", nargs="*", help="registry names (default: all)")
    args = parser.parse_args(argv)
    for name in args.datasets or SERIES_MODES:
        folder = OUT / PAGES.get(name, name)
        folder.mkdir(parents=True, exist_ok=True)
        for mode in SERIES_MODES[name].values():
            for theme, suffix in (("light", ""), ("dark", "_dark")):
                draw(name, mode, folder / f"series_{mode.name}{suffix}.png", theme)
        print(f"{name}: {len(SERIES_MODES[name])} modes -> {folder}")


if __name__ == "__main__":
    main()
