"""Draw the gallery thumbnails of the dataset examples from the figure summaries.

The dataset examples (``examples/datasets/``) are not executed when the
documentation is built, so their thumbnails are fixed images. Each one shows the
class counts per split read from ``doc/_static/figures/<page>/summary.json``
(written by ``make_figures.py``), so it needs neither the data nor a browser::

    python doc/tools/make_thumbnails.py esc50 ci4r_77ghz:ci4r pastis_r chikusei mstar

An argument ``<figures dir>[:<thumbnail name>]`` writes
``doc/_static/thumbnails/<thumbnail name>.png``.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOC = Path(__file__).resolve().parents[1]
SPLITS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}  # dataviz palette order


def thumbnail(figures_dir: Path, out: Path) -> None:
    summary = json.loads((figures_dir / "summary.json").read_text())
    counts = {k: np.asarray(v, float) for k, v in summary["counts"].items() if k in SPLITS}
    order = np.argsort(-sum(counts.values()))
    fig, ax = plt.subplots(figsize=(4, 2.8), dpi=100)
    bottom = np.zeros(len(order))
    for split, color in SPLITS.items():
        if split in counts:
            h = counts[split][order]
            ax.bar(np.arange(len(order)), h, bottom=bottom, width=0.8, color=color,
                   edgecolor="white", linewidth=0.6 if len(order) < 30 else 0)
            bottom += h
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_axis_off()
    fig.subplots_adjust(0.04, 0.04, 0.96, 0.96)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pages", nargs="+", help="<figures dir>[:<thumbnail name>]")
    args = parser.parse_args()
    for page in args.pages:
        src, _, name = page.partition(":")
        out = DOC / "_static" / "thumbnails" / f"{name or src}.png"
        thumbnail(DOC / "_static" / "figures" / src, out)
        print("wrote", out.relative_to(DOC))


if __name__ == "__main__":
    main()
