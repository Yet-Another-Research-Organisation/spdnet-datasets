"""Sphinx directive ``dataset-figure``: one figure of a dataset page.

The figures live in ``doc/_static/figures/<page>/`` and are produced on a
machine that holds the data by ``doc/tools/make_figures.py`` and
``make_figures_raw.py`` (see ``doc/tools/figures_io.py``):

- ``<kind>.png`` and ``<kind>_dark.png``: a fixed figure, shown in the light or
  the dark theme;
- ``<kind>.json``: an interactive Plotly figure, embedded as a ``<div>`` that
  ``_static/plotly-figures.js`` fills when it scrolls into view. plotly.js is
  loaded once per page, and only on pages that have such a figure.

MyST::

    ```{dataset-figure} esc50/cov_samples
    ```

reStructuredText (sphinx-gallery examples)::

    .. dataset-figure:: esc50/cov_samples

``dataset-preview <name>`` shows ``_static/previews/<name>/samples.png`` (and its
dark variant), drawn by ``doc/tools/make_previews.py``.

A missing figure is a build warning (an error with ``-W``).
"""

from __future__ import annotations

import html
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)

ALT = {
    "raw_samples": "Examples of the preprocessed data, one per class",
    "class_hist": "Number of samples per class in each part of the split",
    "class_means": "Mean channel profile per class",
    "cov_samples": "A few covariance matrices shown as correlations (interactive)",
    "eigen_spectrum": "Median and 10-90 % band of the sorted log10 eigenvalues (interactive)",
    "split_schema": "Split sizes and number of samples per group",
    "amplitude_hist": "Distribution of the raw values per channel",
    "protocol_overlap": "Chips shared by the train and test parts of each pair of protocols "
                        "(interactive)",
    "split_map": "Map of the samples coloured by split part",
    "date_gaps": "Gap between the Sentinel-2 and Sentinel-1 acquisitions",
    "date_timeline": "Acquisition dates per tile",
    "dates": "Acquisition dates retained per orbit",
}


class DatasetFigure(SphinxDirective):
    required_arguments = 1
    has_content = False
    option_spec = {"alt": directives.unchanged, "caption": directives.unchanged,
                   "height": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        ref = self.arguments[0].strip().strip("/")
        static = Path(self.env.srcdir) / "_static" / "figures"
        depth = self.env.docname.count("/")
        url = "../" * depth + "_static/figures/" + ref
        kind = ref.rsplit("/", 1)[-1]
        default = (f"Covariance-series diagram of the {kind[7:]} mode: sample, tokens and "
                   "series" if kind.startswith("series_") else kind.replace("_", " "))
        alt = html.escape(self.options.get("alt", ALT.get(kind, default)))
        caption = self.options.get("caption")
        cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""

        if (static / f"{ref}.json").is_file():
            self.env.metadata[self.env.docname]["dataset_plotly"] = True
            style = f' style="height:{html.escape(self.options["height"])}"' \
                if "height" in self.options else ""
            out = (f'<figure class="spd-plotly"><div class="spd-plotly-div" role="img" '
                   f'aria-label="{alt}" data-src="{url}.json"{style}></div>{cap}</figure>')
        elif (static / f"{ref}.png").is_file():
            dark = (static / f"{ref}_dark.png").is_file()
            light_cls = ' class="only-light"' if dark else ""
            imgs = f'<img src="{url}.png" alt="{alt}" loading="lazy"{light_cls}>'
            if dark:
                imgs += f'<img src="{url}_dark.png" alt="{alt}" loading="lazy" class="only-dark">'
            out = f'<figure class="spd-figure">{imgs}{cap}</figure>'
        else:
            logger.warning("dataset-figure: no %s.json nor %s.png in _static/figures", ref, ref,
                           location=self.get_location())
            return []
        return [nodes.raw("", out, format="html")]


class DatasetPreview(SphinxDirective):
    """``dataset-preview <name>``: the examples drawn by ``doc/tools/make_previews.py``."""

    required_arguments = 1
    has_content = False
    option_spec = {"caption": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        name = self.arguments[0].strip()
        folder = Path(self.env.srcdir) / "_static" / "previews" / name
        if not (folder / "samples.png").is_file():
            logger.warning("dataset-preview: no _static/previews/%s/samples.png", name,
                           location=self.get_location())
            return []
        url = "../" * self.env.docname.count("/") + f"_static/previews/{name}/samples"
        caption = self.options.get("caption", "A few examples of the data, one per class.")
        alt = html.escape(f"Examples of the {name} data")
        imgs = f'<img src="{url}.png" alt="{alt}" class="only-light">'
        if (folder / "samples_dark.png").is_file():
            imgs += f'<img src="{url}_dark.png" alt="{alt}" class="only-dark">'
        else:
            imgs = imgs.replace(' class="only-light"', "")
        out = (f'<figure class="spd-figure spd-preview">{imgs}'
               f"<figcaption>{html.escape(caption)}</figcaption></figure>")
        return [nodes.raw("", out, format="html")]


def _add_plotly(app, pagename, templatename, context, doctree):
    if app.env.metadata.get(pagename, {}).get("dataset_plotly"):
        app.add_js_file("plotly-figures.js", loading_method="defer")


def setup(app):
    app.add_directive("dataset-figure", DatasetFigure)
    app.add_directive("dataset-preview", DatasetPreview)
    app.connect("html-page-context", _add_plotly)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
