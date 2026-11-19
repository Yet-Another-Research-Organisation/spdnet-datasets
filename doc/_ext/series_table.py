"""Sphinx directive ``series-table``: covariance-series modes read from the registry.

The rows come from :data:`spdnet_datasets.series.SERIES_MODES`, so the tables cannot drift
from the code. ``S``, ``p`` and ``n_t`` are those of a typical sample (the mode's
``example_shape``, every observation valid)::

    ```{series-table} esc50
    ```

    ```{series-table}
    :family: vision
    ```

With a dataset name: the modes of that dataset. Without one: every dataset, restricted to one
``family`` (``temporal`` or ``vision``), each dataset linked to its page.
"""

from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

PAGES = {"uav": "uav_hsi_crop"}  # registry name -> page, when they differ
FAMILY = {"temporal": "time series", "vision": "vision tokens"}


def _value(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, dict):
        return f"{len(v)} groups"
    if isinstance(v, tuple | list):
        return "(" + ", ".join(_value(x) for x in v) + ")"
    return str(v)


def _defaults(mode):
    return ", ".join(f"`{k}={_value(v)}`" for k, v in mode.defaults.items()
                     if v is not None and not (k == "cover" and v is True)) or "--"


def _row(mode):
    S, p, n_t = mode.nominal()
    shape = " × ".join(f"{a} {n}" for a, n in zip(mode.axes, mode.example_shape, strict=True))
    small = " (LW, `allow_small_n`)" if mode.allow_small_n else ""
    return [str(S), str(p), f"{n_t:,}".replace(",", " "), f"{n_t / p:.1f}{small}",
            mode.estimator.replace("ledoit_wolf", "Ledoit-Wolf").replace("scm", "SCM"),
            shape, _defaults(mode)]


class SeriesTable(SphinxDirective):
    optional_arguments = 1
    has_content = False
    option_spec = {"family": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        from spdnet_datasets.series import SERIES_MODES

        if self.arguments:
            name = self.arguments[0].strip()
            lines = ["| Mode | Family | Token | S | p | n_t | n_t / p | Estimator | Sample | "
                     "Defaults |", "|---|---|---|---|---|---|---|---|---|---|"]
            for i, mode in enumerate(SERIES_MODES[name].values()):
                label = f"`{mode.name}`" + (" (default)" if i == 0 else "")
                lines.append("| " + " | ".join([label, FAMILY[mode.family], mode.description,
                                                *_row(mode)]) + " |")
        else:
            family = self.options.get("family", "temporal")
            lines = ["| Dataset | Mode | S | p | n_t | n_t / p | Estimator | Sample | "
                     "Defaults |", "|---|---|---|---|---|---|---|---|---|"]
            for name, modes in SERIES_MODES.items():
                for i, mode in enumerate(modes.values()):
                    if mode.family != family:
                        continue
                    page = PAGES.get(name, name)
                    label = f"`{mode.name}`" + (" (default)" if i == 0 else "")
                    lines.append("| " + " | ".join([f"{{doc}}`{name} <{page}>`", label,
                                                    *_row(mode)]) + " |")
        container = nodes.container(classes=["series-table"])
        self.state.nested_parse(StringList(lines, source="series-table"), self.content_offset,
                                container)
        return [container]


def setup(app):
    app.add_directive("series-table", SeriesTable)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
