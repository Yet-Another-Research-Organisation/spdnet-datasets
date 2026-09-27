"""Sphinx directive ``dataset-catalogue``: one row per registered dataset.

Reads ``DatasetManager.DATASETS`` after importing ``spdnet_datasets.real``, so
a dataset registered with ``@DatasetManager.register_dataset`` appears in the
documentation without editing it::

    ```{dataset-catalogue}
    ```
"""

from __future__ import annotations

import inspect

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective


def _summary(cls) -> str:
    doc = inspect.getdoc(cls) or ""
    return doc.strip().split("\n")[0].rstrip(".").replace("|", "\\|")


def _options(cls) -> str:
    """Dataset-specific constructor arguments (not those of BaseDataset)."""
    params = list(inspect.signature(cls.__init__).parameters.values())[1:]
    shown = []
    for p in params:
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) or p.name == "data_dir":
            continue
        default = "" if p.default is inspect.Parameter.empty else f"={p.default!r}"
        shown.append(f"`{p.name}{default}`")
    return ", ".join(shown) or "--"


class DatasetCatalogue(SphinxDirective):
    has_content = False

    def run(self) -> list[nodes.Node]:
        import spdnet_datasets.real  # noqa: F401  (registers the datasets)
        from spdnet_datasets import DatasetManager

        lines = [
            "| `name` | Class | Description | Specific options |",
            "|---|---|---|---|",
        ]
        for name in sorted(DatasetManager.DATASETS):
            cls = DatasetManager.DATASETS[name]
            lines.append(
                f"| `{name}` | {{py:class}}`~{cls.__module__}.{cls.__name__}` | "
                f"{_summary(cls)} | {_options(cls)} |"
            )
        container = nodes.container(classes=["dataset-catalogue"])
        self.state.nested_parse(
            StringList(lines, source="dataset-catalogue"), self.content_offset, container
        )
        return [container]


def setup(app):
    app.add_directive("dataset-catalogue", DatasetCatalogue)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
