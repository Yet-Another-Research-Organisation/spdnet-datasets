"""Sphinx configuration for spdnet-datasets."""

import os
import re
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("_ext"))

project = "spdnet-datasets"
author = "Matthieu Gallet, Ammar Mian"
copyright = f"{datetime.now().year}, {author}"
try:
    release = package_version("spdnet-datasets")
except PackageNotFoundError:
    release = "dev"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "catalogue",  # _ext/catalogue.py: table of the registered datasets
    "dataset_figures",  # _ext/dataset_figures.py: fixed or interactive dataset figures
]

# Example gallery (../examples). Only the tutorials run at build time: they use
# synthetic data. The dataset examples read $SPDNET_DATA_ROOT, which the CI does
# not have; they are rendered without execution and embed the Plotly figures of
# the dataset pages. No Binder / Colab / JupyterLite links: the data are private.
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": re.escape(os.sep) + "tutorials" + re.escape(os.sep) + "plot_",
    "subsection_order": ["../examples/tutorials", "../examples/datasets", "*"],
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": True,
    "remove_config_comments": True,
    "image_scrapers": ("matplotlib",),
    "reference_url": {"spdnet_datasets": None},
    "abort_on_example_error": True,
    "show_memory": False,
}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3

# API reference: hand-written pages with autodoc (see doc/reference/)
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_default_options = {"exclude-members": "__init__, __new__"}
add_module_names = False
# Optional or heavy dependencies are not installed in the documentation build
autodoc_mock_imports = ["yetanotherspdnet", "h5py", "zarr", "pyarrow", "soundfile",
                        "rasterio", "geopandas", "spectral", "tifffile", "moabb"]
python_maximum_signature_line_length = 88
autosummary_generate = False

# Docstrings are in Google style ("Args:", "Returns:")
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "datasets/_template.md", "datasets/_cite", "tools"]

# ---------------------------------------------------------------------------
# HTML: pydata-sphinx-theme, dark by default, one accent colour per section
# (doc/_static/custom.css keys the accent off <body data-section>, set by
# doc/_templates/layout.html from the first path segment of the page).
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "spdnet-datasets"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]
html_js_files = []
html_favicon = "_static/favicon.png"
html_logo = "_static/logo.png"
html_show_sourcelink = False
html_context = {
    "default_mode": "dark",
    "github_user": "Yet-Another-Research-Organisation",
    "github_repo": "spdnet-datasets",
    "github_version": "main",
    "doc_path": "doc",
}
html_theme_options = {
    "logo": {"text": "spdnet-datasets", "alt_text": "spdnet-datasets"},
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Yet-Another-Research-Organisation/spdnet-datasets",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_toc_level": 2,
    "show_nav_level": 1,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "pygments_light_style": "tango",
    "pygments_dark_style": "github-dark",
}
# no left sidebar on the landing page and the references page
html_sidebars = {"index": [], "references": []}
