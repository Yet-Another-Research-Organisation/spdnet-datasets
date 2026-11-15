.. _examples:

Examples
========

Runnable scripts, each rendered as a page with its code, output and figures.
Every page can be downloaded as a Python script or a Jupyter notebook (links at
the bottom of each page, and for the whole gallery at the bottom of this page).

- **Tutorials** run when the documentation is built. They use synthetic or
  tiny fake data, so they work anywhere, without any dataset.
- **Dataset examples** load one real dataset per family. They read
  ``$SPDNET_DATA_ROOT`` and are therefore *not* executed when the documentation
  is built; their figures are the interactive figures of the dataset pages.

There are no Binder or Colab links: the datasets are not public, and the
tutorials only need ``pip install spdnet-datasets``.
