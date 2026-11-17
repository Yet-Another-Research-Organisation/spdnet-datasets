# Adding a dataset

A dataset is a subclass of `BaseDataset` registered under a name. Nothing
else needs wiring: the manager, the Hydra configs of `spdnet-training` and
the catalogue of {doc}`catalogue` pick it up.

```python
# src/spdnet_datasets/real/my_dataset.py
from pathlib import Path

import numpy as np
import torch

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.manager import DatasetManager


@DatasetManager.register_dataset("my_dataset")
class MyDataset(BaseDataset):
    """One-line description, shown in the catalogue.

    Expected structure:
        data_dir/<class>/<sample>.npy   # (n, n) covariance matrices
    """

    def __init__(self, scaling_factor: float = 1.0, **kwargs):
        super().__init__(**kwargs)          # data_dir, max_classes, seed, ...
        self.scaling_factor = scaling_factor
        self._load_data()

    def _load_data(self):
        classes = sorted(p.name for p in Path(self.data_dir).iterdir() if p.is_dir())
        self.classes = self._limit_classes(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        by_class = {
            c: [(path, self.class_to_idx[c]) for path in sorted((Path(self.data_dir) / c).glob("*.npy"))]
            for c in self.classes
        }
        self.samples = self._limit_samples_per_class(by_class)  # max_samples_per_class

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = torch.from_numpy(np.load(path)).double() * self.scaling_factor
        return x, label
```

Then export it in `src/spdnet_datasets/real/__init__.py` (the import runs the
decorator):

```python
from .my_dataset import MyDataset
```

## Conventions

- Return `(x, label)` with `x` a **float64** tensor: SPD matrices `(n, n)`,
  or `(C, H, W)` images for an image mode.
- Accept `**kwargs` and pass them to `BaseDataset.__init__`, which handles
  `data_dir`, `classes`, `max_classes`, `max_samples_per_class`, `seed` and
  `verbose`.
- Use `self._limit_classes` and `self._limit_samples_per_class` so that the
  subset options behave the same across datasets.
- Add a test that loads a few samples when the data are available (see
  `tests/`).

## Documentation page

Every dataset has a page in `doc/datasets/`. The build is strict (`-W`): a
page that is in no toctree, or a figure that is missing, fails it.

1. Copy `doc/datasets/_template.md` to `doc/datasets/<name>.md` and fill every
   section.
2. On a machine that holds the data, draw its figures and previews:

   ```bash
   PYTHONPATH=src python doc/tools/make_figures.py <name> --kwargs '{...}' --split <split>
   PYTHONPATH=src python doc/tools/make_previews.py <name>   # after adding a function for it
   ```

   They are written to `doc/_static/figures/<name>/` and
   `doc/_static/previews/<name>/`; commit them, the CI has no data.
3. Add the page to the toctree **and** the tiles of its domain page
   (`doc/datasets/audio.md`, `hyperspectral.md`, `sar.md`, …).
4. Write its publication in `doc/datasets/_cite/<name>.md` (resolve the DOI on
   <https://doi.org>; without a DOI, link the source page), end the page with
   its *How to cite* block and add it to `doc/references.md`.

```{tip}
`SPDNET_DATA_ROOT=/nonexistent make -C doc html` reproduces the CI build.
```
