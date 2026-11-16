# Installation

spdnet-datasets needs Python 3.11 or later and PyTorch.

```bash
pip install "spdnet-datasets @ git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets"
```

Optional dependencies are grouped by family:

| Extra | For |
|---|---|
| `robust` | Tyler and Student-t estimators (`yetanotherspdnet`) |
| `synthetic` | synthetic SPD datasets (`yetanotherspdnet`) |
| `audio` | ESC-50, GTZAN, BirdSet (`pyarrow`, `soundfile`) |
| `sits` | PASTIS-R (`geopandas`) |
| `sar` | SAR datasets (`zarr`, `rasterio`, `Pillow`) |
| `radar` | CI4R (`Pillow`) |

For development, with the tests and the documentation:

```bash
git clone https://github.com/Yet-Another-Research-Organisation/spdnet-datasets.git
cd spdnet-datasets && pip install -e ".[dev,test,docs]"
pip install -r doc/requirements.txt   # yetanotherspdnet (Tyler / Student-t) for the tutorials
make -C doc html          # strict build; runs the tutorials of the gallery
```

## Building and publishing the documentation

The site is built from `doc/` with Sphinx (warnings are errors) and published
on GitHub Pages by the workflow `.github/workflows/docs.yml`:

| Event | What the workflow does |
|---|---|
| pull request touching `doc/`, `examples/`, `src/` or `pyproject.toml` | builds the site, so that a broken page is caught before merging |
| push to `main`, published release, manual run (*Actions > Documentation > Run workflow*) | builds and deploys the site |

The repository's Pages source is **GitHub Actions** (*Settings > Pages >
Build and deployment > Source*): the built HTML (`doc/_build/html`) is uploaded
as an artifact, so it does not have to be committed nor to live in a `docs/`
folder. The CI has no dataset: only the tutorials of the gallery, which use
synthetic data, are executed; the dataset examples are shown with their
precomputed figures. Locally, `SPDNET_DATA_ROOT=/nonexistent make -C doc html`
reproduces the CI build.

The published site: <https://yet-another-research-organisation.github.io/spdnet-datasets/>

## Where the data live

The datasets themselves are not distributed with the package. They are read
from `$SPDNET_DATA_ROOT/<DOMAIN>/<Name>` (default `/DATA/DATASET`) or fetched
into `$SPDNET_CACHE_DIR` (default `~/.cache/spdnet_datasets`); see
{doc}`/guide/fetch`.
