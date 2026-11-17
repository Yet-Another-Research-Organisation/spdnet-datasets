# Installation

spdnet-datasets needs **Python 3.11 or later** and **PyTorch 2.0 or later**. The
package is installed from GitHub; the datasets themselves are not distributed
with it (see [Where the data live](#where-the-data-live)).

## Install the package

`````{tab-set}

````{tab-item} pip
```bash
python -m venv .venv && source .venv/bin/activate
pip install "spdnet-datasets @ git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets"

# with optional extras, e.g. audio and SAR readers
pip install "spdnet-datasets[audio,sar] @ git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets"
```
````

````{tab-item} uv
```bash
# in a uv project
uv add "spdnet-datasets @ git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets"
uv add "spdnet-datasets[audio,sar] @ git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets"

# or in an existing virtual environment
uv pip install "spdnet-datasets[audio,sar] @ git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets"
```
````

````{tab-item} Development
```bash
git clone https://github.com/Yet-Another-Research-Organisation/spdnet-datasets.git
cd spdnet-datasets
pip install -e ".[dev,test,docs]"        # or: uv pip install -e ".[dev,test,docs]"
pip install -r doc/requirements.txt      # yetanotherspdnet, for the robust estimators
pytest                                   # tests without data; real-data tests are skipped
```
````

`````

```{tip}
Extras combine: `spdnet-datasets[audio,sits,sar,radar]` installs the readers of
every family at once.
```

## Optional extras

The core package reads the hyperspectral, skeleton and radar covariances with
NumPy, SciPy, h5py, tifffile and spectral. The other readers are optional:

| Extra | Installs | Needed for |
|---|---|---|
| `robust` | `yetanotherspdnet` | Tyler and Student-t estimators (`estimator="tyler"` / `"student"`) |
| `synthetic` | `yetanotherspdnet` | {doc}`synthetic SPD datasets </datasets/synthetic>` |
| `audio` | `pyarrow`, `soundfile` | ESC-50, GTZAN, BirdSet XCM |
| `sits` | `geopandas` | PASTIS-R |
| `sar` | `zarr`, `rasterio`, `Pillow` | S1SLC_CVDL (zarr store), MSTAR, AIR-PolSAR-Seg, FUSAR-Ship |
| `radar` | `Pillow` | CI4R spectrograms (PNG) |
| `archives` | `py7zr`, `multivolumefile` | ATRNet-STAR `.7z` volumes when the `7z` command is missing |
| `gdrive` | `gdown` | optional download of the MVDoppler snapshots from Google Drive |
| `moabb` | `moabb>=1.7.2` | {doc}`MOABB adapter </datasets/moabb>` (EEG) |
| `nilearn` | `nilearn>=0.14.1` | {doc}`nilearn adapter </datasets/nilearn>` (fMRI connectomes) |
| `docs` | Sphinx, pydata-sphinx-theme, sphinx-design, … | building this documentation |
| `test`, `dev` | pytest; ruff, pre-commit | development |

```{important}
`yetanotherspdnet` (extras `robust` and `synthetic`) is not on PyPI. Install it
from GitHub first, then the extra:

    pip install "yetanotherspdnet @ git+https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet"
```

## PyTorch for CPU or GPU

pip installs the default PyTorch wheel of your platform (CUDA on Linux x86-64).
To choose, install PyTorch **before** spdnet-datasets, from the index that
matches your hardware (see [pytorch.org](https://pytorch.org/get-started/locally/)):

````{tab-set}
```{tab-item} CPU only
    pip install torch --index-url https://download.pytorch.org/whl/cpu
```
```{tab-item} CUDA 12.x
    pip install torch --index-url https://download.pytorch.org/whl/cu126
```
```{tab-item} uv
    # pyproject.toml of your project
    [tool.uv.sources]
    torch = { index = "pytorch-cpu" }

    [[tool.uv.index]]
    name = "pytorch-cpu"
    url = "https://download.pytorch.org/whl/cpu"
    explicit = true
```
````

The loaders only use the CPU; the GPU matters for the models trained on their
output.

## Where the data live

The datasets are read from a read-only copy, or fetched into a writable cache:

| Variable | Default | Role |
|---|---|---|
| `SPDNET_DATA_ROOT` | `/DATA/DATASET` | read-only copies, laid out as `<DOMAIN>/<Name>/` |
| `SPDNET_CACHE_DIR` | `~/.cache/spdnet_datasets` | downloads, preprocessed tensors, covariances |
| `SPDNET_OPEN_BROWSER` | `1` | `0` never opens the dataset page for a manual download |

```bash
export SPDNET_DATA_ROOT=/DATA/DATASET            # shared copy (if any)
export SPDNET_CACHE_DIR=/scratch/$USER/spdnet    # a large, fast disk
```

Every `fetch_<name>()` and every dataset class also accepts `data_dir=` and
`cache_dir=`. The staged cache and the checksums are described in
{doc}`/guide/fetch`.

## Credentials: Kaggle and Hugging Face

Most datasets download without an account. A few need credentials; they are
read from environment variables, the tool's own token file, the system keyring
or a hidden prompt, in that order, and are never written in logs or URLs.

`````{tab-set}

````{tab-item} Kaggle
HyperLeaf2024 and the wheat disease dataset are Kaggle competitions. On
<https://www.kaggle.com/settings/api>, click *Generate New Token*, then store
the access token without leaving it in the shell history:

```bash
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
read -rs T && printf %s "$T" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token && unset T
```

`KAGGLE_API_TOKEN` works too. The legacy `~/.kaggle/kaggle.json` (or
`KAGGLE_USERNAME` / `KAGGLE_KEY`) is still accepted when no access token is
found. Accept the competition rules on its page with the same account.
````

````{tab-item} Hugging Face
Gated datasets (ATRNet-STAR) need a read token: create one on
huggingface.co (*Settings > Access Tokens*), accept the dataset conditions on
its page, then

```bash
hf auth login            # writes ~/.cache/huggingface/token
```

or set `HF_TOKEN`. Public Hugging Face mirrors (ESC-50, GTZAN, BirdSet,
CPAZMaL) need no token.
````

````{tab-item} Keyring
On a shared machine, prefer the system keyring (optional `keyring` package):

```python
from spdnet_datasets.core.access import store_secret

store_secret("kaggle_token", "KGAT_...")
store_secret("huggingface", "hf_...")
```
````

`````

```{warning}
Without credentials, a fetch that needs them raises `MissingCredentialsError`
with these instructions. Datasets distributed by hand (MSTAR, S1SLC_CVDL, …)
raise `ManualDownloadRequired` and tell you what to download: see
{doc}`/guide/fetch`.
```

## Check the installation

```python
import spdnet_datasets
from spdnet_datasets import DatasetManager
import spdnet_datasets.real  # registers the loaders

print(sorted(DatasetManager.DATASETS))   # registered dataset names
```

Then follow the {doc}`quickstart`.

## Building the documentation

The site is built from `doc/` with Sphinx (warnings are errors), the
[pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io) and
sphinx-design:

```bash
pip install -e ".[docs]" -r doc/requirements.txt
SPDNET_DATA_ROOT=/nonexistent make -C doc html     # same build as the CI
python -m http.server -d doc/_build/html           # http://localhost:8000
```

The build needs **no dataset**: the tutorials of the gallery use synthetic
data, and the figures and previews of the dataset pages are precomputed files
under `doc/_static/`. They are regenerated on a machine that holds the data:

| Script | Writes |
|---|---|
| `doc/tools/make_figures.py`, `make_figures_raw.py` | `doc/_static/figures/<page>/`: fixed PNG figures (light and dark) and interactive Plotly JSON |
| `doc/tools/make_previews.py` | `doc/_static/previews/<dataset>/`: data previews and tile thumbnails |

The workflow `.github/workflows/docs.yml` publishes the site on GitHub Pages:

| Event | What the workflow does |
|---|---|
| pull request touching `doc/`, `examples/`, `src/` or `pyproject.toml` | builds the site, so that a broken page is caught before merging |
| push to `main`, published release, manual run | builds and deploys the site |

The repository's Pages source is **GitHub Actions** (*Settings > Pages >
Build and deployment > Source*). The published site:
<https://yet-another-research-organisation.github.io/spdnet-datasets/>
