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
make -C doc html          # strict build; runs the tutorials of the gallery
```

## Where the data live

The datasets themselves are not distributed with the package. They are read
from `$SPDNET_DATA_ROOT/<DOMAIN>/<Name>` (default `/DATA/DATASET`) or fetched
into `$SPDNET_CACHE_DIR` (default `~/.cache/spdnet_datasets`); see
{doc}`/guide/fetch`.
