# Fetching and caching

Each dataset has a `fetch_<name>()` function, inspired by
[`nilearn.datasets`](https://nilearn.github.io/stable/modules/datasets.html).
It only depends on {mod}`spdnet_datasets.core`, so it can be used on its own.

```python
from spdnet_datasets.hyperspectral.gsoff import fetch_gsoff

bunch = fetch_gsoff(data_dir=None, keep="keep_covariance", estimator="ledoit_wolf")
bunch.covariances.shape, bunch.labels, bunch.groups, bunch.splits
```

## Directories

| Variable | Default | Content |
|---|---|---|
| `SPDNET_DATA_ROOT` | `/DATA/DATASET` | read-only copies of the datasets, `<DOMAIN>/<Name>/` |
| `SPDNET_CACHE_DIR` | `~/.cache/spdnet_datasets` | downloads and preprocessing cache |

`data_dir=` overrides the dataset directory; `cache_dir=` overrides the cache.

## Stages

```text
<cache_dir>/<name>/
  raw/                        downloaded sources (datasets fetched from the Internet)
  processed/<params_hash>/    preprocessed tensors used for the covariances
  covariance/<params_hash>/   unscaled covariances, labels, groups, metadata
```

Each stage ends with a `.done` file holding its parameters. A finished stage is
not recomputed (pass `force=True` to rebuild). If preprocessing fails, the
downloads are kept: fix the problem and call `fetch_*` again, nothing is
downloaded twice. Changing a parameter (window size, estimator...) creates a new
`<params_hash>` directory instead of overwriting the previous result.

## Downloads and checksums

Downloads are written to a `.part` file, resumed with HTTP range requests, and
renamed once complete. Their SHA-256 (or MD5) checksum is verified.

Datasets already present under `SPDNET_DATA_ROOT` are **not** downloaded: the
fetch checks the expected files and compares them to the dataset's
`MANIFEST_NAS.tsv` (sizes; MD5 with `verify="md5"` when the manifest has it).
The local copy is never modified.

## What is kept: `keep`

| `keep` | Kept on disk | Default for |
|---|---|---|
| `keep_processed` | covariances and preprocessed tensors | downloaded datasets |
| `keep_covariance` | covariances only | — |
| `keep_full` | downloads, preprocessed tensors and covariances | local datasets |

`return_raw=True` needs the preprocessed tensors, hence `keep_processed` or
`keep_full`.

## Sources that need an account or a manual download

Every `fetch_<name>()` states its access mode in its docstring and on the
dataset page:

| Mode | What the fetch does | What you provide |
|---|---|---|
| direct | downloads public files (Zenodo, GitHub, public Hugging Face) and checks their checksum | nothing |
| Kaggle | downloads the dataset or competition archive through the Kaggle API | Kaggle credentials, and the rules accepted on the dataset or competition page |
| Hugging Face (gated) | downloads with a token | a read token, and the conditions accepted on the dataset page |
| manual | opens the dataset page and raises `ManualDownloadRequired` with the list of files to download | the path of the downloaded file or folder: `fetch_<name>(archive="~/Downloads/<file>")` |

### Credentials

Credentials are never written in code, URLs or logs, and the
`Authorization` header is dropped when a download is redirected to another
host. They are looked up in this order ({mod}`spdnet_datasets.core.access`):

1. environment variables: `KAGGLE_USERNAME` and `KAGGLE_KEY`; `HF_TOKEN`;
2. the tool's own file: `~/.kaggle/kaggle.json` (create it from
   *kaggle.com > Settings > API > Create New Token*, then `chmod 600`);
   the Hugging Face token cache written by `huggingface-cli login`;
3. the system keyring (optional `keyring` package):

   ```python
   from spdnet_datasets.core.access import store_secret
   store_secret("huggingface", "hf_...")                      # once
   store_secret("kaggle", '{"username": "...", "key": "..."}')
   ```

4. a hidden prompt, when the fetch runs in a terminal.

Without credentials, the fetch raises `MissingCredentialsError` with these
instructions. On a shared machine, prefer the keyring or a file readable only
by you over environment variables in shell history.

### Manual downloads

```python
from spdnet_datasets.sar.s1slc_cvdl import fetch_s1slc_cvdl

fetch_s1slc_cvdl()            # opens the IEEE DataPort page, explains what to download
fetch_s1slc_cvdl(archive="~/Downloads/S1SLC_CVDL.rar")    # checks and extracts it
```

The archive is checked (size, checksum when the source publishes one) and
extracted into the cache (`raw/`, or `<name>/downloads/` for the hyperspectral
and SAR datasets, which keeps the sources outside the `keep` policy); paths that
would escape the cache folder are rejected. A `.7z` archive needs the `7z`
command or the optional `py7zr` package (`pip install 'spdnet-datasets[archives]'`),
a `.rar` archive the `unrar`, `7z` or `bsdtar` command. When the data come from
the cache, the Bunch returned by the fetch holds `root`: pass it as `data_dir`
to the dataset class. An already extracted folder can be given instead of an
archive; it is then read in place. Set `SPDNET_OPEN_BROWSER=0` to never open
a browser (servers, batch jobs).
