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
