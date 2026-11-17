# Covariance estimation

## Estimators

The loaders estimate their covariances with
`spdnet_datasets.core.estimators.estimate_covariance(X, estimator=...)`,
where `X` holds `n` observations (rows) of `p` channels (columns). Every
dataset class takes the same `estimator=` argument:

- `"scm"` (default): sample covariance matrix (SCM),
  $\frac1n \sum_i (x_i - \bar x)(x_i - \bar x)^\top$;
- `"ledoit_wolf"`: Ledoit–Wolf shrinkage toward a scaled identity, better
  conditioned when n is close to p;
- `"tyler"` and `"student"`: Tyler and Student-t M-estimators, taken from
  `yetanotherspdnet.functions.m_estimators` (install the `robust` extra).

Before estimating, the function checks that n ≥ 2p
(`InsufficientSamplesError`). Afterwards, it checks that the result is
symmetric positive definite (`NotSPDError`). The loaders discard and count
the samples that fail either check; see the page of each dataset.

```python
import numpy as np
from spdnet_datasets.core.estimators import estimate_covariance

X = np.random.rand(225, 63)                   # 15 × 15 pixels × 63 bands
C = estimate_covariance(X, estimator="ledoit_wolf")   # 63 × 63
```

Pass `estimator=` to the dataset classes. A former argument such as
`cov_method="ledoit_wolf"` is silently absorbed by `**kwargs`, and the SCM is
used instead.

The former classes `EstimateCovariance` (NumPy) and `EstimateCovarianceTorch`
(PyTorch, on a GPU: graphics processing unit) of `spdnet_datasets.estimator` remain available for images
(`from_image`, `method="scm"` or `"ledoit_wolf"`, `exclude_zero=True` to
ignore the background).

## Precomputation

Covariances are no longer produced by standalone scripts. Each dataset's
`fetch_<name>()` function reads (or downloads) the data, preprocesses it and
estimates the covariances. It stores the results in the `processed/` and
`covariance/` stages of the cache, and the dataset class calls it on first
use. See {doc}`/guide/fetch` for the cache layout and the `keep` option, and the
{doc}`index` pages for the construction of each dataset. For
example, `fetch_chikusei` replaces the former `chikusei_preprocess` script,
and `fetch_placenta` replaces `placenta_preprocess`.

The scripts left in `spdnet_datasets.utils` (`covariance_precompute`,
`ntu120_covariance_precompute`, …) only serve the deprecated loaders of
`spdnet_datasets.real`.
