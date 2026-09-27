# Covariance estimation

## Estimators

`EstimateCovariance` (NumPy) and `EstimateCovarianceTorch` (PyTorch, GPU)
compute the covariance of the pixels of a hyperspectral image, the samples
being the pixels and the features the bands:

- `method="scm"`: sample covariance $\frac1n \sum_i (x_i - \bar x)(x_i - \bar x)^\top$
  (`remove_mean=False` drops the centring);
- `method="ledoit_wolf"`: Ledoit–Wolf shrinkage toward a scaled identity,
  well conditioned when there are few pixels per band;
- `exclude_zero=True` ignores the pixels that are zero in every band
  (background).

```python
import numpy as np
from spdnet_datasets.estimator import EstimateCovariance

estimator = EstimateCovariance(method="scm", remove_mean=True)
image = np.random.rand(48, 352, 204)       # H × W × bands
cov = estimator.from_image(image)           # 204 × 204
```

Robust estimators (Tyler, Student-t M-estimators), differentiable and usable
as network layers, are in `yetanotherspdnet.functions.m_estimators`.

## Precomputation scripts

`spdnet_datasets.utils` holds one script per dataset that reads the raw data
and writes the covariance files used by `mode="cov"`
(`covariance_precompute`, `ci4r_covariance_precompute`, …), and the
preprocessing of datasets delivered in other formats (`chikusei_preprocess`,
`placenta_preprocess`, …). Run them once per dataset; see the docstring of
each module for its arguments.
