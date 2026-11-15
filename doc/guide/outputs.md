# Outputs

## The default: covariances only

`dataset[i]` returns `(x, y)`. By default `x` is the `p×p` covariance (float64).
A training loop written for SPD matrices therefore needs no change.

## Optional outputs

Two optional outputs are selected in the dataset constructor:

| Flag | Adds | Shape |
|---|---|---|
| `return_mean=True` | per-channel mean of the observations | `(p,)` |
| `return_raw=True` | the preprocessed data the covariance was computed from | `(N, M, p)` or `(T, p)` |

When at least one is requested, `x` becomes a tuple in this fixed order:

| `return_mean` | `return_raw` | `x` |
|---|---|---|
| False | False | `cov` |
| True | False | `(cov, mean)` |
| False | True | `(cov, raw)` |
| True | True | `(cov, mean, raw)` |

```python
ds = SomeDataset(return_mean=True)
(cov, mean), y = ds[0]
for (cov, mean), y in loader:      # the default collate keeps the tuple
    ...
```

`raw` is optional: datasets that do not provide it raise a `ValueError` listing
their available outputs.

## Raw-or-mean datasets (SAR)

Most SAR datasets (AIR-PolSAR-Seg, ATRNet-STAR, FUSAR-Ship, MSTAR, S1SLC_CVDL)
provide **only** the raw chips or the channel means. Choose exactly one of
`return_raw=True` or `return_mean=True`; `x` is then that array alone.

## Labels

`y` is an integer class index for classification, a float or a vector for
regression ({doc}`tasks`), and `-1` on unlabelled splits.

## Normalisation

With `normalize=True` (default), covariances are multiplied by one scalar

$$ s = \exp\!\Big(-\operatorname{mean}_{C \in \text{train}} \operatorname{mean}_k \log \lambda_k(C)\Big), $$

so that the geometric mean of the training eigenvalues is 1. The scalar is
fitted on the training split only and applied to every split. Cached
covariances are stored unscaled. The mean and the raw data are multiplied by
$\sqrt{s}$, so that they stay consistent with the covariance (the covariance
of $\sqrt{s}\,x$ is $s\,C$); the SAR datasets that only return raw data or
means (no covariance) are not scaled.
