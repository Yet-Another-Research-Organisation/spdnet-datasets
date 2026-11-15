# Principles

**spdnet-datasets** turns heterogeneous measurements (hyperspectral images, radar
spectrograms, synthetic aperture radar (SAR) time series, skeletons, audio...)
into **symmetric positive definite (SPD) matrices**, mostly covariance matrices,
for SPDNet-style networks. It also serves the data before the covariance step,
so the same split can feed a CNN, a Transformer or a Vision Transformer (ViT).

## One sample, one covariance

A sample is a set of `n` observations of `p` channels, e.g. the `N×M` pixels of
a window with `p` spectral bands, or the `T` time steps of a spectrogram with
`p` frequency bins. Its covariance is a `p×p` SPD matrix.

```text
 raw tensor (N, M, p)  ──reshape──▶  X (n = N·M, p)  ──estimator──▶  C (p, p) SPD
```

Four rules hold for every dataset:

1. **Enough observations.** A covariance is only estimated when `n ≥ 2p`.
   Otherwise {class}`~spdnet_datasets.core.InsufficientSamplesError` is raised:
   no ill-conditioned matrix is produced silently.
2. **SPD check.** Every estimate is checked for symmetry and strictly positive
   eigenvalues ({func}`~spdnet_datasets.core.check_spd`).
3. **Normalisation fitted on the training split.** A single scalar brings the
   geometric mean of the training eigenvalues to 1 ({doc}`outputs`).
4. **Leakage-free splits.** Samples carry a group (subject, patch, spatial
   block, parcel, site...) and the provided splits never separate a group
   ({doc}`splits`).

## Estimators

`estimator=` selects the covariance estimator ({mod}`spdnet_datasets.core.estimators`):

| Name | Estimator | Notes |
|---|---|---|
| `scm` | sample covariance matrix (SCM), normalised by `n` | default |
| `ledoit_wolf` | Ledoit–Wolf shrinkage ({class}`sklearn.covariance.LedoitWolf`) | well conditioned when `n` is close to `2p` |
| `tyler` | Tyler's M-estimator, trace normalised to `p` | robust to heavy tails and outliers; defined up to scale |
| `student` | Student-t M-estimator with `nu` degrees of freedom | robust, keeps its scale |

The two M-estimators are fixed points of

$$
\Sigma = \frac1n \sum_{i=1}^n u\!\left(x_i^\top \Sigma^{-1} x_i\right) x_i x_i^\top,
\qquad u_{\text{Tyler}}(q) = \frac{p}{q},\quad
u_{\text{Student}}(q) = \frac{p+\nu}{\nu+q},
$$

computed with `yetanotherspdnet.functions.m_estimators`
(`pip install "spdnet-datasets[robust]"`).

## Where the data live

Datasets are read from `$SPDNET_DATA_ROOT/<DOMAIN>/<Name>` (default
`/DATA/DATASET`), which is never written to. Downloads, preprocessed tensors and
covariances go to `$SPDNET_CACHE_DIR` (default `~/.cache/spdnet_datasets`).
See {doc}`fetch`.

## Package layout

```text
spdnet_datasets/
  core/                 config, estimators, normalize, outputs, tasks, splits, fetch
  audio/ skeleton/ radar_microdoppler/ sits/ hyperspectral/ sar/
                        one module per dataset: fetch_<name>() + <Name>Dataset
  real/                 deprecated aliases of the former loaders
  synthetic/            synthetic SPD datasets
```
