---
html_theme.sidebar_secondary.remove: true
---

# spdnet-datasets

```{raw} html
<div class="hero">
  <div>
    <p class="hero-title">spdnet-datasets</p>
    <p class="hero-sub">SPD matrices from real measurements, ready for SPDNet</p>
    <p class="tagline">Loaders that turn audio, radar, hyperspectral, SAR, satellite time
    series and skeleton data into batches of symmetric positive definite covariance
    matrices, with leakage-free splits, a reproducible cache and synthetic generators.</p>
  </div>
  <div class="hero-logo"><img src="_static/logo.png" alt="spdnet-datasets logo"></div>
</div>
```

{bdg-primary-line}`Python ≥ 3.11` {bdg-primary-line}`PyTorch` {bdg-primary-line}`MIT licence`
{bdg-primary-line}`25 datasets + MOABB + nilearn`

````{container} hero-buttons
```{button-ref} getting_started/installation
:ref-type: doc
:color: primary
:class: sd-px-4

{octicon}`rocket` Get started
```
```{button-ref} datasets/index
:ref-type: doc
:color: secondary
:outline:
:class: sd-px-4

{octicon}`database` Browse the datasets
```
```{button-ref} auto_examples/index
:ref-type: doc
:color: secondary
:outline:
:class: sd-px-4

{octicon}`image` See the gallery
```
````

```python
from spdnet_datasets import PastisRDataset, create_dataloaders

ds = PastisRDataset(scenario="C", estimator="ledoit_wolf", return_mean=True)
train, val, test = create_dataloaders(ds, split="official", batch_size=64)
(cov, mean), y = next(iter(train))      # cov: (64, 16, 16) SPD, mean: (64, 16)
```

## Explore the documentation

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting started
:link: getting_started/index
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-start

Install the package and its extras, set the data and cache folders, add your
Kaggle or Hugging Face credentials, and load a first dataset.
:::

:::{grid-item-card} {octicon}`book` User guide
:link: guide/index
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-guide

What a sample is, outputs and tasks, the fetch cache, leakage-free splits and
the configuration-dictionary entry point.
:::

:::{grid-item-card} {octicon}`database` Datasets
:link: datasets/index
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-datasets

One page per dataset, by domain, with previews, figures, splits, leakage risks
and how to cite it. Plus covariance estimation and synthetic data.
:::

:::{grid-item-card} {octicon}`image` Examples
:link: auto_examples/index
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-examples

Tutorials that run without any data, and one example per dataset family, as
`.py` scripts or Jupyter notebooks.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: reference/index
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-api

Dataset classes, the `core` contract (estimators, outputs, tasks, splits,
fetch), the manager and the synthetic generators.
:::

:::{grid-item-card} {octicon}`mortar-board` References
:link: references
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-refs

The publications behind every dataset, with their DOI, to cite the data you
train on.
:::
::::

## Datasets by domain

::::{grid} 2 3 4 4
:gutter: 2
:class-container: dataset-tiles

:::{grid-item-card} Audio
:link: datasets/audio
:link-type: doc
:img-top: _static/previews/esc50/thumb.png
:img-alt: MFCC frames of an ESC-50 clip
:class-card: sd-card-hover

ESC-50, GTZAN, BirdSet XCM, UrbanSound8K
:::

:::{grid-item-card} Skeleton
:link: datasets/skeleton
:link-type: doc
:img-top: _static/previews/ntu120/thumb.png
:img-alt: an NTU RGB+D 120 skeleton
:class-card: sd-card-hover

HDM05, NTU RGB+D 120
:::

:::{grid-item-card} Micro-Doppler radar
:link: datasets/radar
:link-type: doc
:img-top: _static/previews/ci4r/thumb.png
:img-alt: a CI4R micro-Doppler spectrogram
:class-card: sd-card-hover

CI4R, DopNet, MVDoppler
:::

:::{grid-item-card} Satellite image time series
:link: datasets/sits
:link-type: doc
:img-top: _static/previews/pastis_r/thumb.png
:img-alt: a PASTIS-R patch in false colours
:class-card: sd-card-hover

PASTIS-R
:::

:::{grid-item-card} Hyperspectral
:link: datasets/hyperspectral
:link-type: doc
:img-top: _static/previews/chikusei/thumb.png
:img-alt: the Chikusei scene in false colours
:class-card: sd-card-hover

Chikusei, GSOFF, UAV-HSI-Crop, HyperLeaf, DeepHS-Fruit and 4 more
:::

:::{grid-item-card} SAR
:link: datasets/sar
:link-type: doc
:img-top: _static/previews/mstar/thumb.png
:img-alt: MSTAR target chips
:class-card: sd-card-hover

MSTAR, FUSAR-Ship, ATRNet-STAR, Pol-InSAR-Island and 3 more
:::

:::{grid-item-card} External: EEG and fMRI
:link: datasets/external
:link-type: doc
:img-top: _static/previews/nilearn/thumb.png
:img-alt: a covariance matrix of EEG channels
:class-card: sd-card-hover

MOABB (EEG) and nilearn (fMRI connectomes) adapters
:::

:::{grid-item-card} Synthetic data
:link: datasets/synthetic
:link-type: doc
:img-top: _static/previews/synthetic/thumb.png
:img-alt: synthetic SPD matrices
:class-card: sd-card-hover

Controlled SPD datasets for experiments
:::
::::

## Why spdnet-datasets

::::{grid} 1 2 3 3
:gutter: 3
:class-container: features

:::{grid-item-card} {octicon}`shield-check` Leakage-free splits
Every sample carries a group (subject, parcel, recording, spatial block) and the
provided splits never separate a group. See {doc}`guide/splits`.
:::

:::{grid-item-card} {octicon}`check-circle` Checked SPD matrices
Covariances are only estimated with `n ≥ 2p` observations and checked for
positive eigenvalues: nothing is regularised silently. See {doc}`guide/principles`.
:::

:::{grid-item-card} {octicon}`stack` Raw data too
The same split serves the covariance, the channel means or the raw tensor, so a
CNN or a Transformer can be compared on equal terms. See {doc}`guide/outputs`.
:::

:::{grid-item-card} {octicon}`download` Fetch and cache
`fetch_<name>()` downloads, checks and caches each dataset in stages; nothing
is downloaded or recomputed twice. See {doc}`guide/fetch`.
:::

:::{grid-item-card} {octicon}`beaker` Estimators
SCM, Ledoit–Wolf, Tyler and Student-t estimators behind one argument. See
{doc}`datasets/covariance`.
:::

:::{grid-item-card} {octicon}`plug` Adapters
MOABB (EEG) and nilearn (fMRI) datasets through the same interface. See
{doc}`datasets/moabb` and {doc}`datasets/nilearn`.
:::
::::

## Where it fits

```text
spdnet-datasets ──▶ yetanotherspdnet (models) ──▶ spdnet-training (training loop, Hydra configs)
   loaders, splits,       SPDnet, RResNet,             spdnet-train / spdnet-optimize
   synthetic data         batch normalization
```

```{admonition} Citing the data
:class: tip

When you publish results obtained on a dataset, cite its original publication:
each dataset page ends with a *How to cite* block, and {doc}`references` lists
them all.
```

```{toctree}
:hidden:

getting_started/index
guide/index
datasets/index
auto_examples/index
reference/index
references
```
