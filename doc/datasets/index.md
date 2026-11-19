# Datasets

Every dataset has its own page: source and licence, how to get the data, raw
data, classes, covariance construction, splits, leakage risks, supported tasks,
usage and how to cite it. Pick a domain, or start from the
{doc}`catalogue` of the registered loaders.

## By domain

::::{grid} 1 2 3 4
:gutter: 3
:class-container: dataset-tiles

:::{grid-item-card} Audio
:link: audio
:link-type: doc
:img-top: ../_static/previews/esc50/thumb.png
:img-alt: MFCC frames of an ESC-50 clip
:class-card: sd-card-hover

Environmental sounds, music genres, bird songs.

{bdg-secondary-line}`4 datasets`
:::

:::{grid-item-card} Skeleton
:link: skeleton
:link-type: doc
:img-top: ../_static/previews/ntu120/thumb.png
:img-alt: an NTU RGB+D 120 skeleton
:class-card: sd-card-hover

Motion capture and 3D action recognition.

{bdg-secondary-line}`2 datasets`
:::

:::{grid-item-card} Micro-Doppler radar
:link: radar
:link-type: doc
:img-top: ../_static/previews/ci4r/thumb.png
:img-alt: a CI4R micro-Doppler spectrogram
:class-card: sd-card-hover

Human activities, hand gestures, gait.

{bdg-secondary-line}`3 datasets`
:::

:::{grid-item-card} Satellite image time series
:link: sits
:link-type: doc
:img-top: ../_static/previews/pastis_r/thumb.png
:img-alt: a PASTIS-R patch in false colours
:class-card: sd-card-hover

Crop types from Sentinel-1 and Sentinel-2.

{bdg-secondary-line}`1 dataset`
:::

:::{grid-item-card} Hyperspectral
:link: hyperspectral
:link-type: doc
:img-top: ../_static/previews/chikusei/thumb.png
:img-alt: the Chikusei scene in false colours
:class-card: sd-card-hover

Airborne, UAV and laboratory imagers.

{bdg-secondary-line}`9 datasets`
:::

:::{grid-item-card} SAR
:link: sar
:link-type: doc
:img-top: ../_static/previews/mstar/thumb.png
:img-alt: MSTAR target chips
:class-card: sd-card-hover

PolSAR, SAR time series, target chips.

{bdg-secondary-line}`7 datasets`
:::

:::{grid-item-card} External: EEG and fMRI
:link: external
:link-type: doc
:img-top: ../_static/previews/nilearn/thumb.png
:img-alt: a functional connectome
:class-card: sd-card-hover

MOABB and nilearn adapters.

{bdg-secondary-line}`2 adapters`
:::
::::

## Working with datasets

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`table` Catalogue
:link: catalogue
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

Every registered loader with its specific options, generated from the
registry; hyperspectral modes and expected layouts.
:::

:::{grid-item-card} {octicon}`beaker` Covariance estimation
:link: covariance
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

SCM, Ledoit–Wolf, Tyler and Student-t estimators, and the precomputation
scripts.
:::

:::{grid-item-card} {octicon}`cpu` Synthetic data
:link: synthetic
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

Generators of SPD datasets whose class differences are known.
:::

:::{grid-item-card} {octicon}`plus-circle` Adding a dataset
:link: adding
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

Subclass `BaseDataset`, register it, write its fetch, tests and page.
:::
::::

```{toctree}
:hidden:
:caption: By domain

audio
skeleton
radar
sits
hyperspectral
sar
external
```

```{toctree}
:hidden:
:caption: Working with datasets

catalogue
covariance
synthetic
adding
```
