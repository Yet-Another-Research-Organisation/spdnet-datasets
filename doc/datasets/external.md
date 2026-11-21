# External: EEG and fMRI

Two adapters bring datasets distributed by other libraries into the same
interface: covariances, leakage-free splits grouped by subject or site, and
`create_dataloaders`. The other library downloads and preprocesses the data;
spdnet-datasets estimates and caches the covariances.

::::{grid} 1 2 2 2
:gutter: 3
:class-container: dataset-tiles

:::{grid-item-card} MOABB adapter (EEG)
:link: moabb
:link-type: doc
:img-top: ../_static/previews/moabb/thumb.png
:img-alt: covariance of EEG channels
:class-card: sd-card-hover

Brain-computer interface datasets of MOABB: motor imagery, P300 and SSVEP
paradigms, with MOABB's evaluations.

{bdg-secondary-line}`channels × channels` {bdg-info-line}`extra moabb`
:::

:::{grid-item-card} nilearn adapter (fMRI)
:link: nilearn
:link-type: doc
:img-top: ../_static/previews/nilearn/thumb.png
:img-alt: a functional connectome
:class-card: sd-card-hover

One connectome per subject for ABIDE, ADHD-200, the development fMRI
dataset, NKI enhanced (surface) and the OpenNeuro ds000030 cohort (four
diagnoses), with site-grouped splits and cross-validation folds.

{bdg-secondary-line}`regions × regions` {bdg-info-line}`extra nilearn`
:::
::::

```{toctree}
:hidden:

moabb
nilearn
```
