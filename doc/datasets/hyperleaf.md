# HyperLeaf2024

```{list-table}
:stub-columns: 1

* - Domain
  - hyperspectral imaging (HSI), laboratory, wheat leaves
* - Task(s)
  - fine classification (`target="cultivar"`: 4 classes · `"fertilizer"`: 3 · `"combined"`: 12) · regression (GrainWeight, Gsw, PhiPS2)
* - Covariance
  - 204×204, n = 1 070 to 7 044 leaf pixels (median 2 747; n / 2p ≥ 2.6)
* - Outputs
  - cov · mean · raw (n leaf pixels × 204, n varies); `mode="image"`: band images for CNNs
* - Samples
  - 1 590 labelled leaves (none discarded)
* - Splits
  - `by_plot` = `recommended` (24 reconstructed plots, fold 0: 960 / 310 / 320) · `random` (sample-level 70 / 10 / 20: leaks plots, see {doc}`/guide/splits`)
* - Module
  - `spdnet_datasets.hyperspectral.hyperleaf` (`fetch_hyperleaf`, `HyperLeafDataset`, registry name `hyperleaf`)
```

```{dataset-preview} hyperleaf
```

## Description and source

HyperLeaf2024 comes from a Kaggle competition. It holds hyperspectral images
of wheat leaves taken by a laboratory camera, with 204 VNIR (visible and
near-infrared) bands. Each image is labelled with its cultivar (4), its
fertilizer level (0, 0.5 or 1) and three continuous targets: grain weight
(GrainWeight), stomatal conductance (Gsw) and the photosystem II efficiency
(PhiPS2). Acquisition dates are not provided.

- Reference: W. M. Laprade *et al.*, "HyperLeaf2024 – a hyperspectral imaging
  dataset for classification and regression of wheat leaves", CVPR Workshops
  (FGVC11), 2024, pp. 1234–1243 (CVPR: Conference on Computer Vision and
  Pattern Recognition; FGVC11: 11th Workshop on Fine-Grained Visual
  Categorization). The reference was not re-verified.
- Data: <https://www.kaggle.com/competitions/hyperleaf2024> (Kaggle account
  required; URL not verified). Licence: competition terms (check them before
  any redistribution).
- Only the 1 590 training images are labelled. The 820 test images, used
  for the competition submission, are not loaded: every split is cut from
  the training set.

## Getting the data

Kaggle competition [`hyperleaf2024`](https://www.kaggle.com/competitions/hyperleaf2024)
(there is no Kaggle dataset of that name). It needs Kaggle credentials (see
[Fetching](../guide/fetch.md)) and the competition rules accepted with the same
account. Without a local copy, `fetch_hyperleaf()` downloads the competition
archive into `<cache>/HyperLeaf2024/downloads/` (no checksum published) and reads
the folder holding `train.csv` and `images/`.

```python
fetch_hyperleaf(archive="~/Downloads/hyperleaf2024.zip")
```

## Raw data

`images/<ImageId>.tiff` holds images of 48 × 352 pixels × 204 bands, stored
as (bands, rows, columns), uint16 raw DN (digital numbers). There is no
white or dark reference, and saturated values (65 535) exist. The background
is 0, which is 84 % of the pixels in the median image. `train.csv` holds the
targets, with the cultivars one-hot encoded. The raw output is the set of
leaf pixels, an (n, p) float32 array: exactly the observations of the
covariance (background excluded), so its covariance is `cov` and its mean is
`mean`. Since n varies from leaf to leaf (1 070 to 7 044), use `batch_size=1`
or a `collate_fn`; `mode="image"` gives the whole image. The figure shows the
band-mean image.

```{dataset-figure} hyperleaf/raw_samples
```

## Classes

- Cultivar (default): Heerup 410, Kvium 350, Rembrandt 430 and Sheriff 400
  leaves.
- Fertilizer: 0.0 → 590, 0.5 → 160 and 1.0 → 840.
- Combined (fertilizer × cultivar): 12 classes, from 20 (0.5_Rembrandt) to
  290 (1.0_Heerup).
- Regression: GrainWeight from 3 100 to 9 742 (24 distinct values only),
  Gsw from −0.0096 to 0.157 (a few negative values), and PhiPS2 from 0.136
  to 0.741.

The figures show the cultivar counts and the mean spectrum per cultivar.

```{dataset-figure} hyperleaf/class_hist
```

```{dataset-figure} hyperleaf/class_means
```

## Covariance construction

1. **Observations**: the non-zero (leaf) pixels of the image, from 1 070 to
   7 044. Images with fewer than 2p = 408 leaf pixels would be discarded and
   counted: none are.
2. **Bands**: all 204 (`band_step=1`). `band_step=2` gives p = 102.
3. **Estimator**: SCM (sample covariance matrix) by default;
   `"ledoit_wolf"`, `"tyler"` or `"student"` are also available.
4. **Normalisation**: covariances are scaled on the training part of `split`
   (geometric mean of the eigenvalues = 1). Means and raw pixels are
   multiplied by √s.

`mode="image"` keeps the former image mode. `x` is then the image restricted
to `bands` (by default bands 81, 51 and 21, approximately red, green and blue),
standardised per channel, as float32 of shape (C, 48, 352). It is meant for
CNN (convolutional neural network) backbones, and no covariance is
computed.

```{dataset-figure} hyperleaf/cov_samples
```

```{dataset-figure} hyperleaf/eigen_spectrum
```

## Splits

The official split (labelled train, unlabelled test) cannot be evaluated
locally. The paper does not describe plots, but GrainWeight takes only 24
distinct values, because all the leaves of a plot share it. The loader
reconstructs the plot as `GrainWeight | cultivar | Fertilizer` (24 groups of
20 to 110 leaves). It then splits with a `StratifiedGroupKFold(5)` on the
cultivar (seed 0): test = fold `fold`, validation = fold `fold + 1`. With
`fold=0`: 960 / 310 / 320 leaves.

```{dataset-figure} hyperleaf/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| leaves of the same plot in train and test | groups = reconstructed plot; checked by the real-data test | the plot is a reconstruction, not documented by the authors |
| random split of leaves | not offered | the atlas measures a cultivar TS-LR (tangent-space logistic regression) accuracy of 0.915 with a random split, against 0.669 grouped; for GrainWeight, R² is 0.79 against 0.18 |
| regression on GrainWeight | grouped split (one value per plot) | with only 24 values, the regression is close to a classification of plots |

## Supported tasks

- `target="cultivar"` (default), `"fertilizer"` or `"combined"` with
  `task="fine"`.
- `task="regression"`: `y` is the vector of `regression_targets` (default
  `("GrainWeight", "Gsw", "PhiPS2")`). Pass `regression_targets=("Gsw",)` for
  a single target. `drop_classes` and `merge_classes` do not apply.
- `balance="undersample"`, `"oversample"` or `"weights"` acts on the training
  part only (classification).

## Usage

```python
from spdnet_datasets.hyperspectral.hyperleaf import HyperLeafDataset

ds = HyperLeafDataset(target="cultivar")                        # ds.splits["by_plot"]
fert = HyperLeafDataset(target="fertilizer", subset="train", estimator="ledoit_wolf")
reg = HyperLeafDataset(task="regression", regression_targets=("GrainWeight",))
img = HyperLeafDataset(mode="image", bands=[81, 51, 21])
```

## Migration notes

- **`target=` replaces the former `task=`** for the label choice. `task` now
  means fine / coarse / regression, as for every dataset, and
  `task="cultivar"` raises an error.
- `split="train"` raises an error: use `subset="train"`. `subset=None` is the
  default.
- Use `estimator=`. `cov_method=...` is silently absorbed by `**kwargs` and
  gives the SCM.

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/hyperleaf.md
```
````
