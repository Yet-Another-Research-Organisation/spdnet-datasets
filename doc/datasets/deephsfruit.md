# DeepHS-Fruit

```{list-table}
:stub-columns: 1

* - Domain
  - hyperspectral imaging (HSI), laboratory, fruit ripeness
* - Task(s)
  - fine classification: `classification_mode="fruit"` (5 fruits) or `"ripeness"` (3 states)
* - Covariance
  - 224×224 (VIS camera, all bands; `band_bin` averages groups of bands), one per cube, n = the pixels of the fruit (ripeness mode: 2 001 to 107 509, median 35 078)
* - Outputs
  - cov · mean · raw (n_pixels × p, one face of a fruit)
* - Samples
  - 3 193 cubes (fruit mode) · 596 (ripeness mode); none discarded
* - Splits
  - `by_fruit` = `recommended` (grouped by physical fruit; fruit mode 1 915 / 639 / 639, ripeness mode 356 / 120 / 120) · `official` (ripeness mode: 384 / 72 / 140, leaks) · `random` (sample-level 70 / 10 / 20, ignores the groups, see {doc}`/guide/splits`)
* - Module
  - `spdnet_datasets.hyperspectral.deephsfruit` (`fetch_deephsfruit`, `DeepHSFruitDataset`, registry name `deephsfruit`)
```

```{dataset-preview} deephsfruit
```

## Description and source

DeepHS-Fruit holds ENVI (Environment for Visualizing Images) cubes, in
float32 reflectance, of single fruits on a black background: avocado, kaki,
kiwi, mango and papaya. Three cameras were used:

- VIS (visible): Specim FX10, 224 bands, 398–1 004 nm;
- NIR (near infrared): INNOSPEC RedEye, 252 bands, 920–1 727 nm;
- VIS_COR: Corning microHSI 410, 249 bands, 408–901 nm.

The v2 annotations give the ripeness (3 states), 5 fine states and the
firmness.

- Reference: L. A. Varga, J. Makowski and A. Zell, "Measuring the ripeness of
  fruit with hyperspectral imaging and deep learning", IJCNN 2021
  (International Joint Conference on Neural Networks). The article describes
  version 1; version 2 (the data read here) adds recordings and new
  annotations (update of 9 January 2024) and is published in the same
  repository, which still cites this article.
- Repository: <https://github.com/cogsys-tuebingen/deephs_fruit>. Licence: not
  stated.

## Getting the data

Manual (226 GB): the sources of the authors are listed in
<https://github.com/cogsys-tuebingen/deephs_fruit> (HTTP index and torrent of the
v2 folder; the server did not answer on 2026-11-15 and 2026-11-23). Without a
local copy under `<SPDNET_DATA_ROOT>/HYPERSPECTRAL/DeepHS-Fruit`,
`fetch_deephsfruit()` opens the repository page and raises
`ManualDownloadRequired`; `archive=` accepts the downloaded folder, or a zip of
it, with the layout of the authors.

```python
fetch_deephsfruit(archive="~/copies/DeepHS-Fruit")
```

## Raw data

Cubes are stored as
`<Fruit>/<VIS|NIR|VIS_COR>/<day>/<fruit>_<day>_<k>_<front|back>.{hdr,bin}`
(ENVI, BIP, float32 reflectance, already cut out on a zero background), with
the v2 annotations in `annotations/*_v2.json`. `train_all_v2.json` lists every
record (3 193 in VIS), `train_only_labeled_v2.json`, `val_v2.json` and
`test_v2.json` the annotated ones (384, 72 and 140 in VIS).
`merged_annotations.json` is not used: it merges v1 and v2, and gives one file
several fruits (different weights and firmness).

```{dataset-figure} deephsfruit/ripeness/raw_samples
```

## Classes

- Fruit mode (every VIS record of `train_all_v2.json`): Kiwi 1 096, Avocado 932,
  Mango 538, Kaki 371 and Papaya 256.
- Ripeness mode (the annotated VIS cubes): perfect 300, overripe 156 and unripe
  140. `ripe` is merged into `perfect`.

```{dataset-figure} deephsfruit/class_hist
```

```{dataset-figure} deephsfruit/ripeness/class_hist
```

```{dataset-figure} deephsfruit/ripeness/class_means
```

## Covariance construction

1. **Fruit mask**: Otsu threshold on the mean of bands 150–199 (VIS, about
   800–930 nm; all bands for the other cameras), over the whole frame. The
   zero background falls below it, and so does the dark rim of the fruit.
2. **Clipping**: reflectances outside [0, 1] (extremes −60 and 111 in VIS) are
   clipped.
3. **Bands**: all of them by default (p = 224 in VIS); `band_bin=k` averages
   groups of k bands (`band_bin=4`: p = 56).
4. One covariance per cube, over the pixels of the mask.

| Mode | Cubes | p | n (min / median / max) | κ median (SCM) |
|---|---|---|---|---|
| ripeness | 596 | 224 | 2 001 / 35 078 / 107 509 | 6.5e5 |
| fruit | 3 193 | 224 | 2 001 / 33 510 / 106 836 | 7.0e5 |

`camera="NIR"` (252 bands) and `camera="VIS_COR"` (249 bands) read the other
cameras. With `normalize=True` (the default), the covariances are multiplied
by a scalar fitted on the training part of `split`.

```{dataset-figure} deephsfruit/ripeness/cov_samples
```

```{dataset-figure} deephsfruit/ripeness/eigen_spectrum
```

## Splits

- **`by_fruit` (recommended)**: `StratifiedGroupKFold(5)` (seed 0), with
  groups = physical fruit `<Fruit>|<fruit>_<day>_<k>`, which brings both faces
  together. Test = fold `fold` and validation = fold `fold + 1`. Fruit mode:
  1 915 / 639 / 639; ripeness mode: 356 / 120 / 120.
- **`official`** (ripeness mode): the train / val / test files of the authors,
  384 / 72 / 140. It **leaks**: 46 of the 72 validation cubes and 46 of the 140
  test cubes have the other face of their fruit in the training part.

```{dataset-figure} deephsfruit/ripeness/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| both faces of one fruit on both sides (official split: 46 / 72 val, 46 / 140 test) | groups = physical fruit in `by_fruit`; checked by the real-data test | none for the faces |
| the same fruit recorded on several days | — | not traceable: the number `k` is reused by other fruits |

## End-to-end validation

The whole chain (raw cubes → mask → clipping → covariance → normalisation fitted on the
training part → outputs) was checked on the real data on 23 November 2026, through the
end-to-end harness of the package:

- every covariance is SPD (596 and 3 193 checked, symmetric to machine precision);
- the geometric mean of the eigenvalues of the training covariances is 1 after scaling
  (mean log-eigenvalue below 1e-13);
- the outputs are coherent: on 50 training samples, the covariance of `raw` equals
  `cov` (relative error ≤ 9e-8) and the mean of `raw` equals `mean` (≤ 9e-9), which
  checks the factors `s` on `cov` and `√s` on `raw` and `mean`;
- two builds give bit-identical covariances and identical splits; `by_fruit` has no
  physical fruit on two sides, while the leak of `official` is detected;
- the data loaders serve every sample of each part.

Balanced accuracy on the test part (chance 0.33 for ripeness, 0.20 for fruit), tangent
space + logistic regression (TS-LR, C = 0.1) and an MLP on the log-Euclidean features (mean
± standard deviation over 3 seeds):

| Configuration | Split | TS-LR | MLP |
|---|---|---|---|
| ripeness, p = 224, SCM | `by_fruit` | 0.48 | 0.52 ± 0.03 |
| ripeness, p = 224, Ledoit-Wolf | `by_fruit` | 0.48 | 0.54 ± 0.02 |
| ripeness, `band_bin=4` (p = 56) | `by_fruit` | 0.56 | 0.57 ± 0.01 |
| ripeness, p = 224 | `official` | 0.39 | 0.52 ± 0.02 |
| ripeness, p = 224 | random (ignores the fruits) | — | 0.60 |
| fruit, p = 224 | `by_fruit` | 0.99 | 1.00 ± 0.00 |

Ripeness is the hard task: with the faces of a fruit kept together, about 0.5 to 0.57.
Splitting at random, ignoring the physical fruit, inflates the MLP score by about 8
points. Recognising the fruit is trivial (0.99). The standardisation of the authors
(fixed per-band mean and standard deviation of the VIS camera, `normalize.py` of their
repository), applied to the covariances as `D⁻¹ C D⁻¹`, raises the median condition number
from 6.5e5 to 1.1e6 and leaves TS-LR unchanged (0.47); it is not applied by the loader.

## Supported tasks

- `classification_mode="fruit"` (5 classes) or `"ripeness"` (3 classes), both
  with `task="fine"`. The firmness and the storage days are in `meta` but are
  not served as regression targets.
- `drop_classes`, `merge_classes` and `balance=...` work as for the other
  datasets.

## Usage

```python
from spdnet_datasets.hyperspectral import DeepHSFruitDataset

ripe = DeepHSFruitDataset(classification_mode="ripeness")             # 596 × 224 × 224
ripe = DeepHSFruitDataset(classification_mode="ripeness", band_bin=4) # p = 56
fruit = DeepHSFruitDataset(camera="NIR")                              # 252 bands
```

## Migration notes

- The covariances are now computed from the raw cubes. The former loader read
  the historical `cov_VIS/*.pt` files, whose construction was undocumented,
  and its ripeness labels were matched to the wrong fruits (by record id
  instead of header file): every result obtained with it is invalid.
- Ripeness mode now holds the 596 annotated VIS cubes (384 before) and offers
  the official split.
- `return_mean=True` and `return_raw=True` are available; `modality=` is kept as
  an alias of `camera=`.

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/deephsfruit.md
```
````
