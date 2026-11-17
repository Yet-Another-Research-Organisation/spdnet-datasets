# DeepHS-Fruit

```{warning}
⚠️ **Outside the maintained selection.** DeepHS-Fruit is not one of the
datasets maintained by this package. The loader is only adapted to the
common interface: it reads the **historical precomputed covariances**
(`cov_VIS/*.pt`) as they are. Their construction (pixel mask, estimator,
scaling) is undocumented and has not been re-verified. Raw data and channel
means are not served, and the n ≥ 2p rule cannot be checked. A `UserWarning`
is raised at construction.
```

```{danger}
**Ripeness labels were attributed to the wrong fruit by the former loader.**
The `cov_VIS` files are numbered like the records of
`merged_annotations.json`, but the v2 annotation files number their records
differently: id 1 is fruit 01 in `merged`, and fruit 20 in
`train_only_labeled_v2`. The former `ripeness` mode matched the two by id.
**Every past result on ripeness obtained with the former loader is
invalid.** The labels are now matched through the header file of each
record.
```

```{list-table}
:stub-columns: 1

* - Domain
  - hyperspectral imaging (HSI), laboratory, fruit ripeness
* - Task(s)
  - fine classification: `classification_mode="fruit"` (5 fruits) or `"ripeness"` (3 states)
* - Covariance
  - 224×224, precomputed (VIS, visible-range camera); n unknown
* - Outputs
  - cov only
* - Samples
  - 3 193 covariances (fruit mode) · 384 (ripeness mode)
* - Splits
  - `by_fruit` = `recommended` (grouped by physical fruit; fruit mode 1 915 / 639 / 639, ripeness mode 230 / 77 / 77)
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
  (International Joint Conference on Neural Networks). v2 annotations:
  University of Tübingen, update of 9 January 2024. Neither was re-verified.
- Repository: <https://github.com/cogsys-tuebingen/deephs_fruit> (URL not
  verified). Licence: see the authors' repository (not verified).

## Getting the data

Manual, and outside the selection: the `cov_VIS/*.pt` covariances read here
are a local precomputation that no download provides, so the sources of the
authors (226 GB, links in <https://github.com/cogsys-tuebingen/deephs_fruit>;
servers unreachable on 2026-11-15) cannot feed this loader. Without a local copy,
`fetch_deephsfruit()` opens the repository page and raises
`ManualDownloadRequired`; `archive=` accepts a copy with the local layout
(`cov_VIS/`, `annotations/`).

```python
fetch_deephsfruit(archive="~/copies/DeepHS-Fruit")
```

## Raw data

Cubes are stored as `<Fruit>/<VIS|NIR|VIS_COR>/<day>/<fruit>_<day>_<k>_<front|back>.{hdr,bin}`,
with annotations in `annotations/*.json`. The loader reads neither the
cubes nor the masks: it reads `cov_VIS/<fruit>_VIS_<day>_<record>_<side>.pt`,
one 224 × 224 covariance per VIS cube. There is therefore no raw-data figure.

## Classes

- Fruit mode (every VIS covariance): Kiwi 1 096, Avocado 932, Mango 538, Kaki
  371 and Papaya 256.
- Ripeness mode (the VIS cubes annotated in `train_only_labeled_v2`): perfect
  216, overripe 86 and unripe 82. `ripe` is merged into `perfect`. By fruit,
  these are Kiwi 138, Avocado 92, Kaki 56, Mango 56 and Papaya 42.

```{dataset-figure} deephsfruit/class_hist
```

```{dataset-figure} deephsfruit/ripeness/class_hist
```

## Covariance construction

The covariances are the historical `cov_VIS/*.pt` files, read **read-only**
and converted to float64. The bands (224), the pixels kept and the estimator
are not documented. The atlas notes that the cubes are already cut out
(background at 0). It also notes values outside [0, 1] (VIS extremes −60 and
111), which an unclipped estimator would include. The `estimator=` and
`keep=` options have no effect.

With `normalize=True` (the default), the covariances are multiplied by a
scalar fitted on the training part of `split`, so that the geometric mean of
their eigenvalues is 1.

The figures below show two covariances of each mode and the eigenvalue
spectrum.

```{dataset-figure} deephsfruit/cov_samples
```

```{dataset-figure} deephsfruit/eigen_spectrum
```

```{dataset-figure} deephsfruit/ripeness/cov_samples
```

```{dataset-figure} deephsfruit/ripeness/eigen_spectrum
```

## Splits

- **`by_fruit` (recommended)**: `StratifiedGroupKFold(5)` (seed 0), with
  groups = physical fruit. A physical fruit is `<Fruit>|<fruit>_<day>_<k>`,
  which brings both faces together. Test = fold `fold` and validation = fold
  `fold + 1`. Fruit mode has 1 644 groups and gives 1 915 / 639 / 639.
  Ripeness mode has 238 groups and gives 230 / 77 / 77.
- **The official v2 split is not offered.** The VIS cubes of `val_v2` (72) and
  `test_v2` (140) are absent from `merged_annotations.json`, and have no
  `cov_VIS` file. It would also leak: 46 of the 140 VIS test cubes have the
  other face of their fruit in train.

```{dataset-figure} deephsfruit/split_schema
```

```{dataset-figure} deephsfruit/ripeness/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| both faces of one fruit on both sides (official v2 split: 46 / 140 VIS test cubes) | groups = physical fruit; checked by the real-data test | none for the faces |
| the same fruit under another number on another day | — | not verifiable without metadata |
| ripeness labels matched to the wrong fruit (former loader) | matched by header file | none; past ripeness results invalid |
| undocumented covariance construction | warning | unknown |

## Supported tasks

- `classification_mode="fruit"` (5 classes) or `"ripeness"` (3 classes, 384
  covariances), both with `task="fine"`. There is no coarse map and no
  regression (the firmness is not served).
- `drop_classes`, `merge_classes` and `balance=...` work as for the other
  datasets. Only the VIS modality exists (`modality="VIS"`).

## Usage

```python
import warnings
from spdnet_datasets.hyperspectral.deephsfruit import DeepHSFruitDataset

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)    # "historical precomputed covariances"
    ripe = DeepHSFruitDataset(classification_mode="ripeness", subset="train")
```

## Migration notes

- Ripeness labels are now matched by header file (see the warning above).
  Results from the former loader are invalid.
- `return_mean=True` and `return_raw=True` raise an error.
- `split="train"` raises an error: use `subset="train"`. `subset=None` is the
  default.
