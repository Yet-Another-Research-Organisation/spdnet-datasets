# Hyperspectral placenta

```{list-table}
:stub-columns: 1

* - Domain
  - hyperspectral imaging (HSI), surgical microscope, tissue classification
* - Task(s)
  - fine classification (5 classes: Artery, Stroma, Suture, Umbilical, Vein)
* - Covariance
  - 37×37 (515–697 nm), n = 576 pixels of a 24×24 window (n / 2p = 7.8)
* - Outputs
  - cov · mean · raw (24 × 24 × 37)
* - Samples
  - 76 862 windows (77 203 minus 341 singular sample covariance matrices, SCM, 0.44 %)
* - Splits
  - `by_image` = `recommended` (81 images, fold 0: 45 939 / 15 851 / 15 072) · `dye_transfer` (red_blue → ICG: 46 303 / 0 / 30 559) · `random` (sample-level 70 / 10 / 20, ignores the groups, see {doc}`/guide/splits`)
* - Module
  - `spdnet_datasets.hyperspectral.placenta` (`fetch_placenta`, `PlacentaDataset`, registry name `placenta`)
```

```{dataset-preview} placenta
```

## Description and source

The dataset holds 101 hyperspectral images of 1 024 × 1 024 pixels of four
human placentas, used for microsurgical training. The images were taken with
a Senop HSC-2 camera mounted on a Zeiss OPMI Pentero 900 surgical microscope
under xenon illumination. They are flat-field corrected (reflectance 0–1) and
annotated manually. Acquisition dates are not given.

- Reference: S. Puustinen, J. Hyttinen, A.-P. Elomaa and H. Vrzáková,
  "Hyperspectral placenta dataset: hyperspectral image acquisition,
  annotations, and processing of biological tissues in microsurgical
  training", *Data in Brief* 50, 109526, 2023,
  doi:[10.1016/j.dib.2023.109526](https://doi.org/10.1016/j.dib.2023.109526).
- Licence: CC BY 4.0, according to the article.
- Dyes: images P007–P053 are `red_blue`, P054–P087 are ICG (indocyanine
  green). The ICG images were taken after the red/blue injection, on the same
  placentas.

## Getting the data

Direct, no account: Zenodo record
[8045940](https://zenodo.org/records/8045940) (CC BY-NC-SA 4.0). Without a local
`imagettes/`, `fetch_placenta()` downloads the four zips of the annotated images
P007-P101 (7.86 GB, md5 of the record checked) into
`<cache>/Hyperspectral_Placenta_Dataset/downloads/` and rebuilds the windows with
`utils.placenta_preprocess.build_windows` (same rule as the LISTIC extraction,
written image by image). On BERTHA the rebuild gives the same 77 203 windows of
81 images, and the 24 519 windows of P007-P030 are bit-identical to the local
copy. The other files of the record (16.7 GB) are not used.

```python
fetch_placenta(archive="~/Downloads/placenta")  # folder with the four zips or their folders
```

## Raw data

The loader reads the local derived version `imagettes/`:
`placenta_windows_data.npy` (77 203 × 24 × 24 × 37, float32) and
`placenta_windows_metadata.json`. The metadata give `sample_id` (the source
image), `dye_type`, `class`, `original_class` and `position`. The windows do
not overlap (grid of step 24) and have a purity ≥ 0.9. They come from the 81
images P007–P087 only:

- P001–P006 (no dye) and P088–P101 (77 bands up to 900 nm) are **not used**;
- the "Artery, ICG" and "Stroma, ICG" annotations are merged into Artery and
  Stroma (`original_class` keeps them);
- the "Specular reflection" class of the masks is not kept.

The raw output is the 24 × 24 × 37 window; the figure shows its band-mean
image.

```{dataset-figure} placenta/raw_samples
```

## Classes

Stroma 56 110, Artery 10 142, Vein 7 308, Umbilical 3 188 and Suture 114
windows (imbalance ≈ 490). The mean spectra show a step near 645 nm in every
class, probably a filter change of the sensor.

```{dataset-figure} placenta/class_hist
```

```{dataset-figure} placenta/class_means
```

## Covariance construction

1. **Bands**: all 37 bands (515.2–697.5 nm), `band_step=1`. With n = 576,
   n / 2p = 7.8, so no band subsampling is needed.
2. **Observations**: the 576 pixels of the window. Windows with fewer than 2p
   valid pixels would be discarded: none are.
3. **Estimator**: SCM (sample covariance matrix) by default. **341 SCM are
   singular** (0.44 %, the atlas says 0.45 %); they are discarded and counted
   in `info["excluded_not_spd"]`. Another estimator (`"ledoit_wolf"`, ...)
   may keep them. The splits are drawn on the list of all windows, so two
   estimators share the same test images.
4. **Normalisation**: covariances are scaled on the training part of `split`
   (geometric mean of the eigenvalues = 1). Means and raw windows are
   multiplied by √s.

```{dataset-figure} placenta/cov_samples
```

```{dataset-figure} placenta/eigen_spectrum
```

## Splits

The article describes no classification split. The loader offers two:

- **`by_image` (recommended)**: `StratifiedGroupKFold(5)` with groups =
  `sample_id` (81 images, seed 0). Test = fold `fold` and validation = fold
  `fold + 1`. With `fold=0`: 45 939 train, 15 851 validation and 15 072 test
  windows.
- **`dye_transfer`**: train on the 47 `red_blue` images (46 303 windows),
  test on the 34 ICG images (30 559). There is no validation part. This
  measures transfer between dyes, not between subjects.

```{dataset-figure} placenta/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| adjacent windows of one image on both sides | groups = source image; checked by the real-data test | none |
| same placenta in train and test (81 images from 4 placentas, no placenta identifier) | — | **not measurable**: a placenta is almost surely on both sides; successive images of one area too |
| same placenta under two dyes (`dye_transfer`) | — | the subjects are shared by construction |
| singular SCM | discarded and counted | 0.44 % of the windows |

## Supported tasks

- `task="fine"`: 5 classes. There is no coarse map. `drop_classes=["Suture"]`
  removes the smallest class (114 windows). `merge_classes={"Artery":
  "Vessel", "Vein": "Vessel", "Umbilical": "Vessel"}` gives three classes:
  Vessel, Stroma and Suture.
- `balance="undersample"` is advisable, because Stroma holds 73 % of the
  windows. `"oversample"` and `"weights"` also act on the training part only.

## Usage

```python
from spdnet_datasets.hyperspectral.placenta import PlacentaDataset

ds = PlacentaDataset()                                     # ds.splits: by_image, dye_transfer
train = PlacentaDataset(subset="train", balance="undersample", estimator="ledoit_wolf")
dye = PlacentaDataset(split="dye_transfer")
```

## Migration notes

- The former loader passed a torch tensor to the NumPy estimator in raw mode.
  The raw output is now a NumPy window converted once.
- `split="train"` raises an error: use `subset="train"`. `subset=None` is the
  default.
- Use `estimator=`. `cov_method=...` is silently absorbed by `**kwargs` and
  gives the SCM.

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/placenta.md
```
````
