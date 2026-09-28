# Datasets

## Catalogue

Generated from the registry, so it lists every loader of the installed
version. *Specific options* are the constructor arguments beyond the common
ones of {doc}`loading` (`max_classes`, `max_samples_per_class`, `seed`, …).

```{dataset-catalogue}
```

## Modes of the hyperspectral loaders

The hyperspectral loaders (HyperLeaf, Chikusei, GSOFF, UAV, DeepHS-Fruit,
Placenta, Kaggle Wheat) can serve the same images in several forms:

`mode="cov"`
: Precomputed covariance matrices, read from disk (fastest). Produced by the
  precomputation scripts of {doc}`covariance`.

`mode="raw"`
: Covariance computed on the fly from the hyperspectral image, with
  `cov_method` (`"scm"` or `"ledoit_wolf"`) and `exclude_zero` (ignore the
  background pixels, zero in every band).

`mode="image"` (HyperLeaf)
: The image itself, restricted to the bands given in `bands` (default
  `(81, 51, 21)`, about 640 / 550 / 460 nm, i.e. RGB-like), standardized per
  band: `(len(bands), 48, 352)` float32 tensors, for a CNN backbone followed by
  covariance pooling (`spdnet-training`'s backbones).

`preload=True` reads everything into memory at construction.

## Expected layouts

HyperLeaf (`subdir: HyperLeaf2024`, 1590 leaves; `task` = `"cultivar"`
(4 classes), `"fertilizer"` (3) or `"combined"` (12)):

```text
HyperLeaf2024/
├── cov/<id>.pt        # 204×204 covariances (mode="cov")
├── images/<id>.tiff   # 204 bands × 48 × 352, one TIFF page per band
└── train.csv          # labels
```

HDM05 (2086 sequences, 117 classes):

```text
HDM05/
└── <id>_<frames>_<class>.npy   # 93×93 covariances of the joint coordinates
```

The HDM05 matrices are small (geometric mean of the eigenvalues about
$9 \cdot 10^{-3}$); the SPDNet batch normalization experiments scale them by
`scaling_factor=190`, while the GBWBN reference code uses them unscaled. The
Bures–Wasserstein geometry is not scale invariant, so this choice matters for
GBWBN.

FUSAR-Ship (`subdir: FUSAR_Ship1.0`; Gaofen-3 SAR ship chips, one channel):

```text
FUSAR_Ship1.0/
├── <Category>/<SubCategory>/Ship_CxxSyyNzzzz.tiff   # 512×512 uint8, 5243 chips
└── meta.csv                                         # AIS match-up (MMSI, length, width, ...)
```

The classes are very imbalanced: Cargo 2114, Other 1644, Fishing 789,
Tanker 248, Unspecified 142, Tug 64, Passenger 64, Dredger 61, Reserved 37,
LawEnforce 35, then 16, 15, 6, 5 and 3 chips. `min_samples_per_class`
(default 30) drops the rare classes first, then `max_samples_per_class` caps
the others; with the defaults and a cap of 350, 10 classes and 1701 chips
remain. `class_level="fine"` uses the sub-categories; `exclude_classes`
drops named classes (e.g. the non-type categories `Other`, `Unspecified`,
`Reserved`). Chips are returned as `(1, image_size, image_size)` float32
images: `log1p` of the SAR intensity, resized (default 224), standardized per
chip. Split with `stratified: true`.

```yaml
name: fusar
path: /DATA
subdir: FUSAR_Ship1.0
min_samples_per_class: 30
max_samples_per_class: 350
image_size: 224
stratified: true
```

The docstring of each class (see {doc}`../reference/real`) gives the layout
of the other datasets.
