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

The docstring of each class (see {doc}`../reference/real`) gives the layout
of the other datasets.
