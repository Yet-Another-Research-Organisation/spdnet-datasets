# HyperPistachio

```{warning}
⚠️ **Outside the maintained selection — no loader.** HyperPistachio is
documented here for reference only. `spdnet-datasets` provides no loader, no
`fetch_*` function and no registry name for it. The notes below summarise
the dataset atlas fiche.
```

## Description and source

HyperPistachio is the public subset of a dataset of pistachio nuts
contaminated with aflatoxin B1 (AFB1). It contains 52 hyperspectral cubes
taken by a Resonon Pika XC2 laboratory imager: 256 × 384 pixels, with 462
bands from 386.88 to 1 003.60 nm. The cubes are stored as ENVI (Environment for Visualizing Images) files in BIL
(band interleaved by line) order, uint16, with reflectance × 10 000. Each
cube shows **one** pistachio on a dark background. There are 26 levels of
controlled contamination, from 0 to 114.67 µg/kg, with 2 pistachios per
level.

- Reference: A. Sheikh-Akbari and H. Mehrabinejad, "HyperPistachio: A
  Hyperspectral Image Dataset of Aflatoxin B1 Contaminated Pistachio Nuts",
  Zenodo, 5 May 2026,
  doi:[10.5281/zenodo.20027441](https://doi.org/10.5281/zenodo.20027441).
  This reference was verified. No article is associated with it yet.
- **Licence conflict**: the Zenodo metadata say CC BY-ND 4.0. The
  description, however, restricts the subset to academic teaching and
  research, without redistribution or commercial use. Apply the stricter
  rule and do not redistribute the data.

## Why it is not loaded

- **52 objects only**, so any covariance-based benchmark has a very high
  variance. Windows inside a pistachio (22 × 22 or 32 × 32 pixels) multiply
  the samples, but not the objects.
- **Contamination level confounded with the session**: the two pistachios of
  a level share their shutter time, frame rate and acquisition minute. The
  Spearman correlation between shutter and concentration is 0.87.
- **Full spectral covariance**: p = 462 with no band grouping. A window of
  22 × 22 gives n = 484 ≈ p, below the n ≥ 2p rule of this package. A 32 × 32
  window gives n / p = 2.2, so n / 2p = 1.1.
- **No official split.** The concentration is given per level (batch), not
  measured per nut.

## If you use it

- Split by level (`StratifiedGroupKFold`, group = level). The two pistachios
  of a level, and all their windows, must stay on the same side.
- Control the confounder. One way is to restrict the data to the 13 levels
  that share the 14.74 ms shutter time.
- Threshold the concentration at the European Union (EU) limit for AFB1 in pistachios sold to
  the final consumer: 8 µg/kg (Regulation (EU) 2023/915).
- Clip the specular highlights, where reflectance exceeds 1 (up to 14 % of the
  values of one cube).
