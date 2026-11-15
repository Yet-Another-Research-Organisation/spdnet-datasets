# MSTAR

```{list-table}
:stub-columns: 1

* - Domain
  - SAR (synthetic aperture radar), automatic target recognition
* - Task(s)
  - fine classification (10 vehicle classes)
* - Covariance
  - none: raw or mean outputs only
* - Outputs
  - raw (128, 128, 1) intensity, amplitude, or (128, 128, 2) complex · mean (per channel)
* - Splits
  - `official` (SOC: 17° train, 15° test; 2 472 / 275 / 2 425)
* - Module
  - {mod}`spdnet_datasets.sar.mstar` (`fetch_mstar`, `MSTARDataset`)
```

## Description and source

MSTAR (Moving and Stationary Target Acquisition and Recognition) is the public
release of the AFRL/DARPA program from the 1990s. AFRL is the US Air Force
Research Laboratory and DARPA the Defense Advanced Research Projects Agency.
A Sandia airborne radar imaged ten ground vehicles in X band (about 9.6 GHz),
HH polarisation (horizontal emission and reception), at a resolution of about
0.3 m × 0.3 m. There is one chip per azimuth over 0–360°, at two depression
angles: 17° and 15°.

The loader implements the 10-class SOC (standard operating conditions)
protocol: training at 17°, test at 15°.

- Ross T. D. et al., *Standard SAR ATR evaluation experiments using the MSTAR
  public release data set*, Proc. SPIE 3370, 1998, doi:10.1117/12.321859.
- Keydel E. R., Lee S. W., Moore J. T., *MSTAR extended operating conditions:
  a tutorial*, Proc. SPIE 2757, 1996, doi:10.1117/12.242059.
- Copy used: the GitHub mirror <https://github.com/omae11/MSTAR-dataset>. Its
  licence restricts redistribution to non-commercial academic use. The data
  must be downloaded by hand and placed under `<root>/SAR/MSTAR_SOC`.

## Raw data

```text
soc/{train,test}/<class>/HBxxxxx.jpg           8-bit intensity
soc_complex/{train,test}/<class>/HBxxxxx.0nn   Phoenix header + big-endian float32 amplitude, then phase
```

Chip sizes depend on the class, from 128 × 128 to 193 × 192 pixels. Every chip
is centre-cropped to `crop × crop` pixels (default 128, the smallest size), so
all samples have the same shape. `source` selects the representation:

- `"jpg"` (default): (128, 128, 1) intensity;
- `"amplitude"`: (128, 128, 1) amplitude of the complex chip;
- `"complex"`: (128, 128, 2) real and imaginary parts.

```{raw} html
<iframe src="../_static/figures/mstar/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/mstar/amplitude_hist.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

Ten vehicles: 2S1, BMP2, BRDM2, BTR60, BTR70, D7, T62, T72, ZIL131 and ZSU234.
The classes are nearly balanced: 195 to 274 test chips per class. The second
figure shows the distribution of the chip mean for each class.

```{raw} html
<iframe src="../_static/figures/mstar/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/mstar/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

The loader builds no covariance. The mission restricts the SAR chip datasets
to raw or mean outputs (`OUTPUTS = ("raw", "mean")`), and asking for a
covariance raises `ValueError`. Exactly one of `return_raw=True` and
`return_mean=True` must be set. The mean is the per-channel mean of the
cropped chip. A covariance model can be built from the raw chip, for example
over pixel neighbourhoods. That choice is left to the user.

## Splits

`official` follows the SOC protocol:

| Part | Chips | Definition |
|---|---|---|
| train | 2 472 | 17° depression, minus the validation draw |
| val | 275 | 10 % of the 17° chips (`val_size`), stratified by class |
| test | 2 425 | 15° depression |

The `groups` attribute holds the file stem (`HBxxxxx`). It is a file name, not
a physical unit: the same stem appears in several classes (for example
`HB03333` in BMP2, BTR60, BTR70 and T72). The group-size histogram below
therefore says nothing about leakage. The right-hand panel shows group sizes
of 1 to 6.

```{raw} html
<iframe src="../_static/figures/mstar/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| Same vehicle in train and test | none possible: SOC images one vehicle per class at 17° and 15° | **inherent**: SOC measures generalisation over 2° of depression, not recognition of new vehicles |
| Near-identical views | none: 99.8 % of test chips have a training chip of the same class within 2° of azimuth (atlas measurement) | inherent to the protocol |
| Exact duplicates | md5 checked: no file appears in both train and test | none |
| Validation drawn from the training chips | stratified draw | the validation score is optimistic (same depression, neighbouring azimuths) |

## Supported tasks

- `task="fine"`: 10 classes.
- `drop_classes` / `merge_classes` take class names.
- `balance` acts on the training part only.
- There is no coarse map and no regression target.

## Usage

```python
from spdnet_datasets.sar.mstar import MSTARDataset

ds = MSTARDataset(source="jpg", crop=128, return_raw=True)
x, y = ds[0]                         # (128, 128, 1) float32, class index
train_idx = ds.splits["official"]["train"]
```
