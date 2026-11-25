<!--
Template of a dataset page. Copy to <name>.md, fill every section, then run
`python doc/tools/make_figures.py <registered name> --kwargs '{...}' --split <split>`
on a machine holding the data to produce doc/_static/figures/<name>/ (fixed
PNG figures and interactive JSON, see doc/tools/figures_io.py), and
`python doc/tools/make_previews.py <name>` for doc/_static/previews/<name>/.
Add the page to the toctree and the tiles of its domain page
(doc/datasets/<domain>.md), and its publication to doc/references.md
(doc/datasets/_cite/<name>.md). This file is excluded from the build (conf.py: exclude_patterns).
-->
# <Dataset name>

```{list-table}
:stub-columns: 1

* - Domain
  - <domain>
* - Task(s)
  - fine classification (<k> classes) · coarse · regression
* - Covariance
  - <p>×<p>, n = <n> observations (n / 2p ≥ <ratio>)
* - Outputs
  - cov · mean · raw (<shape>)
* - Splits
  - `official` (<sizes>) · `recommended` (<sizes>) · `random` (sample-level 70 / 10 / 20, ignores the groups, see {doc}`/guide/splits`)
* - Module
  - {mod}`spdnet_datasets.<domain>.<module>` (`fetch_<name>`, `<Name>Dataset`)
```

## Description and source

<What is measured, by which sensor, where and when. Reference paper with DOI and
link; licence; where the data come from.>

## Raw data

<Layout of the files, shapes, value ranges.>

```{dataset-figure} <name>/raw_samples
```

## Classes

```{dataset-figure} <name>/class_hist
```

```{dataset-figure} <name>/class_means
```

## Covariance construction

<Windows / parcels / segments, filters, band selection, normalisation, dates
kept, number of observations per covariance, samples discarded and why.>

```{dataset-figure} <name>/cov_samples
```

```{dataset-figure} <name>/eigen_spectrum
```

## Splits

<Official split from the paper, or the proposed leakage-free split and why the
paper's split was not used. Sizes per split.>

```{dataset-figure} <name>/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| <same subject / overlapping windows / ...> | <grouping / buffer / ...> | <none / documented> |

## Supported tasks

<task="fine" | "coarse" (coarse map) | "regression" (targets); classes that can be
dropped or merged; balancing.>

## Usage

```python
from spdnet_datasets.<domain>.<module> import <Name>Dataset
ds = <Name>Dataset(split="official", subset="train")
```
