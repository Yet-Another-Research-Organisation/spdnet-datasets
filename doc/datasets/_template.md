<!--
Template of a dataset page. Copy to <name>.md, fill every section, then run
`python doc/tools/make_figures.py <registered name> --kwargs '{...}' --split <split>`
on a machine holding the data to produce doc/_static/figures/<name>/*.html.
This file is excluded from the build (conf.py: exclude_patterns).
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
  - `official` (<sizes>) · `recommended` (<sizes>)
* - Module
  - {mod}`spdnet_datasets.<domain>.<module>` (`fetch_<name>`, `<Name>Dataset`)
```

## Description and source

<What is measured, by which sensor, where and when. Reference paper with DOI and
link; licence; where the data come from.>

## Raw data

<Layout of the files, shapes, value ranges.>

```{raw} html
<iframe src="../_static/figures/<name>/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

```{raw} html
<iframe src="../_static/figures/<name>/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/<name>/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

<Windows / parcels / segments, filters, band selection, normalisation, dates
kept, number of observations per covariance, samples discarded and why.>

```{raw} html
<iframe src="../_static/figures/<name>/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/<name>/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

## Splits

<Official split from the paper, or the proposed leakage-free split and why the
paper's split was not used. Sizes per split.>

```{raw} html
<iframe src="../_static/figures/<name>/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
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
