# Audio

Clips and recordings become the covariance of their MFCC (or log-mel) frames: one SPD matrix per clip, with the official folds or recording-grouped splits.

::::{grid} 1 2 3 3
:gutter: 3
:class-container: dataset-tiles

:::{grid-item-card} ESC-50
:link: esc50
:link-type: doc
:img-top: ../_static/previews/esc50/thumb.png
:img-alt: preview of ESC-50
:class-card: sd-card-hover

Environmental sounds, 5 s clips

{bdg-secondary-line}`20×20 MFCC` {bdg-secondary-line}`50 classes` {bdg-secondary-line}`1 979 clips`
:::

:::{grid-item-card} GTZAN
:link: gtzan
:link-type: doc
:img-top: ../_static/previews/gtzan/thumb.png
:img-alt: preview of GTZAN
:class-card: sd-card-hover

Music genres, 30 s excerpts

{bdg-secondary-line}`20×20 MFCC` {bdg-secondary-line}`10 genres`
:::

:::{grid-item-card} BirdSet XCM
:link: birdset_xcm
:link-type: doc
:img-top: ../_static/previews/birdset_xcm/thumb.png
:img-alt: preview of BirdSet XCM
:class-card: sd-card-hover

Bird species from Xeno-Canto recordings

{bdg-secondary-line}`20×20 MFCC` {bdg-secondary-line}`95 species`
:::

:::{grid-item-card} UrbanSound8K
:link: urbansound8k
:link-type: doc
:img-top: ../_static/previews/urbansound8k/thumb.png
:img-alt: preview of UrbanSound8K
:class-card: sd-card-hover

Urban sounds; reference page, no loader

{bdg-secondary-line}`10 classes` {bdg-warning-line}`no loader`
:::

::::

```{toctree}
:hidden:

esc50
gtzan
birdset_xcm
urbansound8k
```
