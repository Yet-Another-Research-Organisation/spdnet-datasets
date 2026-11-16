"""Audio datasets: one covariance of frame features (MFCC or log-mel) per recording.

- :mod:`.esc50`: ESC-50, environmental sounds, 5 official folds.
- :mod:`.gtzan`: GTZAN, music genres, exact duplicates removed, stratified folds.
- :mod:`.birdset_xcm`: BirdSet XCM (partial copy), bird species, folds grouped by recordist.

Decoding needs the ``audio`` extra (``pyarrow``, ``soundfile``), imported lazily.
"""

from .birdset_xcm import BirdSetXCMDataset, fetch_birdset_xcm
from .esc50 import ESC50Dataset, fetch_esc50
from .gtzan import GTZANDataset, fetch_gtzan

__all__ = ["BirdSetXCMDataset", "ESC50Dataset", "GTZANDataset", "fetch_birdset_xcm",
           "fetch_esc50", "fetch_gtzan"]
