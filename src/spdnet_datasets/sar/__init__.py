"""SAR datasets.

With covariances: CPAZMaL (:mod:`.cpazmal`), Pol-InSAR-Island
(:mod:`.pol_insar_island`). Raw or mean outputs only (``OUTPUTS = ("raw",
"mean")``): ATRNet-STAR, FUSAR-Ship, MSTAR, AIR-PolSAR-Seg, S1SLC_CVDL.
"""

from .cpazmal import CPAZMaLDataset, fetch_cpazmal

__all__ = ["CPAZMaLDataset", "fetch_cpazmal"]
