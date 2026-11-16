"""SAR datasets.

With covariances: CPAZMaL (:mod:`.cpazmal`), Pol-InSAR-Island
(:mod:`.pol_insar_island`), S1 Iceland SAR (:mod:`.s1_iceland`, local only). Raw or mean outputs only (``OUTPUTS = ("raw",
"mean")``): ATRNet-STAR, FUSAR-Ship, MSTAR, AIR-PolSAR-Seg, S1SLC_CVDL
(``zarr`` is imported lazily, extra ``sar``).
"""

from .air_polsar_seg import AIRPolSARSegDataset, fetch_air_polsar_seg
from .atrnet_star import ATRNetSTARDataset, fetch_atrnet_star
from .cpazmal import CPAZMaLDataset, fetch_cpazmal
from .fusar_ship import FUSARShipDataset, fetch_fusar_ship
from .mstar import MSTARDataset, fetch_mstar
from .pol_insar_island import PolInSARIslandDataset, fetch_pol_insar_island
from .s1_iceland import S1IcelandDataset, fetch_s1_iceland
from .s1slc_cvdl import S1SLCCVDLDataset, fetch_s1slc_cvdl

__all__ = [
    "S1IcelandDataset", "fetch_s1_iceland",
    "AIRPolSARSegDataset", "ATRNetSTARDataset", "CPAZMaLDataset", "FUSARShipDataset",
    "MSTARDataset", "PolInSARIslandDataset", "S1SLCCVDLDataset",
    "fetch_air_polsar_seg", "fetch_atrnet_star", "fetch_cpazmal", "fetch_fusar_ship",
    "fetch_mstar", "fetch_pol_insar_island", "fetch_s1slc_cvdl",
]
