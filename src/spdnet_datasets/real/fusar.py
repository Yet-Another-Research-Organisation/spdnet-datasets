"""Deprecated location of the FUSAR-Ship loader: see :mod:`spdnet_datasets.sar.fusar_ship`.

Kept so that ``from spdnet_datasets.real.fusar import FUSARShipDataset`` and the
``"fusar"`` registry name keep working.
"""

from ..sar.fusar_ship import FUSARShipDataset, fetch_fusar_ship

__all__ = ["FUSARShipDataset", "fetch_fusar_ship"]
