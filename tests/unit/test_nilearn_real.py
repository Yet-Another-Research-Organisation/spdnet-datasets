"""nilearn adapter on real data: ABIDE PCP, 20 subjects from 4 sites, TT atlas (p = 97).

Run with ``pytest -m "slow and network"``. nilearn downloads the ``rois_tt``
``.1D`` files (about 20 x 150 kB) into ``$NILEARN_DATA``; the covariances go to
``tmp_path``.
"""

import numpy as np
import pytest

from spdnet_datasets.core.estimators import check_spd
from spdnet_datasets.core.splits import assert_no_leakage
from spdnet_datasets.nilearn import NilearnConnectomes

pytestmark = [pytest.mark.slow, pytest.mark.network]

pytest.importorskip("nilearn")

# 5 subjects of each of 4 sites whose scans have T >= 2 * 97 time points
SUBJECTS = {"PITT": [50003, 50004, 50005, 50006, 50007],
            "USM": [50433, 50434, 50435, 50436, 50437],
            "UM_1": [50272, 50273, 50274, 50275, 50276],
            "YALE": [50551, 50552, 50555, 50557, 50558]}


def test_abide_tt_20_subjects(tmp_path):
    ids = [s for subs in SUBJECTS.values() for s in subs]
    ds = NilearnConnectomes("abide", atlas="tt", fetcher_kwargs={"SUB_ID": ids},
                            return_raw=True, cache_dir=tmp_path)
    print("kept", len(ds), "excluded short", ds.n_excluded_short, "not SPD",
          ds.n_excluded_not_spd, "T", sorted(set(ds.metadata["n_timepoints"])))
    assert len(ds) + ds.n_excluded == 20 and ds.n_excluded_short == 0
    assert len(ds) >= 12
    (cov, raw), y = ds[0]
    assert tuple(cov.shape) == (97, 97) and raw.shape[1] == 97 and raw.shape[0] >= 194
    check_spd(ds.covariances)
    assert set(ds.sites) <= set(SUBJECTS) and ds.metadata["subject"].is_unique
    sg = ds.splits["site_grouped"]
    assert ds.splits["recommended"] is sg
    assert_no_leakage(ds.sites[sg["train"]], ds.sites[sg["test"]], what="site")
    assert np.isin(ds.labels, [0, 1]).all()
