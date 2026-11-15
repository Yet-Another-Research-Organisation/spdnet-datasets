"""Real online sources of the hyperspectral and SAR fetches (``pytest -m "slow and network"``).

Only files < 50 MB are downloaded; the big archives are probed with a 4-byte
HTTP Range request (magic bytes and total size from ``Content-Range``).
Kaggle and gated Hugging Face tests are skipped without credentials.
"""

import base64
import json
import os
import urllib.parse
import urllib.request

import pytest

from spdnet_datasets.core import access as A
from spdnet_datasets.core.fetch import _NoCrossHostAuth, download, md5sum
from spdnet_datasets.hyperspectral import chikusei as CHK
from spdnet_datasets.hyperspectral import hyperleaf as HL
from spdnet_datasets.hyperspectral import kaggle_wheat as KW
from spdnet_datasets.hyperspectral import placenta as PL
from spdnet_datasets.hyperspectral import rices90 as RI
from spdnet_datasets.sar import air_polsar_seg as AIR
from spdnet_datasets.sar import atrnet_star as ATR
from spdnet_datasets.sar import fusar_ship as FUS
from spdnet_datasets.sar import pol_insar_island as POL

pytestmark = [pytest.mark.slow, pytest.mark.network]

SMALL = {  # Zenodo 8045940, md5 of the record
    "Code.zip": ("12512a516b91f564b1aec83a171a709b", 14_626),
    "placenta_HSI_description.xlsx": ("b9a19a0c2e1299b822d6014b361eddac", 58_151),
}


@pytest.mark.parametrize("name", SMALL)
def test_small_zenodo_files_of_the_placenta_record(name, tmp_path):
    md5, size = SMALL[name]
    url = PL.ZENODO + urllib.parse.quote(name) + "?download=1"
    path = download(url, tmp_path / name, checksum=md5, algo="md5")
    assert path.stat().st_size == size and md5sum(path) == md5
    assert download(url, path, checksum=md5, algo="md5") == path  # idempotent


PROBES = [  # (url, magic bytes, total size)
    (CHK.URL, b"PK\x03\x04", CHK.ZIP_SIZE),
    (RI.URL, b"PK\x03\x04", RI.ZIP_SIZE),
    *[(PL.ZENODO + urllib.parse.quote(n) + "?download=1", b"PK\x03\x04", s)
      for n, (_, s) in PL.ZIPS.items()],
    (POL.TAR_URL, b"10.3", POL.TAR_SIZE),  # tar: first member "10.35097-1700/"
    (AIR.URL, b"PK\x03\x04", AIR.ZIP_SIZE),
    (FUS.URL, b"PK\x03\x04", FUS.ZIP_SIZE),
]


@pytest.mark.parametrize("url, magic, size", PROBES, ids=lambda x: str(x)[:60])
def test_direct_archive_urls_serve_the_expected_file(url, magic, size):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-3"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        head = resp.read(4)  # never read() the body: a server ignoring Range sends it all
        status, content_range = resp.status, resp.headers.get("Content-Range", "")
    assert head == magic
    assert status == 206 and content_range.endswith(f"/{size}")


def test_atrnet_star_public_tree_matches_the_registered_volumes():
    for (rep, _), (name, sizes) in ATR.HF_ARCHIVES.items():
        folder = urllib.parse.quote(name.rsplit("/", 1)[0])
        url = f"https://huggingface.co/api/datasets/{ATR.HF_REPO}/tree/main/{folder}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            tree = {e["path"]: e.get("size") for e in json.load(resp)}
        for k, size in enumerate(sizes, start=1):
            assert tree[f"{name}.{k:03d}"] == size, (rep, name, k)
        assert f"{name}.{len(sizes) + 1:03d}" not in tree


def test_gated_atrnet_star_volume_with_a_token():
    token = A.hf_token(interactive=False, required=False)
    if not token:
        pytest.skip("no Hugging Face token (HF_TOKEN)")
    name, sizes = ATR.HF_ARCHIVES[("Slant_Range", "SOC_40classes")]
    url = f"https://huggingface.co/datasets/{ATR.HF_REPO}/resolve/main/{name}.001"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                               "Range": "bytes=0-5"})
    with urllib.request.build_opener(_NoCrossHostAuth).open(req, timeout=60) as resp:
        assert resp.read(6) == b"7z\xbc\xaf\x27\x1c"


@pytest.mark.parametrize("competition", [KW.COMPETITION, HL.COMPETITION])
def test_kaggle_competition_files_with_credentials(competition):
    if not (os.environ.get("KAGGLE_USERNAME") or os.path.exists(os.path.expanduser(
            os.path.join(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle"), "kaggle.json")))):
        pytest.skip("no Kaggle credentials")
    user, key = A.kaggle_credentials(interactive=False)
    token = base64.b64encode(f"{user}:{key}".encode()).decode()
    url = f"https://www.kaggle.com/api/v1/competitions/data/list/{competition}"
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        assert resp.status == 200
