"""Online access of the hyperspectral and SAR fetches, on a simulated network.

Direct sources: checksum (or size) checked, idempotent (no second request).
Kaggle / gated Hugging Face: authentication header. Manual sources:
``ManualDownloadRequired``, then ingestion of a fake archive giving the same
result as the local copy.
"""

import base64
import hashlib
import io
import json
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile

import h5py
import numpy as np
import pytest
import tifffile

from spdnet_datasets.core import access as A
from spdnet_datasets.core.access import ManualDownloadRequired, MissingCredentialsError
from spdnet_datasets.hyperspectral import chikusei as CHK
from spdnet_datasets.hyperspectral import deephsfruit as DHS
from spdnet_datasets.hyperspectral import gsoff as GS
from spdnet_datasets.hyperspectral import hyperleaf as HL
from spdnet_datasets.hyperspectral import kaggle_wheat as KW
from spdnet_datasets.hyperspectral import placenta as PL
from spdnet_datasets.hyperspectral import rices90 as RI
from spdnet_datasets.hyperspectral import uav as UAV
from spdnet_datasets.sar import air_polsar_seg as AIR
from spdnet_datasets.sar import atrnet_star as ATR
from spdnet_datasets.sar import cpazmal as CPZ
from spdnet_datasets.sar import fusar_ship as FUS
from spdnet_datasets.sar import mstar as MST
from spdnet_datasets.sar import pol_insar_island as POL
from spdnet_datasets.sar import s1slc_cvdl as SLC
from spdnet_datasets.utils import sources

from .test_hsi_deephsfruit import fruit_dir  # noqa: F401
from .test_hsi_gsoff import gsoff_dir  # noqa: F401
from .test_hsi_hyperleaf import leaf_dir  # noqa: F401
from .test_hsi_kaggle_wheat import wheat_dir  # noqa: F401
from .test_hsi_rices90 import rices_dir  # noqa: F401
from .test_hsi_uav import uav_dir  # noqa: F401
from .test_sar_air_polsar_seg import air_dir  # noqa: F401
from .test_sar_atrnet_star import atr_dir  # noqa: F401
from .test_sar_cpazmal import cpaz_dir  # noqa: F401
from .test_sar_fusar_ship import fusar_dir  # noqa: F401
from .test_sar_mstar import mstar_dir  # noqa: F401
from .test_sar_pol_insar_island import CELL, pol_dir  # noqa: F401


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    monkeypatch.setenv("SPDNET_OPEN_BROWSER", "0")
    monkeypatch.setattr(A, "_keyring_get", lambda name: None)
    for var in ("KAGGLE_USERNAME", "KAGGLE_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "no_kaggle"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "no_hf"))


def _zip_tree(src, dest, top="pkg", extra=None):
    with zipfile.ZipFile(dest, "w") as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, f"{top}/{p.relative_to(src)}" if top else str(p.relative_to(src)))
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return dest


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Resp(io.BytesIO):
    status = 200

    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def auth_net(monkeypatch):
    """Authenticated downloads: ``serve(url, bytes)``; ``seen`` = (url, Authorization)."""
    served, seen = {}, []

    class _Opener:
        def open(self, req, timeout=None):
            seen.append((req.full_url, req.get_header("Authorization")))
            return _Resp(served[req.full_url])

    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: _Opener())
    return served.__setitem__, seen


# --- hyperspectral: direct ---------------------------------------------------------------------

def _chikusei_zip(tmp_path):
    from scipy.io import savemat

    rng = np.random.default_rng(0)
    H = W = 80
    cube = rng.integers(1, 4000, size=(H, W, 128)).astype(np.uint16)
    gt = (1 + (np.arange(H)[:, None] // 10 + np.arange(W)[None] // 10) % 2).astype(np.uint8)
    src = tmp_path / "chk_src" / "Chikusei_MATLAB"
    src.mkdir(parents=True)
    with h5py.File(src / CHK.SCENE_FILE, "w") as f:
        f["chikusei"] = cube.transpose(2, 1, 0)  # MATLAB v7.3 order (B, W, H)
    savemat(src / CHK.GT_FILE, {"GT": {"gt": gt, "classnames": np.array(["Water", "Forest"], dtype=object),
                                       "classcolor": np.zeros((2, 3))}})
    return _zip_tree(src.parent, tmp_path / CHK.ZIP, top=""), cube, gt


def test_chikusei_download_rebuilds_the_v2_windows(tmp_path, fake_net, monkeypatch):
    serve, calls = fake_net
    archive, cube, gt = _chikusei_zip(tmp_path)
    serve(CHK.URL, archive.read_bytes())
    monkeypatch.setattr(CHK, "ZIP_SIZE", archive.stat().st_size)
    monkeypatch.setattr(CHK, "BLOCK", 40)  # 2 x 2 blocks: ids 0, 1, 8, 9 -> train, train, val, train
    b = CHK.fetch_chikusei(tmp_path / "missing", cache_dir=tmp_path / "c")
    assert calls == [CHK.URL]
    assert b.root == sources.downloads_dir(CHK.NAME, tmp_path / "c")
    assert set(b.labels) <= {"Water", "Forest"} and b.covariances.shape[1:] == (64, 64)
    with h5py.File(b.prepared_path) as f:
        C, split = f["w15/centers"][:], f["w15/split"][:]
        assert (C % CHK.STRIDE == 0).all()
        r, c = C[0]
        assert np.array_equal(f["w15/windows"][0], cube[r - 7:r + 8, c - 7:c + 8])
        assert np.array_equal(f["w15/labels"][:], gt[C[:, 0], C[:, 1]])
    val = (C[:, 0] >= 40) & (C[:, 1] < 40)
    assert (split[val] != 0).all() and set(split[val]) <= {-1, 1}
    near = ~val & (C[:, 0] > 40 - 16) & (C[:, 1] < 39 + 16)  # closer than w + 1 to val
    assert (split[near] == -1).all()
    CHK.fetch_chikusei(tmp_path / "missing", cache_dir=tmp_path / "c")
    assert calls == [CHK.URL]  # idempotent
    monkeypatch.setattr(CHK, "ZIP_SIZE", 1)
    with pytest.raises(OSError, match="expected 1"):
        CHK.fetch_chikusei(tmp_path / "missing", cache_dir=tmp_path / "c2")


def _stiff(path, cube):
    wl = np.linspace(515, 697, cube.shape[-1]).astype(np.float32)
    with tifffile.TiffWriter(path) as t:
        for i in range(cube.shape[-1]):
            tags = [(65000, "f", len(wl), list(wl), True)] if i == 0 else []
            t.write((cube[..., i] * 65535).astype(np.uint16), extratags=tags)


def _mtiff(path, masks):
    with tifffile.TiffWriter(path) as t:
        for label, mask in masks.items():
            b = label.encode()
            t.write(mask.astype(np.uint8), photometric="minisblack",
                    extratags=[(65001, "s", len(b), b, True)])


def _placenta_zips(tmp_path):
    rng = np.random.default_rng(0)
    out = tmp_path / "zenodo"
    out.mkdir()
    zips = {}
    for k, name in enumerate(PL.ZIPS):
        sid = f"P{7 + 24 * k:03d}"
        cube = rng.random((48, 48, 37)).astype(np.float32)
        masks = {"Artery": np.zeros((48, 48), bool), "Stroma, ICG": np.zeros((48, 48), bool)}
        masks["Artery"][:24, :24] = True
        masks["Stroma, ICG"][24:, :] = True
        d = tmp_path / "plac_src" / name[:-4]
        d.mkdir(parents=True)
        _stiff(d / f"{sid}.tif", cube)
        _mtiff(d / f"{sid}, masks.tif", masks)
        z = _zip_tree(d, out / name, top="")
        zips[name] = (_md5(z), z.stat().st_size)
    return out, zips


def test_placenta_download_rebuilds_the_windows(tmp_path, fake_net, monkeypatch):
    serve, calls = fake_net
    out, zips = _placenta_zips(tmp_path)
    monkeypatch.setattr(PL, "ZIPS", zips)
    for name in zips:
        serve(PL.ZENODO + urllib.parse.quote(name) + "?download=1", (out / name).read_bytes())
    b = PL.fetch_placenta(tmp_path / "missing", cache_dir=tmp_path / "c", n_splits=2)
    assert len(calls) == 4 and all("%20" in u for u in calls)
    # 1 Artery window + 2 Stroma windows per image, 4 images
    assert len(b.labels) == 12 and set(b.labels) == {"Artery", "Stroma"}
    windows = np.load(b.root / PL.WINDOWS)
    assert windows.shape == (12, 24, 24, 37) and windows.dtype == np.float32
    meta = json.loads((b.root / PL.METADATA).read_text())
    assert meta[0] == {"sample_id": "P007", "dye_type": "red_blue", "class": "Artery",
                       "original_class": "Artery", "position": [0, 0]}
    PL.fetch_placenta(tmp_path / "missing", cache_dir=tmp_path / "c", n_splits=2)
    assert len(calls) == 4  # idempotent
    # same zips given by hand (offline)
    b2 = PL.fetch_placenta(tmp_path / "missing", cache_dir=tmp_path / "c3", archive=out, n_splits=2)
    np.testing.assert_allclose(b2.covariances, b.covariances)
    bad = dict(zips)
    first = next(iter(bad))
    bad[first] = ("0" * 32, bad[first][1])
    monkeypatch.setattr(PL, "ZIPS", bad)
    with pytest.raises(OSError, match="Checksum"):
        PL.fetch_placenta(tmp_path / "missing", cache_dir=tmp_path / "c2")


def test_rices90_download_and_checksum(rices_dir, tmp_path, fake_net, monkeypatch):  # noqa: F811
    serve, calls = fake_net
    z = _zip_tree(rices_dir, tmp_path / RI.ZIP, top="RGB and VIS-NIR HSI Data for 90 Rice Seed Varieties",
                  extra={"__MACOSX/RGB and VIS-NIR HSI Data for 90 Rice Seed Varieties/index.csv": "x"})
    serve(RI.URL, z.read_bytes())
    monkeypatch.setattr(RI, "ZIP_SIZE", z.stat().st_size)
    monkeypatch.setattr(RI, "ZIP_MD5", "0" * 32)
    with pytest.raises(OSError, match="Checksum"):
        RI.fetch_rices90(tmp_path / "missing", cache_dir=tmp_path / "c")
    monkeypatch.setattr(RI, "ZIP_MD5", _md5(z))
    b = RI.fetch_rices90(tmp_path / "missing", cache_dir=tmp_path / "c")
    ref = RI.fetch_rices90(rices_dir, cache_dir=tmp_path / "ref")
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert "__MACOSX" not in str(b.root)
    RI.fetch_rices90(tmp_path / "missing", cache_dir=tmp_path / "c")
    assert len(calls) == 2  # the failed attempt, then one download


# --- hyperspectral: Kaggle -------------------------------------------------------------------------

@pytest.mark.parametrize("module, fixture, top, kwargs", [
    (KW, "wheat_dir", "data", {}),
    (HL, "leaf_dir", "", {"band_step": 4}),
])
def test_kaggle_competitions_send_basic_auth(module, fixture, top, kwargs, request, tmp_path,
                                              auth_net, monkeypatch):
    root = request.getfixturevalue(fixture)
    fetch = KW.fetch_kaggle_wheat if module is KW else HL.fetch_hyperleaf
    with pytest.raises(MissingCredentialsError):
        fetch(tmp_path / "missing", cache_dir=tmp_path / "c", **kwargs)
    serve, seen = auth_net
    url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{module.COMPETITION}"
    serve(url, _zip_tree(root, tmp_path / "comp.zip", top=top).read_bytes())
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    b = fetch(tmp_path / "missing", cache_dir=tmp_path / "c", **kwargs)
    assert seen == [(url, "Basic " + base64.b64encode(b"user:secret").decode())]
    ref = fetch(root, cache_dir=tmp_path / "ref", **kwargs)
    np.testing.assert_allclose(b.covariances, ref.covariances)
    fetch(tmp_path / "missing", cache_dir=tmp_path / "c", **kwargs)
    assert len(seen) == 1


# --- hyperspectral: manual ---------------------------------------------------------------------

def _manual_then_archive(fetch, root, tmp_path, archive, compare="covariances", **kw):
    with pytest.raises(ManualDownloadRequired, match="archive="):
        fetch(tmp_path / "missing", cache_dir=tmp_path / "c", **kw)
    b = fetch(tmp_path / "missing", cache_dir=tmp_path / "c", archive=archive, **kw)
    ref = fetch(root, cache_dir=tmp_path / "ref", **kw)
    if compare == "covariances":
        np.testing.assert_allclose(b.covariances, ref.covariances)
    else:
        assert b.meta.drop(columns=[c for c in b.meta if c == "path"]).equals(
            ref.meta.drop(columns=[c for c in ref.meta if c == "path"]))
    # the ingested copy is found again in the cache without archive=
    again = fetch(tmp_path / "missing", cache_dir=tmp_path / "c", **kw)
    assert again.root == b.root
    return b


def test_uav_manual_then_archive_without_source_folder(uav_dir, tmp_path, monkeypatch):  # noqa: F811
    z = _zip_tree(uav_dir / "source", tmp_path / "UAV.zip", top="UAV-HSI-Crop-V6")
    monkeypatch.setattr(UAV, "ARCHIVE_SIZE", z.stat().st_size)
    b = _manual_then_archive(UAV.fetch_uav, uav_dir, tmp_path, z, scenes=("MJK_N", "MJK_S"),
                             block=48)
    assert (b.root / "source" / "Test" / "gt").is_dir()
    monkeypatch.setattr(UAV, "ARCHIVE_SIZE", 1)
    with pytest.raises(OSError, match="size"):
        UAV.fetch_uav(tmp_path / "missing", cache_dir=tmp_path / "c4", archive=z)


def test_gsoff_manual_then_archive(gsoff_dir, tmp_path):  # noqa: F811
    with pytest.raises(ManualDownloadRequired, match="contact the package maintainer"):
        GS.fetch_gsoff(tmp_path / "missing", cache_dir=tmp_path / "c")
    z = _zip_tree(gsoff_dir / "source", tmp_path / "gsoff.zip", top="baidu")
    _manual_then_archive(GS.fetch_gsoff, gsoff_dir, tmp_path, z, block=30)


def test_deephsfruit_manual_then_archive(fruit_dir, tmp_path):  # noqa: F811
    with pytest.raises(ManualDownloadRequired, match="not in the sources"):
        DHS.fetch_deephsfruit(tmp_path / "missing", cache_dir=tmp_path / "c")
    z = _zip_tree(fruit_dir, tmp_path / "fruit.zip", top="DeepHS-Fruit")
    b = DHS.fetch_deephsfruit(tmp_path / "missing", cache_dir=tmp_path / "c", archive=z)
    ref = DHS.fetch_deephsfruit(fruit_dir)
    assert b.meta.equals(ref.meta) and np.allclose(b.covariances[0], ref.covariances[0])


# --- SAR ---------------------------------------------------------------------------------------

def test_cpazmal_archive_is_checked_and_used_in_place(cpaz_dir, tmp_path, monkeypatch):  # noqa: F811
    f = cpaz_dir / CPZ.FILENAME
    with pytest.raises(OSError, match="Checksum"):
        CPZ.fetch_cpazmal(tmp_path / "missing", cache_dir=tmp_path / "c", archive=f, n_dates=20)
    monkeypatch.setitem(CPZ.FILES, CPZ.HF_URL, _sha(f))
    b = CPZ.fetch_cpazmal(tmp_path / "missing", cache_dir=tmp_path / "c", archive=cpaz_dir, n_dates=20)
    ref = CPZ.fetch_cpazmal(cpaz_dir, cache_dir=tmp_path / "ref", n_dates=20)
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert not (tmp_path / "c" / "cpazmal" / "downloads").exists()  # nothing copied


@pytest.mark.parametrize("module", [AIR, FUS])
def test_drive_downloads_checksum_and_quota_fallback(module, request, tmp_path, fake_net,
                                                     monkeypatch):
    serve, calls = fake_net
    if module is AIR:
        root, _ = request.getfixturevalue("air_dir")
        fetch, kw, top = AIR.fetch_air_polsar_seg, {"cell": 8, "macro": 32}, "Raw_AIR-PolarSAR-Seg"
    else:
        fixture = request.getfixturevalue("fusar_dir")  # = tmp_path: copy the dataset files
        root = tmp_path / "fusar_src"
        for name in ("Cargo", "Fishing"):
            shutil.copytree(fixture / name, root / name)
        shutil.copy(fixture / "meta.csv", root / "meta.csv")
        fetch, kw, top = FUS.fetch_fusar_ship, {}, ""
    serve(module.URL, b"<html>Google Drive - quota exceeded</html>")
    with pytest.raises(ManualDownloadRequired, match="automatic download failed"):
        fetch(tmp_path / "missing", cache_dir=tmp_path / "c", **kw)
    z = _zip_tree(root, tmp_path / module.ZIP, top=top)
    serve(module.URL, z.read_bytes())
    monkeypatch.setattr(module, "ZIP_SHA256", _sha(z))
    monkeypatch.setattr(module, "ZIP_SIZE", z.stat().st_size)
    b = fetch(tmp_path / "missing", cache_dir=tmp_path / "c", **kw)
    ref = fetch(root, cache_dir=tmp_path / "ref", **kw)
    assert b.meta.drop(columns="path", errors="ignore").equals(ref.meta.drop(columns="path", errors="ignore"))
    fetch(tmp_path / "missing", cache_dir=tmp_path / "c", **kw)
    assert len(calls) == 2  # quota page, then the file; nothing afterwards


def _bag(pol_root, tmp_path):
    inner = _zip_tree(pol_root, tmp_path / "Pol-InSAR-Island_updated.zip", top="Pol-InSAR-Island")
    tar = tmp_path / POL.TAR
    with tarfile.open(tar, "w") as t:
        t.add(tmp_path / "Pol-InSAR-Island_updated.zip", arcname=f"10.35097-1700/{POL.INNER}")
        info = tarfile.TarInfo("10.35097-1700/manifest-md5.txt")
        data = f"{_md5(inner)}  {POL.INNER}\n".encode()
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    return tar, inner


def test_pol_insar_island_downloads_the_bagit_tar(pol_dir, tmp_path, fake_net, monkeypatch):  # noqa: F811
    serve, calls = fake_net
    tar, inner = _bag(pol_dir, tmp_path)
    serve(POL.TAR_URL, tar.read_bytes())
    monkeypatch.setattr(POL, "TAR_MD5", _md5(tar))
    monkeypatch.setattr(POL, "TAR_SIZE", tar.stat().st_size)
    monkeypatch.setattr(POL, "INNER_MD5", "0" * 32)
    with pytest.raises(OSError, match="Checksum"):
        POL.fetch_pol_insar_island(tmp_path / "missing", cache_dir=tmp_path / "c", cell=CELL)
    monkeypatch.setattr(POL, "INNER_MD5", _md5(inner))
    kw = {"cell": CELL, "block": 16}
    b = POL.fetch_pol_insar_island(tmp_path / "missing", cache_dir=tmp_path / "c", **kw)
    ref = POL.fetch_pol_insar_island(pol_dir, cache_dir=tmp_path / "ref", **kw)
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert not (tmp_path / "c" / "Pol-InSAR-Island" / "downloads" / "bag").exists()
    POL.fetch_pol_insar_island(tmp_path / "missing", cache_dir=tmp_path / "c", **kw)
    assert calls == [POL.TAR_URL]
    monkeypatch.setattr(POL, "TAR_MD5", "0" * 32)  # refused / corrupted download: manual
    with pytest.raises(ManualDownloadRequired, match="automatic download failed"):
        POL.fetch_pol_insar_island(tmp_path / "missing", cache_dir=tmp_path / "c3", **kw)  # downloaded once (the tar is kept)
    # the inner zip given by hand
    b2 = POL.fetch_pol_insar_island(tmp_path / "missing", cache_dir=tmp_path / "c2", archive=inner,
                                    **kw)
    np.testing.assert_allclose(b2.covariances, ref.covariances)


def test_s1slc_manual_then_npy_archive_converted_to_zarr(tmp_path):
    rng = np.random.default_rng(0)
    src = tmp_path / "S1SLC_CVDL"
    data = {}
    for city in SLC.CITIES:
        (src / city).mkdir(parents=True)
        hh = (rng.normal(size=(12, 5, 5)) + 1j * rng.normal(size=(12, 5, 5))).astype(np.complex64)
        hv = (rng.normal(size=(12, 5, 5)) + 1j * rng.normal(size=(12, 5, 5))).astype(np.complex64)
        np.save(src / city / SLC.NPY[0], hh)
        np.save(src / city / SLC.NPY[1], hv)
        np.save(src / city / SLC.NPY[2], (1 + np.arange(12) % 3)[:, None])
        data[city] = (hh, hv)
    with pytest.raises(ManualDownloadRequired, match="S1SLC_CVDL.rar"):
        SLC.fetch_s1slc_cvdl(tmp_path / "missing", cache_dir=tmp_path / "c")
    z = _zip_tree(src, tmp_path / "S1SLC_CVDL.zip", top="S1SLC_CVDL")
    b = SLC.fetch_s1slc_cvdl(tmp_path / "missing", cache_dir=tmp_path / "c", archive=z)
    assert len(b.meta) == 36 and set(b.meta["label"]) == {"1", "2", "3"}
    z_ = SLC._open(b.root)
    hh, hv = data["Sao Paulo"]
    x = np.asarray(z_["Sao Paulo"]["data"][:])
    assert x.dtype == np.float32 and x.shape == (12, 5, 5, 4)
    np.testing.assert_array_equal(x, np.stack([hh.real, hh.imag, hv.real, hv.imag], -1))
    assert SLC.fetch_s1slc_cvdl(tmp_path / "missing", cache_dir=tmp_path / "c").root == b.root


def test_mstar_manual_then_mirror_zip(mstar_dir, tmp_path):  # noqa: F811
    with pytest.raises(ManualDownloadRequired, match="omae11/MSTAR-dataset"):
        MST.fetch_mstar(tmp_path / "missing", cache_dir=tmp_path / "c")
    z = _zip_tree(mstar_dir, tmp_path / "mirror.zip",
                  top="MSTAR-dataset-d99007290d20ce52933c8bf0f8c3f461acd522ee")
    for source in ("jpg", "complex"):
        b = MST.fetch_mstar(tmp_path / "missing", cache_dir=tmp_path / "c", archive=z, source=source)
        ref = MST.fetch_mstar(mstar_dir, source=source)
        assert b.meta.equals(ref.meta)


def _atr_volumes(atr_root, tmp_path, volume=20_000):
    """7z of the SOC_40classes protocol cut in volumes (plain byte split, as 7-Zip does)."""
    py7zr = pytest.importorskip("py7zr")
    pytest.importorskip("multivolumefile")
    whole = tmp_path / "SOC_40classes.7z"
    with py7zr.SevenZipFile(whole, "w") as z:
        z.writeall(atr_root / "Slant_Range" / "SOC_40classes", arcname="SOC_40classes")
    data = whole.read_bytes()
    (tmp_path / "vols").mkdir()
    vols = []
    for k in range(0, len(data), volume):
        vols.append(tmp_path / "vols" / f"SOC_40classes.7z.{k // volume + 1:03d}")
        vols[-1].write_bytes(data[k:k + volume])
    return vols


def test_atrnet_star_gated_download_of_one_protocol(atr_dir, tmp_path, auth_net, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sources.shutil, "which", lambda name: None)  # py7zr path
    vols = _atr_volumes(atr_dir, tmp_path)
    with pytest.raises(MissingCredentialsError, match="huggingface"):
        ATR.fetch_atrnet_star(tmp_path / "missing", cache_dir=tmp_path / "c")
    serve, seen = auth_net
    name = "Slant_Range/Complex_Float_Quad/SOC_40classes.7z"
    for v in vols:
        serve(f"https://huggingface.co/datasets/{ATR.HF_REPO}/resolve/main/{name}{v.suffix}",
              v.read_bytes())
    monkeypatch.setitem(ATR.HF_ARCHIVES, ("Slant_Range", "SOC_40classes"),
                        (name, tuple(v.stat().st_size for v in vols)))
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    b = ATR.fetch_atrnet_star(tmp_path / "missing", cache_dir=tmp_path / "c")
    assert len(seen) == len(vols) >= 2 and {a for _, a in seen} == {"Bearer hf_secret"}
    ref = ATR.fetch_atrnet_star(atr_dir, cache_dir=tmp_path / "ref")
    assert b.meta.equals(ref.meta)
    assert (b.root / "Slant_Range" / "SOC_40classes").is_symlink()
    ATR.fetch_atrnet_star(tmp_path / "missing", cache_dir=tmp_path / "c")
    assert len(seen) == len(vols)  # idempotent


def test_atrnet_star_quotes_the_amplitude_file_name_and_checks_sizes(tmp_path, auth_net,
                                                                     monkeypatch):
    serve, seen = auth_net
    name, sizes = ATR.HF_ARCHIVES[("Amplitude_8bit", None)]
    url = (f"https://huggingface.co/datasets/{ATR.HF_REPO}/resolve/main/"
           f"{urllib.parse.quote(name)}.001")
    assert " " not in url and url.isascii()
    serve(url, b"not the expected size")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    with pytest.raises(OSError, match=f"expected {sizes[0]}"):
        ATR.fetch_atrnet_star(tmp_path / "missing", cache_dir=tmp_path / "c",
                              representation="Amplitude_8bit")
    assert seen == [(url, "Bearer hf_secret")]


def test_atrnet_star_archive_first_volume_or_local_layout(atr_dir, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sources.shutil, "which", lambda name: None)
    vols = _atr_volumes(atr_dir, tmp_path)
    ref = ATR.fetch_atrnet_star(atr_dir, cache_dir=tmp_path / "ref")
    b = ATR.fetch_atrnet_star(tmp_path / "missing", cache_dir=tmp_path / "c", archive=vols[0])
    assert b.meta.equals(ref.meta)
    b2 = ATR.fetch_atrnet_star(tmp_path / "missing", cache_dir=tmp_path / "c2", archive=atr_dir,
                               split="EOC_depression")
    assert b2.root == atr_dir


# --- dataset classes without local copy: the fetch fills the cache ------------------------------

def test_hsi_classes_build_from_the_fetch_without_local_copy(rices_dir, fruit_dir, tmp_path,  # noqa: F811
                                                             fake_net, monkeypatch):
    monkeypatch.setenv("SPDNET_DATA_ROOT", str(tmp_path / "empty_root"))
    serve, calls = fake_net
    z = _zip_tree(rices_dir, tmp_path / RI.ZIP, top="rices")
    serve(RI.URL, z.read_bytes())
    monkeypatch.setattr(RI, "ZIP_MD5", _md5(z))
    monkeypatch.setattr(RI, "ZIP_SIZE", z.stat().st_size)
    cache = tmp_path / "c"
    ds = RI.Rices90Dataset(cache_dir=cache, verbose=False, return_raw=True)
    (cov, raw), _ = ds[0]
    assert calls == [RI.URL] and raw.ndim == 2 and ds.data_dir.is_relative_to(cache)
    RI.Rices90Dataset(cache_dir=cache, verbose=False)  # cache reused, no new request
    assert calls == [RI.URL]
    # manual source: ManualDownloadRequired, then the cache filled by an archive is reused
    with pytest.raises(ManualDownloadRequired):
        DHS.DeepHSFruitDataset(cache_dir=cache, verbose=False)
    DHS.fetch_deephsfruit(cache_dir=cache, archive=_zip_tree(fruit_dir, tmp_path / "f.zip"))
    ds = DHS.DeepHSFruitDataset(cache_dir=cache, verbose=False)
    assert len(ds) > 0 and (ds.data_dir / "cov_VIS").is_dir()


def test_sar_classes_build_from_the_fetch_without_local_copy(fusar_dir, mstar_dir, tmp_path,  # noqa: F811
                                                             fake_net, monkeypatch):
    monkeypatch.setenv("SPDNET_DATA_ROOT", str(tmp_path / "empty_root"))
    serve, calls = fake_net
    src = tmp_path / "fusar_src"
    for name in ("Cargo", "Fishing"):
        shutil.copytree(fusar_dir / name, src / name)
    shutil.copy(fusar_dir / "meta.csv", src / "meta.csv")
    z = _zip_tree(src, tmp_path / FUS.ZIP, top="")
    serve(FUS.URL, z.read_bytes())
    monkeypatch.setattr(FUS, "ZIP_SHA256", _sha(z))
    monkeypatch.setattr(FUS, "ZIP_SIZE", z.stat().st_size)
    cache = tmp_path / "c"
    ds = FUS.FUSARShipDataset(cache_dir=cache, image_size=16, return_raw=True, verbose=False)
    assert calls == [FUS.URL] and ds[0][0].shape == (16, 16, 1)
    assert ds.data_dir.is_relative_to(cache)
    # manual source: ManualDownloadRequired, then the cache filled by an archive is reused
    with pytest.raises(ManualDownloadRequired):
        MST.MSTARDataset(cache_dir=cache, return_raw=True, verbose=False)
    MST.fetch_mstar(cache_dir=cache, archive=_zip_tree(mstar_dir, tmp_path / "m.zip"))
    ds = MST.MSTARDataset(cache_dir=cache, crop=16, return_raw=True, verbose=False)
    assert len(ds) == 30 and ds[0][0].shape == (16, 16, 1) and ds.data_dir.is_relative_to(cache)
