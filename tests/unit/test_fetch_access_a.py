"""Online access of the audio, skeleton, micro-Doppler and SITS datasets (simulated network).

Each test serves tiny synthetic sources through the ``fake_net`` fixture (fake
``urlopen``) or gives them with ``archive=``, and checks the resolution order of
the fetch: local copy, then ``archive``, then download (checksum, idempotence),
then :class:`~spdnet_datasets.core.access.ManualDownloadRequired`.
"""

import base64
import hashlib
import io
import json
import sys
import tarfile
import types
import urllib.parse
import zipfile

import numpy as np
import pytest

from spdnet_datasets.core.access import ManualDownloadRequired

pytest.importorskip("pyarrow")
pytest.importorskip("soundfile")

from tests.unit.test_audio_datasets import _wav, esc50_dir, gtzan_dir  # noqa: E402,F401
from tests.unit.test_radar_ci4r import P as CI4R_P  # noqa: E402
from tests.unit.test_radar_ci4r import make_ci4r  # noqa: E402
from tests.unit.test_radar_dopnet import P as DOP_P  # noqa: E402
from tests.unit.test_radar_dopnet import make_dopnet  # noqa: E402
from tests.unit.test_radar_mvdoppler import make_mvdoppler  # noqa: E402
from tests.unit.test_sits_pastis_r import REF, pastis_dir  # noqa: E402,F401
from tests.unit.test_skeleton_datasets import hdm05_dir  # noqa: E402,F401

pytestmark = [pytest.mark.filterwarnings("ignore:HDM05 has no actor:UserWarning"),
              pytest.mark.filterwarnings("ignore:BirdSet XCM:UserWarning")]


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    monkeypatch.setenv("SPDNET_OPEN_BROWSER", "0")


def _zip(folder, dest, prefix=None):
    """Zip every file under ``folder`` (paths relative to ``folder.parent`` or ``prefix``)."""
    base = folder.parent if prefix is None else folder
    with zipfile.ZipFile(dest, "w") as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, (prefix or "") + str(f.relative_to(base)))
    return dest


# ---------------------------------------------------------------------- ESC-50, GTZAN
def test_esc50_archive_is_ingested_without_network(esc50_dir, tmp_path, fake_net,  # noqa: F811
                                                   monkeypatch):
    from spdnet_datasets.audio import esc50

    _, calls = fake_net
    ref = esc50.fetch_esc50(esc50_dir, cache_dir=tmp_path / "local")
    archive = _zip(esc50_dir, tmp_path / "esc50.zip")
    b = esc50.fetch_esc50(tmp_path / "absent", cache_dir=tmp_path / "c", archive=archive)
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert calls == [] and archive.exists()
    assert (tmp_path / "c" / "esc50" / "downloads" / "archive" / "esc50.zip" / ".extracted").exists()
    # a shard named like a registered one is checked
    bad = {"https://example.org/esc50_0.parquet": "0" * 64}
    monkeypatch.setattr(esc50, "FILES", bad)
    with pytest.raises(OSError, match="Checksum"):
        esc50.fetch_esc50(tmp_path / "absent", cache_dir=tmp_path / "c2", archive=esc50_dir)


def test_gtzan_archive_folder_is_read_in_place(gtzan_dir, tmp_path, fake_net):  # noqa: F811
    from spdnet_datasets.audio import gtzan

    _, calls = fake_net
    b = gtzan.fetch_gtzan(tmp_path / "absent", cache_dir=tmp_path / "c", archive=gtzan_dir)
    ref = gtzan.fetch_gtzan(gtzan_dir, cache_dir=tmp_path / "local")
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert calls == [] and not (tmp_path / "c" / "gtzan" / "downloads").exists()


# ------------------------------------------------------------------------ BirdSet XCM
def _birdset_sources(tmp_path):
    """Two shards (tar.gz of WAV bytes named .ogg) and the metadata parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = tmp_path / "hub"
    src.mkdir()
    rows, shards = [], {}
    for k, species in ((1, ("sp_a", "sp_b")), (2, ("sp_c",))):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for sp in species:
                for i in range(12):
                    path = f"data/xeno-canto/europe/{sp}/XC{k}{sp[-1]}{i:02d}.ogg"
                    blob = _wav(len(rows), seconds=1.5)
                    info = tarfile.TarInfo(path)
                    info.size = len(blob)
                    tar.addfile(info, io.BytesIO(blob))
                    rows.append({"filepath": path, "ebird_code": sp, "length": 1,
                                 "recordist": f"rec{i % 6}" if i else None,
                                 "detected_events": [[0.2, 0.6]]})
        shards[k] = buf.getvalue()
        (src / f"XCM_shard_{k:04d}.tar.gz").write_bytes(shards[k])
    rows.append({"filepath": "data/xeno-canto/europe/sp_z/XC999.ogg", "ebird_code": "sp_z",
                 "length": 1, "recordist": "x", "detected_events": []})  # in no shard
    pq.write_table(pa.Table.from_pylist(rows), src / "XCM_metadata.parquet")
    return src, shards


def _serve_birdset(monkeypatch, bx, src, shards, serve):
    meta_url = bx._HF + bx.METADATA
    monkeypatch.setattr(bx, "FILES", {
        meta_url: serve(meta_url, (src / bx.METADATA).read_bytes())})
    sums = list(bx.SHARD_SHA256)
    for k, blob in shards.items():
        sums[k - 1] = serve(bx._HF + f"XCM_shard_{k:04d}.tar.gz", blob)
    monkeypatch.setattr(bx, "SHARD_SHA256", tuple(sums))


def test_birdset_select_shards_spreads_counts():
    from spdnet_datasets.audio.birdset_xcm import select_shards

    assert select_shards(4) == [1, 61, 122, 182] and select_shards([5, 2, 5]) == [2, 5]
    assert len(select_shards("all")) == 182
    for bad in (0, 183, [0], "some"):
        with pytest.raises(ValueError):
            select_shards(bad)


def test_birdset_shards_download_checksum_idempotence(tmp_path, fake_net, monkeypatch):
    from spdnet_datasets.audio import birdset_xcm as bx

    serve, calls = fake_net
    src, shards = _birdset_sources(tmp_path)
    _serve_birdset(monkeypatch, bx, src, shards, serve)
    absent, cache = tmp_path / "absent", tmp_path / "cache"
    with pytest.raises(ValueError, match="shards="):
        bx.fetch_birdset_xcm(absent, cache_dir=cache)
    with pytest.warns(UserWarning, match="2 of 182 shards"):
        b = bx.fetch_birdset_xcm(absent, cache_dir=cache, shards=[1, 2])
    assert len(calls) == 3
    assert sorted(set(b.labels)) == ["sp_a", "sp_b", "sp_c"] and len(b.labels) == 36
    assert "NA" in set(b.groups)
    downloads = cache / "birdset_xcm" / "downloads"
    assert (downloads / "XCM_shard_0001" / ".extracted").exists()
    assert not list(downloads.glob("*.tar.gz"))  # archives deleted once extracted
    bx.fetch_birdset_xcm(absent, cache_dir=cache, shards=[1, 2], n_mfcc=13)
    one = bx.fetch_birdset_xcm(absent, cache_dir=cache, shards=[2], min_recordings=1)
    assert len(calls) == 3 and set(one.labels) == {"sp_c"}

    # archive mode: the same files given by hand, never deleted, no network
    calls.clear()
    c = bx.fetch_birdset_xcm(absent, cache_dir=tmp_path / "c2", archive=src)
    np.testing.assert_allclose(c.covariances, b.covariances)
    assert calls == [] and len(list(src.glob("*.tar.gz"))) == 2


def test_birdset_corrupted_shard_is_refused(tmp_path, fake_net, monkeypatch):
    from spdnet_datasets.audio import birdset_xcm as bx

    serve, _ = fake_net
    src, shards = _birdset_sources(tmp_path)
    _serve_birdset(monkeypatch, bx, src, shards, serve)
    (src / "XCM_shard_0002.tar.gz").write_bytes(b"corrupted")
    with pytest.raises(OSError, match="Checksum"):
        bx.fetch_birdset_xcm(tmp_path / "absent", cache_dir=tmp_path / "c", archive=src)


# ------------------------------------------------------------------------------ HDM05
def _eth_zip(path):
    """Tiny ``HDM05_SPDData.zip``: 3 actions x 2 sequences x 3 segments of 6 x 6."""
    from scipy.io import savemat

    rng = np.random.default_rng(0)
    root = path.parent / "eth_src" / "HDM05_SPDData"
    names, sets = [], []
    for action in ("clap", "jump", "kick"):
        (root / "feature" / action).mkdir(parents=True)
        for seq in (1, 2, 3, 4, 5):
            for seg in (1, 2):
                A = rng.standard_normal((6, 12))
                savemat(root / "feature" / action / f"{seq}_{seg}.mat",
                        {"Y1": (A @ A.T / 12).astype(np.float32)})
                names.append(f"spd93_half_inter9\\{action}\\{seq}_{seg}.mat")
                sets.append(1 + seg % 2)
    spd = {"id": np.arange(len(names) - 1), "name": np.array(names[:-1], dtype=object),
           "set": np.array(sets[:-1]), "label": np.zeros(len(names) - 1)}
    savemat(root / "metadata.mat", {"spdDir": "C:\\x", "spd": spd})
    return _zip(root, path)


def test_hdm05_local_export_is_manual(hdm05_dir, tmp_path, fake_net):  # noqa: F811
    from spdnet_datasets.skeleton import hdm05

    _, calls = fake_net
    with pytest.raises(ManualDownloadRequired, match="HDM05"):
        hdm05.fetch_hdm05(tmp_path / "absent", cache_dir=tmp_path / "c")
    archive = _zip(hdm05_dir, tmp_path / "hdm05.zip")
    b = hdm05.fetch_hdm05(tmp_path / "absent", cache_dir=tmp_path / "c", archive=archive)
    ref = hdm05.fetch_hdm05(hdm05_dir, cache_dir=tmp_path / "local")
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert calls == []


def test_hdm05_eth_download_conversion_and_idempotence(tmp_path, fake_net, monkeypatch):
    from scipy.io import loadmat

    from spdnet_datasets.skeleton import hdm05

    serve, calls = fake_net
    archive = _eth_zip(tmp_path / "HDM05_SPDData.zip")
    monkeypatch.setattr(hdm05, "FILES", {
        hdm05.ETH_URL: serve(hdm05.ETH_URL, archive.read_bytes())})
    monkeypatch.setattr(hdm05, "ETH_SIZE", archive.stat().st_size)
    cache = tmp_path / "cache"
    b = hdm05.fetch_hdm05(source="eth", cache_dir=cache)
    assert calls == [hdm05.ETH_URL] and b.covariances.shape == (30, 6, 6)
    assert not (cache / "hdm05" / "downloads" / "HDM05_SPDData.zip").exists()
    # the matrices are those of the .mat files (float32 -> float64), labels = folders
    row = list(b.meta["id"]).index("jump/3_2")
    mat = loadmat(cache / "hdm05" / "downloads" / "eth" / "HDM05_SPDData" / "feature" / "jump"
                  / "3_2.mat")["Y1"].astype(np.float64)
    np.testing.assert_allclose(b.covariances[row], (mat + mat.T) / 2)
    assert set(b.labels) == {"clap", "jump", "kick"} and b.meta["group"][row] == "jump/3"
    assert b.meta["set"].isna().sum() == 1 and set(b.meta["set"].dropna()) == {1.0, 2.0}
    parts = b.splits["recommended"]
    assert not set(b.groups[parts["train"]]) & set(b.groups[parts["test"]])
    hdm05.fetch_hdm05(source="eth", cache_dir=cache, limit=10)  # new stage, no download
    assert len(calls) == 1
    with pytest.raises(ValueError, match="c93"):
        hdm05.fetch_hdm05(source="eth", cache_dir=cache, features="c31")
    # the zip given by hand: size and checksum checked, kept
    c = hdm05.fetch_hdm05(source="eth", cache_dir=tmp_path / "c2", archive=archive)
    np.testing.assert_allclose(c.covariances, b.covariances)
    assert archive.exists() and len(calls) == 1


# -------------------------------------------------------------------------- NTU RGB+D
def _write_skeleton(path, bodies_per_frame, rng):
    """Official text format; returns the expected ``skel_body<k>`` arrays."""
    nframes = len(bodies_per_frame)
    skel = np.zeros((max(bodies_per_frame), nframes, 25, 3))
    lines = [str(nframes)]
    for t, nb in enumerate(bodies_per_frame):
        lines.append(str(nb))
        for b in range(nb):
            lines.append(f"7205759403793{b} 0 1 1 0 0 0 -0.1 -0.1 2")
            lines.append("25")
            for j in range(25):
                xyz = [f"{v:.6f}" for v in rng.standard_normal(3)]
                skel[b, t, j] = [float(v) for v in xyz]
                rest = "217.2 221.3 863.0 604.1 -0.17 0.05 0.97 -0.1 2"
                lines.append(" ".join(xyz) + " " + rest)
    path.write_text("\n".join(lines) + "\n")
    return skel


def test_read_skeleton_matches_txt2npy_layout(tmp_path):
    from spdnet_datasets.skeleton.ntu120 import read_skeleton

    rng = np.random.default_rng(0)
    path = tmp_path / "S018C001P008R001A106.skeleton"
    skel = _write_skeleton(path, [0, 1, 2, 2, 1, 0], rng)
    rec = read_skeleton(path)
    assert rec["file_name"] == path.stem and rec["nframes"] == 6 and rec["njoints"] == 25
    assert rec["nbodys"] == [1, 2, 2, 1]  # empty frames are not counted
    assert sorted(k for k in rec if k.startswith("skel")) == ["skel_body0", "skel_body1"]
    np.testing.assert_array_equal(rec["skel_body0"], skel[0])
    np.testing.assert_array_equal(rec["skel_body1"], skel[1])
    assert not rec["skel_body0"][0].any()  # frame without a body stays zero
    empty = tmp_path / "S018C001P008R001A001.skeleton"
    _write_skeleton(empty, [0, 0, 0], rng)
    assert read_skeleton(empty) == {"file_name": empty.stem, "nbodys": [], "njoints": 25,
                                    "nframes": 3}


def test_ntu120_manual_then_skeleton_archives(tmp_path, fake_net):
    from spdnet_datasets.skeleton import ntu120

    _, calls = fake_net
    with pytest.raises(ManualDownloadRequired, match="rose1.ntu.edu.sg"):
        ntu120.fetch_ntu120(tmp_path / "absent", cache_dir=tmp_path / "c")
    rng = np.random.default_rng(1)
    zips, npy = [], tmp_path / "npy" / "NTU_RGBD_120"
    for part, setups in (("s001_to_s017", (1, 2)), ("s018_to_s032", (18, 19))):
        src = tmp_path / part / f"nturgbd_skeletons_{part}"
        src.mkdir(parents=True)
        for setup in setups:
            for subject in (1, 2, 4, 5, 8, 9):
                for action in (1, 50):
                    name = f"S{setup:03d}C001P{subject:03d}R001A{action:03d}"
                    frames = [0] + [1] * (35 + subject) if subject != 9 else [0] * 10
                    _write_skeleton(src / f"{name}.skeleton", frames, rng)
                    out = npy / "npy_skeletons_by_class" / f"A{action:03d}"
                    out.mkdir(parents=True, exist_ok=True)
                    np.save(out / f"{name}.npy", ntu120.read_skeleton(src / f"{name}.skeleton"),
                            allow_pickle=True)
        zips.append(_zip(src, tmp_path / f"nturgbd_skeletons_{part}.zip"))
    b = ntu120.fetch_ntu120(tmp_path / "absent", cache_dir=tmp_path / "c", archive=zips)
    ref = ntu120.fetch_ntu120(npy, cache_dir=tmp_path / "local", verify=None)
    assert len(b.labels) == 40 and sorted(b.excluded["reason"]) == ["no body"] * 8
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert list(b.meta["id"]) == list(ref.meta["id"]) and calls == []
    assert all(z.exists() for z in zips)
    folder = ntu120.fetch_ntu120(tmp_path / "absent", cache_dir=tmp_path / "c3",
                                 archive=tmp_path / "s018_to_s032")
    assert set(folder.meta["setup"]) == {18, 19}


# ------------------------------------------------------------------------------- CI4R
def _serve_ci4r(serve, root, mode_folder):
    from spdnet_datasets.radar_microdoppler import ci4r

    items = []
    for f in sorted((root / "11_class_activity_data" / mode_folder).rglob("*.png")):
        name = "Spectrograms/" + str(f.relative_to(root))
        blob = f.read_bytes()
        serve(f"https://storage.googleapis.com/{ci4r.BUCKET}/{urllib.parse.quote(name)}", blob)
        items.append({"name": name,
                      "md5Hash": base64.b64encode(hashlib.md5(blob).digest()).decode()})
    query = {"prefix": f"Spectrograms/11_class_activity_data/{mode_folder}/",
             "fields": "nextPageToken,items(name,md5Hash)"}
    half = len(items) // 2  # two pages
    first = f"{ci4r.LIST_API}?{urllib.parse.urlencode(query)}"
    serve(first, json.dumps({"items": items[:half], "nextPageToken": "p2"}).encode())
    serve(f"{ci4r.LIST_API}?{urllib.parse.urlencode({**query, 'pageToken': 'p2'})}",
          json.dumps({"items": items[half:]}).encode())
    return len(items)


def test_ci4r_bucket_download_md5_and_idempotence(tmp_path, fake_net, monkeypatch):
    from spdnet_datasets.radar_microdoppler import ci4r

    serve, calls = fake_net
    local = make_ci4r(tmp_path / "src" / "CI4R-MULTI3")
    n = _serve_ci4r(serve, local, "77GHz")
    monkeypatch.setattr(ci4r, "N_IMAGES", n)
    kw = {"p": CI4R_P, "band": 0.5, "verify": None}
    b = ci4r.fetch_ci4r(tmp_path / "absent", tmp_path / "cache", **kw)
    ref = ci4r.fetch_ci4r(local, tmp_path / "local", **kw)
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert len(calls) == n + 2  # two listing pages + every image
    assert any("%20%28copy%29.png" in u for u in calls)  # names are URL-quoted
    ci4r.fetch_ci4r(tmp_path / "absent", tmp_path / "cache", estimator="ledoit_wolf", **kw)
    assert len(calls) == n + 2  # marker: no new listing


def test_ci4r_bad_md5_or_count_is_refused(tmp_path, fake_net, monkeypatch):
    from spdnet_datasets.radar_microdoppler import ci4r

    serve, _ = fake_net
    local = make_ci4r(tmp_path / "src" / "CI4R-MULTI3")
    n = _serve_ci4r(serve, local, "77GHz")
    with pytest.raises(OSError, match="expected 653"):
        ci4r.download_ci4r("77GHz", tmp_path / "c1")
    monkeypatch.setattr(ci4r, "N_IMAGES", n)
    first = sorted((local / "11_class_activity_data" / "77GHz").rglob("*.png"))[0]
    name = "Spectrograms/" + str(first.relative_to(local))
    serve(f"https://storage.googleapis.com/{ci4r.BUCKET}/{urllib.parse.quote(name)}", b"x")
    with pytest.raises(OSError, match="Checksum"):
        ci4r.download_ci4r("77GHz", tmp_path / "c2")


def test_ci4r_archive(tmp_path, fake_net):
    from spdnet_datasets.radar_microdoppler import ci4r

    _, calls = fake_net
    local = make_ci4r(tmp_path / "src" / "CI4R-MULTI3")
    archive = _zip(local, tmp_path / "ci4r.zip")
    ds = ci4r.CI4RDataset(tmp_path / "absent", cache_dir=tmp_path / "c", p=CI4R_P,
                          archive=archive, verify=None, verbose=False)
    ref = ci4r.fetch_ci4r(local, tmp_path / "local", p=CI4R_P, verify=None)
    assert len(ds) == len(ref.labels) and calls == []


# ---------------------------------------------------------------------------- Dop-NET
def test_dopnet_figshare_md5_extract_and_archive(tmp_path, fake_net, monkeypatch):
    from spdnet_datasets.radar_microdoppler import dopnet

    serve, calls = fake_net
    local = make_dopnet(tmp_path / "src" / "Gesture Data")
    archive = _zip(local, tmp_path / "Gesture Data.zip")
    blob = archive.read_bytes()
    serve(dopnet.ARCHIVE_URL, blob)
    monkeypatch.setattr(dopnet, "FILES", {dopnet.ARCHIVE_URL: hashlib.md5(blob).hexdigest()})
    monkeypatch.setattr(dopnet, "ARCHIVE_SIZE", len(blob))
    cache = tmp_path / "cache"
    b = dopnet.fetch_dopnet(tmp_path / "absent", cache, p=DOP_P, verify=None)
    ref = dopnet.fetch_dopnet(local, tmp_path / "local", p=DOP_P, verify=None)
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert calls == [dopnet.ARCHIVE_URL]
    assert not (cache / "dopnet" / "downloads" / "Gesture Data.zip").exists()
    dopnet.fetch_dopnet(tmp_path / "absent", cache, p=DOP_P, energy=0.9, verify=None)
    assert len(calls) == 1
    # archive: size and md5 checked, the user's zip is kept
    ds = dopnet.DopNetDataset(tmp_path / "absent", cache_dir=tmp_path / "c2", p=DOP_P,
                              archive=archive, verify=None, verbose=False)
    assert len(ds) == len(b.labels) and archive.exists() and len(calls) == 1
    monkeypatch.setattr(dopnet, "FILES", {dopnet.ARCHIVE_URL: "0" * 32})
    with pytest.raises(OSError, match="Checksum"):
        dopnet.fetch_dopnet(tmp_path / "absent", tmp_path / "c3", archive=archive)


# -------------------------------------------------------------------------- MVDoppler
def _serve_mv_labels(monkeypatch, mv, local, serve):
    files = {}
    for name, drive_id in mv.LABEL_FILES.items():
        path = local / "Labels_and_metadata" / name
        blob = path.read_bytes() if path.exists() else b"{}"
        files[mv._DRIVE + drive_id] = serve(mv._DRIVE + drive_id, blob)
    monkeypatch.setattr(mv, "FILES", files)


def test_mvdoppler_manual_snapshots_and_direct_labels(tmp_path, fake_net, monkeypatch):
    from spdnet_datasets.radar_microdoppler import mvdoppler as mv

    serve, calls = fake_net
    local, _ = make_mvdoppler(tmp_path / "src" / "MVDoppler")
    _serve_mv_labels(monkeypatch, mv, local, serve)
    with pytest.raises(ManualDownloadRequired, match="snapshots.zip"):
        mv.fetch_mvdoppler(tmp_path / "absent", tmp_path / "c")
    assert calls == []
    archive = _zip(local / "Data", tmp_path / "snapshots.zip", prefix="snapshots/")
    b = mv.fetch_mvdoppler(tmp_path / "absent", tmp_path / "c", archive=archive)
    ref = mv.fetch_mvdoppler(local, tmp_path / "local")
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert len(calls) == 2 and archive.exists()
    mv.fetch_mvdoppler(tmp_path / "absent", tmp_path / "c", estimator="ledoit_wolf")
    assert len(calls) == 2  # labels and snapshots already in the cache


def test_mvdoppler_optional_gdown(tmp_path, fake_net, monkeypatch):
    from spdnet_datasets.radar_microdoppler import mvdoppler as mv

    serve, _ = fake_net
    local, _ = make_mvdoppler(tmp_path / "src" / "MVDoppler")
    _serve_mv_labels(monkeypatch, mv, local, serve)
    blob = _zip(local / "Data", tmp_path / "s.zip").read_bytes()
    seen = []

    def fake_download(id=None, output=None, **kw):
        seen.append(id)
        open(output, "wb").write(blob)
        return output

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=fake_download))
    b = mv.fetch_mvdoppler(tmp_path / "absent", tmp_path / "c", gdown=True)
    assert seen == [mv.SNAPSHOTS_ID] and len(b.labels) > 0
    monkeypatch.setitem(sys.modules, "gdown", None)  # not installed
    with pytest.raises(ImportError, match="gdrive"):
        mv.fetch_mvdoppler(tmp_path / "absent", tmp_path / "c2", gdown=True)


# --------------------------------------------------------------------------- PASTIS-R
def test_pastis_r_archive_is_checked_and_kept(pastis_dir, tmp_path, fake_net,  # noqa: F811
                                              monkeypatch):
    from spdnet_datasets.sits import pastis_r as pr

    _, calls = fake_net
    archive = _zip(pastis_dir / "PASTIS-R", tmp_path / "PASTIS-R.zip")
    with pytest.raises(OSError, match="Checksum"):
        pr.fetch_pastis_r(tmp_path / "absent", tmp_path / "c", ref_date=REF, archive=archive)
    monkeypatch.setattr(pr, "FILES", {pr.ZENODO_URL: hashlib.sha256(
        archive.read_bytes()).hexdigest()})
    b = pr.fetch_pastis_r(tmp_path / "absent", tmp_path / "c", ref_date=REF, archive=archive)
    ref = pr.fetch_pastis_r(pastis_dir, tmp_path / "local", ref_date=REF)
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert calls == [] and archive.exists()
    assert (tmp_path / "c" / "pastis_r" / "downloads" / "archive" / "PASTIS-R").is_dir()
