"""Access to dataset sources that need credentials or a manual download.

Three kinds of sources are handled by the ``fetch_<dataset>()`` functions:

- **direct**: public HTTP files (Zenodo, GitHub, public Hugging Face
  repositories), see :func:`spdnet_datasets.core.fetch.download`;
- **credentials**: Kaggle and gated Hugging Face repositories,
  :func:`kaggle_download` and :func:`hf_download`;
- **manual**: pages behind a login, a form or a quota (IEEE DataPort, ROSE
  Lab, Baidu...): :func:`manual_source` opens the page and explains what to
  download, then the fetch is called again with ``archive=<downloaded file>``.

Credentials are never written in code, URLs or logs. They are looked up in
this order:

1. environment variables (``KAGGLE_API_TOKEN``, or the legacy
   ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``; ``HF_TOKEN``);
2. the tool's own file (``~/.kaggle/access_token``, or the legacy
   ``~/.kaggle/kaggle.json``; the Hugging Face token cache
   ``~/.cache/huggingface/token``);
3. the system keyring, if the optional ``keyring`` package is installed
   (:func:`store_secret` saves a value there);
4. an interactive prompt (hidden input), when a terminal is attached.

Otherwise :class:`MissingCredentialsError` explains how to provide them.
"""

import base64
import getpass
import json
import os
import shutil
import stat
import sys
import tarfile
import warnings
import webbrowser
import zipfile
from pathlib import Path

from .fetch import _hash_file, download

KEYRING_SERVICE = "spdnet-datasets"


class MissingCredentialsError(RuntimeError):
    """Raised when a source needs credentials that were not provided."""


class ManualDownloadRequired(RuntimeError):
    """Raised when a dataset must first be downloaded by hand (see the message)."""


def _interactive(interactive):
    return sys.stdin.isatty() if interactive is None else interactive


def _keyring_get(name):
    try:
        import keyring
    except ImportError:
        return None
    try:
        return keyring.get_password(KEYRING_SERVICE, name)
    except Exception:  # pragma: no cover - backend errors (no keyring daemon...)
        return None


def store_secret(name: str, value: str) -> None:
    """Save a secret in the system keyring (needs the ``keyring`` package)."""
    try:
        import keyring
    except ImportError as exc:
        raise ImportError("store_secret needs the 'keyring' package") from exc
    keyring.set_password(KEYRING_SERVICE, name, value)


def get_secret(name: str, *, env=(), files=(), prompt=None, interactive=None, help_text=""):
    """Return a secret from the environment, files, the keyring or a hidden prompt.

    Parameters
    ----------
    name : str
        Keyring entry (service ``spdnet-datasets``).
    env : iterable of str
        Environment variables tried first.
    files : iterable of path
        Files whose stripped content is the secret.
    prompt : str, optional
        Hidden prompt shown when a terminal is attached.
    interactive : bool, optional
        Force or forbid the prompt (default: when stdin is a terminal).
    help_text : str
        Appended to the error message.
    """
    for var in env:
        if os.environ.get(var):
            return os.environ[var]
    for f in files:
        f = Path(f).expanduser()
        if f.is_file() and f.read_text().strip():
            return f.read_text().strip()
    value = _keyring_get(name)
    if value:
        return value
    if prompt and _interactive(interactive):
        value = getpass.getpass(prompt)
        if value:
            return value
    where = [f"set {' or '.join(env)}"] if env else []
    where += [f"create {f}" for f in files]
    where.append("store it with spdnet_datasets.core.access.store_secret")
    raise MissingCredentialsError(
        f"No credentials for {name!r}: {', or '.join(where)}. {help_text}".strip())


# --- Kaggle -------------------------------------------------------------------------------

def _kaggle_dir():
    return Path(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle")).expanduser()


def _check_private(path):
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        warnings.warn(f"{path} is readable by other users; run: chmod 600 {path}",
                      UserWarning, stacklevel=3)


def kaggle_token():
    """Return a Kaggle API access token (current format), or ``None``.

    Looks at ``KAGGLE_API_TOKEN``, then ``$KAGGLE_CONFIG_DIR/access_token`` or
    ``~/.kaggle/access_token`` (warns if other users can read it), then the
    keyring entry ``kaggle_token``. The token is created on
    https://www.kaggle.com/settings/api ("Generate New Token").
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return os.environ["KAGGLE_API_TOKEN"].strip()
    f = _kaggle_dir() / "access_token"
    if f.is_file() and f.read_text().strip():
        _check_private(f)
        return f.read_text().strip()
    return _keyring_get("kaggle_token")


def kaggle_credentials(interactive=None):
    """Return ``(username, key)`` of the legacy Kaggle API key format.

    Looks at ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``, then ``$KAGGLE_CONFIG_DIR/kaggle.json``
    or ``~/.kaggle/kaggle.json`` (warns if other users can read it), then the
    keyring entry ``kaggle`` (JSON ``{"username", "key"}``), then a prompt.
    Prefer the access token of :func:`kaggle_token`.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]
    cfg = _kaggle_dir() / "kaggle.json"
    if cfg.is_file():
        _check_private(cfg)
        data = json.loads(cfg.read_text())
        return data["username"], data["key"]
    stored = _keyring_get("kaggle")
    if stored:
        data = json.loads(stored)
        return data["username"], data["key"]
    if _interactive(interactive):
        user = input("Kaggle username: ").strip()
        key = getpass.getpass("Kaggle API key (hidden): ").strip()
        if user and key:
            return user, key
    raise MissingCredentialsError(_KAGGLE_HELP)


_KAGGLE_HELP = (
    "Kaggle credentials not found. Generate an access token on "
    "https://www.kaggle.com/settings/api ('Generate New Token') and store it in "
    "~/.kaggle/access_token (chmod 600), or set KAGGLE_API_TOKEN. The legacy "
    "kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY) is also accepted. You must also "
    "accept the dataset or competition rules on its page.")


def kaggle_auth_header(interactive=None) -> str:
    """``Authorization`` header value for the Kaggle API.

    ``Bearer <access token>`` (:func:`kaggle_token`), else ``Basic`` with the
    legacy username and key (:func:`kaggle_credentials`), else a hidden prompt
    for an access token.
    """
    token = kaggle_token()
    if token:
        return f"Bearer {token}"
    try:
        user, key = kaggle_credentials(interactive=False)
    except MissingCredentialsError:
        user = key = None
    if user and key:
        return "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()
    if _interactive(interactive):
        token = getpass.getpass("Kaggle API access token (hidden): ").strip()
        if token:
            return f"Bearer {token}"
    raise MissingCredentialsError(_KAGGLE_HELP)


def kaggle_download(ref: str, dest_dir, *, competition=False, checksum=None, algo="sha256",
                    interactive=None) -> Path:
    """Download a Kaggle dataset (``owner/name``) or competition archive as one zip.

    The rules of the dataset or competition must have been accepted on
    kaggle.com with the same account.
    """
    auth = kaggle_auth_header(interactive)
    if competition:
        url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{ref}"
    else:
        url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}"
    dest = Path(dest_dir) / (ref.replace("/", "__") + ".zip")
    return download(url, dest, checksum=checksum, algo=algo, headers={"Authorization": auth})


# --- Hugging Face -------------------------------------------------------------------------

def hf_token(interactive=None, required=True):
    """Return a Hugging Face token (``HF_TOKEN``, token cache, keyring or prompt)."""
    home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    try:
        return get_secret(
            "huggingface", env=("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"), files=(home / "token",),
            prompt="Hugging Face token (hidden): " if required else None,
            interactive=interactive,
            help_text="Create a read token on huggingface.co (Settings > Access Tokens) and "
                      "accept the dataset conditions on its page.")
    except MissingCredentialsError:
        if required:
            raise
        return None


def hf_download(repo_id: str, filename: str, dest_dir, *, revision="main", repo_type="dataset",
                gated=False, checksum=None, algo="sha256", interactive=None) -> Path:
    """Download one file of a Hugging Face repository (with a token if ``gated``)."""
    prefix = "datasets/" if repo_type == "dataset" else ""
    url = f"https://huggingface.co/{prefix}{repo_id}/resolve/{revision}/{filename}"
    token = hf_token(interactive, required=gated)
    headers = {"Authorization": f"Bearer {token}"} if token else None
    return download(url, Path(dest_dir) / filename, checksum=checksum, algo=algo,
                    headers=headers)


# --- Manual downloads -----------------------------------------------------------------------

def _can_open_browser():
    if os.environ.get("SPDNET_OPEN_BROWSER", "1") == "0":
        return False
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY")
                                                 or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return True


def _safe_members_tar(tar, dest):
    dest = dest.resolve()
    for m in tar.getmembers():
        target = (dest / m.name).resolve()
        if dest not in target.parents and target != dest:
            raise OSError(f"Unsafe path in archive: {m.name}")
        if m.issym() or m.islnk():
            link = (target.parent / m.linkname).resolve()
            if dest not in link.parents:
                raise OSError(f"Unsafe link in archive: {m.name} -> {m.linkname}")
    return tar.getmembers()


def extract_archive(archive, dest_dir) -> Path:
    """Extract a zip / tar archive into ``dest_dir`` once (``.extracted`` marker).

    Paths escaping ``dest_dir`` (``../``, absolute paths, outside links) are
    rejected. Extraction goes to ``dest_dir.part`` and is renamed when complete.
    """
    archive, dest = Path(archive), Path(dest_dir)
    if (dest / ".extracted").exists():
        return dest
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                target = (tmp / name).resolve()
                if tmp.resolve() not in target.parents and target != tmp.resolve():
                    raise OSError(f"Unsafe path in archive: {name}")
            z.extractall(tmp)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(tmp, members=_safe_members_tar(t, tmp))
    else:
        raise OSError(f"{archive} is neither a zip nor a tar archive")
    (tmp / ".extracted").write_text(str(archive))
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return dest


def manual_source(name: str, *, page_url: str, expected: str, dest_dir, archive=None,
                  checksum=None, algo="sha256", size=None, open_browser=None) -> Path:
    """Use a dataset that must be downloaded by hand.

    Without ``archive``: opens ``page_url`` in a browser (when possible) and
    raises :class:`ManualDownloadRequired` with the instructions. With
    ``archive`` (path to the downloaded file, or to an already extracted
    directory): checks it (size, checksum) and extracts it into ``dest_dir``;
    a directory is used in place (read only).
    """
    if archive is None:
        if (_can_open_browser() if open_browser is None else open_browser):
            try:
                webbrowser.open(page_url)
            except Exception:  # pragma: no cover - no browser available
                pass
        raise ManualDownloadRequired(
            f"{name} has to be downloaded by hand.\n"
            f"  1. Open {page_url}\n"
            f"  2. Download: {expected}\n"
            f"  3. Call the fetch again with archive='<path to the downloaded file or folder>'.")
    archive = Path(archive).expanduser()
    if not archive.exists():
        raise FileNotFoundError(f"{archive} does not exist")
    if archive.is_dir():
        return archive
    if size is not None and archive.stat().st_size != size:
        raise OSError(f"{archive}: size {archive.stat().st_size} != expected {size}")
    if checksum and _hash_file(archive, algo) != checksum:
        raise OSError(f"Checksum mismatch for {archive}")
    return extract_archive(archive, dest_dir)
