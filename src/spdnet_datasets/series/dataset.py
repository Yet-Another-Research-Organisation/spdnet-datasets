"""``CovarianceSeries``: a series of covariance tokens per sample of a main loader.

The wrapped loader keeps its role: fetch, class selection, labels, groups and named splits all
come from it. The wrapper reads each sample in its structured form (``structured_raw``), cuts it
into tokens (:mod:`~spdnet_datasets.series.modes`), estimates one ``p x p`` covariance per token
and caches the unscaled result under ``<cache_dir>/series/<name>/`` through
:func:`spdnet_datasets.core.fetch.run_pipeline`.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..core.estimators import NotSPDError, check_spd, estimate_covariance
from ..core.fetch import params_hash, run_pipeline
from ..core.normalize import geometric_scale
from .modes import get_mode

__all__ = ["CovarianceSeries", "collate_series"]

SUBSETS = (None, "train", "val", "test")


def _registry_name(ds):
    from ..manager import DatasetManager

    for name, cls in DatasetManager.DATASETS.items():
        if type(ds) is cls and not name.endswith("_legacy"):
            return name
    raise ValueError(f"{type(ds).__name__} is not a registered dataset")


def _build_loader(name, mode, loader_kwargs):
    from .. import audio, hyperspectral, radar_microdoppler  # noqa: F401  (registration)
    from ..manager import DatasetManager

    if name not in DatasetManager.DATASETS:
        raise ValueError(f"Unknown dataset {name!r}")
    kwargs = {**mode.loader_defaults, **(loader_kwargs or {})}
    if kwargs.get("subset") not in (None, "all"):
        raise ValueError("the wrapped loader must expose every sample: pass subset to "
                         "CovarianceSeries, not to the loader")
    return DatasetManager.DATASETS[name](**kwargs)


def _check_loader(ds, mode):
    """The loader instance must match the mode's loader requirements and expose every sample."""
    if getattr(ds, "subset", None) is not None:
        raise ValueError("the wrapped loader was built with subset="
                         f"{ds.subset!r}: build it with subset=None and pass subset to "
                         "CovarianceSeries")
    settings = {**vars(ds), **getattr(ds, "feature_kwargs", {})}
    for key, value in mode.loader_defaults.items():
        if key in settings and settings[key] != value:
            raise ValueError(f"mode {mode.name!r} needs a loader with {key}={value!r}, got "
                             f"{settings[key]!r}")


def _source_key(ds):
    """Identity of the wrapped samples: loader class, cached stage and exposed rows."""
    bunch = getattr(ds, "bunch", None) or getattr(ds, "_bunch", None) or {}
    stage = (bunch.get("covariance_path") or bunch.get("processed_path")
             or getattr(ds, "_processed", None))
    index = np.asarray(getattr(ds, "_index", np.arange(len(ds))))
    labels = np.asarray(ds.labels)
    return {"loader": type(ds).__name__, "data_dir": str(getattr(ds, "data_dir", "")),
            "stage": str(stage), "index": params_hash({"i": index.tolist()}),
            "labels": params_hash({"y": labels.tolist()}), "n": len(ds)}


def _estimate(obs, estimator, min_n):
    """``(covariance or None, mean, n)`` of one token; ``None`` when the token is degenerate."""
    X = np.asarray(obs, dtype=np.float64)
    n, p = X.shape
    mean = X.mean(axis=0) if n else np.zeros(p)
    if n < min_n or not np.all(np.isfinite(X)):
        return None, mean, n
    C = estimate_covariance(X, estimator, check=False)
    try:
        check_spd(C)
    except NotSPDError:
        return None, mean, n
    return C, mean, n


def _valid(X):
    return X[np.isfinite(X).all(axis=1) & (X != 0).any(axis=1)]


class CovarianceSeries(Dataset):
    """Covariance series built from a main loader.

    ``series[i] -> ((tokens, mask[, mean]), label)``:

    - ``tokens``: ``(S_i, p, p)`` float64, one SPD matrix per token, multiplied by the scale
      ``s`` fitted on the training tokens; masked tokens are identity matrices;
    - ``mask``: ``(S_i,)`` bool, ``False`` for a masked token (too few observations, constant,
      not SPD);
    - ``mean`` (``return_mean=True``): ``(S_i, p)`` float64 token means times ``sqrt(s)``
      (zero for a masked token).

    ``S_i`` varies from sample to sample when the sample length does (audio clips, Dop-NET
    gestures): use :func:`collate_series`, which pads with masked identity tokens.

    Parameters
    ----------
    dataset : str or loader instance
        Registered name (``"esc50"``, ``"hyperleaf"``, ...), or a loader built with
        ``subset=None``.
    mode : str, optional
        Mode of :data:`~spdnet_datasets.series.modes.SERIES_MODES` (default: the dataset's
        first mode).
    split : str, optional
        Split of the loader whose training part fits the scale (default: the loader's
        ``split``).
    subset : {None, "train", "val", "test"}
        Part of ``split`` exposed (``None``: every sample, and :attr:`splits` indexes them).
    loader_kwargs : dict, optional
        Arguments of the loader when ``dataset`` is a name (``data_dir``, ``cache_dir``,
        ``fold``, ``band_step``...); they update the mode's ``loader_defaults``.
    estimator : {"scm", "ledoit_wolf", "tyler", "student"}, optional
        Default: the mode's estimator.
    allow_small_n : bool, optional
        Accept tokens with fewer than ``2p`` observations; they are then estimated with
        Ledoit-Wolf, whatever ``estimator``. Default: the mode's (``False`` except for the
        documented large-p derogations).
    return_mean : bool
    normalize : bool
        Multiply the tokens by the geometric scale fitted on the unmasked training tokens.
    cache_dir, force
        Series cache (default ``SPDNET_CACHE_DIR``), recompute.
    verbose : bool
    **params
        Tokenizer parameters of the mode (every default can be overridden).

    Raises
    ------
    ValueError
        When the parameters give tokens with fewer than ``2p`` observations on a typical
        sample (unless ``allow_small_n``), or an unknown dataset, mode or parameter.
    """

    def __init__(self, dataset, mode=None, *, split=None, subset=None, loader_kwargs=None,
                 estimator=None, allow_small_n=None, return_mean=False, normalize=True,
                 cache_dir=None, force=False, verbose=True, **params):
        if subset == "all":
            subset = None
        if subset not in SUBSETS:
            raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
        if isinstance(dataset, str):
            self.name = dataset
            self.mode = get_mode(dataset, mode)
            self.loader = _build_loader(dataset, self.mode, loader_kwargs)
        else:
            if loader_kwargs:
                raise ValueError("loader_kwargs only apply when dataset is a name")
            self.name = _registry_name(dataset)
            self.mode = get_mode(self.name, mode)
            self.loader = dataset
        _check_loader(self.loader, self.mode)
        self.params = self.mode.resolve(**params)
        if allow_small_n is None:
            allow_small_n = self.mode.allow_small_n
        self.estimator = "ledoit_wolf" if allow_small_n else (estimator or self.mode.estimator)
        self.allow_small_n = allow_small_n
        self.return_mean = return_mean
        self.normalize = normalize
        self.verbose = verbose
        self._check_sizes()
        self._load(cache_dir, force)
        self._setup_split(split, subset)

    # -- tokens -------------------------------------------------------------
    def structured(self, i):
        """Structured sample ``i`` of the loader (unscaled), axes ``mode.axes``."""
        return np.asarray(self.mode.source(self.loader, int(i)))

    def tokenize(self, i):
        """``(observations, positions)`` of loader sample ``i`` (see
        :mod:`~spdnet_datasets.series.tokenizers`), invalid observations dropped."""
        obs, pos = self.mode.tokenize(self.structured(i), **self.params)
        if self.mode.drop_invalid:
            obs = [_valid(o) for o in obs]
        return obs, pos

    def _check_sizes(self):
        """``ValueError`` when the nominal tokens of the first sample have ``n_t < 2p``."""
        obs, _ = self.mode.tokenize(self.structured(0), **self.params)
        p, n_t = obs[0].shape[1], min(len(o) for o in obs)
        if n_t < self.mode.min_n_over_p * p and not self.allow_small_n:
            raise ValueError(
                f"{self.name}/{self.mode.name} with {self.params}: tokens of {n_t} observations "
                f"for p = {p}, fewer than {self.mode.min_n_over_p:g}p = "
                f"{int(np.ceil(self.mode.min_n_over_p * p))}. Enlarge the windows, reduce p "
                "(n_channels, band_window) or pass allow_small_n=True (Ledoit-Wolf).")

    def _min_n(self, p):
        """Fewest observations of an unmasked token."""
        return 2 if self.allow_small_n else int(np.ceil(self.mode.min_n_over_p * p))

    def _compute(self, out_dir):
        out_dir = Path(out_dir)
        n = len(self.loader)
        tokens, means, counts = [], [], []
        for i in range(n):
            obs, _ = self.tokenize(i)
            covs, mus, ns = zip(*(_estimate(o, self.estimator, self._min_n(o.shape[1]))
                                  for o in obs), strict=True)
            tokens.append(covs)
            means.append(mus)
            counts.append(ns)
            if self.verbose and (i + 1) % 1000 == 0:
                print(f"  {self.name}/{self.mode.name}: {i + 1}/{n} samples")
        lengths = np.array([len(t) for t in tokens])
        s_max, p = int(lengths.max()), len(means[0][0])
        T = np.lib.format.open_memmap(out_dir / "tokens.npy", "w+", np.float64, (n, s_max, p, p))
        M = np.zeros((n, s_max, p))
        mask = np.zeros((n, s_max), bool)
        n_obs = np.zeros((n, s_max), np.int64)
        eye = np.eye(p)
        for i, (covs, mus, ns) in enumerate(zip(tokens, means, counts, strict=True)):
            T[i] = eye
            for k, (c, m, nk) in enumerate(zip(covs, mus, ns, strict=True)):
                if c is not None:
                    T[i, k], mask[i, k] = c, True
                M[i, k], n_obs[i, k] = m, nk
        T.flush()
        del T
        np.save(out_dir / "means.npy", M)
        np.save(out_dir / "mask.npy", mask)
        np.save(out_dir / "n_obs.npy", n_obs)
        np.save(out_dir / "lengths.npy", lengths)

    def _load(self, cache_dir, force):
        key = {"mode": self.mode.name, "params": self.params, "estimator": self.estimator,
               "allow_small_n": self.allow_small_n, "source": _source_key(self.loader)}

        def process(_, out_dir):
            (Path(out_dir) / "key.json").write_text(json.dumps(key, default=str, indent=1))

        run = run_pipeline(f"series/{self.name}", params=key, process_fn=process,
                           covariance_fn=lambda _, out: self._compute(out),
                           cache_dir=cache_dir, keep="keep_full", force=force)
        d = Path(run.covariance_dir)
        self.cache_path = d
        self._tokens = np.load(d / "tokens.npy", mmap_mode="r")
        self._means = np.load(d / "means.npy")
        self._mask = np.load(d / "mask.npy")
        self.n_obs = np.load(d / "n_obs.npy")
        self.lengths = np.load(d / "lengths.npy")

    # -- split and normalisation -------------------------------------------
    def _setup_split(self, split, subset):
        ds = self.loader
        splits = getattr(ds, "splits", {}) or {}
        self.split = split if split is not None else getattr(ds, "split", None)
        if (self.normalize or subset is not None) and self.split not in splits:
            raise ValueError(f"Unknown split {self.split!r}; available: {sorted(splits)}")
        self.scale = 1.0
        if self.normalize:
            train = np.unique(np.asarray(splits[self.split]["train"], dtype=int))
            mask = self._mask[train]
            if mask.any():
                self.scale = geometric_scale(np.asarray(self._tokens[train])[mask])
        index = np.arange(len(ds))
        if subset is not None:
            index = np.asarray(splits[self.split][subset], dtype=int)
            splits = {}
        self.subset = subset
        self._index = index
        self.splits = splits
        self.labels = np.asarray(ds.labels)[index]
        groups = getattr(ds, "groups", None)
        self.groups = None if groups is None else np.asarray(groups)[index]
        self.classes = list(getattr(ds, "classes", []))
        if self.verbose:
            print(f"CovarianceSeries({self.name!r}, {self.mode.name!r}): {len(index)} samples, "
                  f"S <= {self._tokens.shape[1]}, p = {self._tokens.shape[-1]}, "
                  f"{100 * self.masked_fraction():.2f} % masked tokens, scale {self.scale:.3e}")

    # -- access ------------------------------------------------------------
    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        j = int(self._index[i])
        S = int(self.lengths[j])
        mask = self._mask[j, :S]
        tokens = np.array(self._tokens[j, :S])
        tokens[mask] *= self.scale
        out = (torch.from_numpy(tokens), torch.from_numpy(mask.copy()))
        if self.return_mean:
            out += (torch.from_numpy(self._means[j, :S] * np.sqrt(self.scale)
                                     * mask[:, None]),)
        y = self.labels[i]
        if getattr(self.loader, "task", None) == "regression":
            return out, torch.tensor(np.asarray(y, dtype=np.float64))
        return out, int(y)

    @property
    def num_classes(self):
        return len(self.classes)

    def get_num_classes(self):
        return len(self.classes)

    def masked_fraction(self):
        """Fraction of the tokens (up to each sample's length) that are masked."""
        valid = np.arange(self._mask.shape[1])[None] < self.lengths[:, None]
        return float(1 - self._mask[valid].mean())

    def describe(self):
        """Size of the series for the current parameters, and measured token statistics.

        Returns a dict with ``dataset``, ``mode``, ``family``, ``params``, ``S`` (largest number of
        tokens), ``S_median``, ``p``, ``n_t`` (median over the unmasked tokens), ``n_t_min``,
        ``n_t_over_p``, ``spd_fraction`` (unmasked tokens) and ``kappa`` (5, 50 and 95 %
        quantiles of the condition number of the unmasked tokens, on at most 2 000 samples).
        """
        p = self._tokens.shape[-1]
        valid = np.arange(self._mask.shape[1])[None] < self.lengths[:, None]
        n_ok = self.n_obs[self._mask]
        rows = np.unique(np.linspace(0, len(self._tokens) - 1, 2000).round().astype(int))
        eig = np.linalg.eigvalsh(np.asarray(self._tokens[rows])[self._mask[rows]])
        kappa = eig[:, -1] / eig[:, 0] if len(eig) else np.array([np.nan])
        n_t = float(np.median(n_ok)) if len(n_ok) else 0.0
        return {"dataset": self.name, "mode": self.mode.name, "family": self.mode.family,
                "params": dict(self.params),
                "S": int(self.lengths.max()), "S_median": float(np.median(self.lengths)),
                "p": int(p), "n_t": n_t, "n_t_min": int(n_ok.min()) if len(n_ok) else 0,
                "n_t_over_p": n_t / p, "spd_fraction": float(self._mask[valid].mean()),
                "kappa": [float(q) for q in np.quantile(kappa, [0.05, 0.5, 0.95])]}


def collate_series(batch):
    """Collate ``series[i]`` items of different lengths: pad with masked identity tokens.

    Returns ``((tokens (B, S, p, p), mask (B, S)[, mean (B, S, p)]), labels)``.
    """
    xs, ys = zip(*batch, strict=True)
    S = max(x[0].shape[0] for x in xs)
    p = xs[0][0].shape[-1]
    tokens = torch.eye(p, dtype=xs[0][0].dtype).repeat(len(xs), S, 1, 1)
    mask = torch.zeros(len(xs), S, dtype=torch.bool)
    for b, x in enumerate(xs):
        tokens[b, :len(x[0])] = x[0]
        mask[b, :len(x[1])] = x[1]
    out = (tokens, mask)
    if len(xs[0]) > 2:
        mean = torch.zeros(len(xs), S, p, dtype=xs[0][2].dtype)
        for b, x in enumerate(xs):
            mean[b, :len(x[2])] = x[2]
        out += (mean,)
    labels = torch.stack([torch.as_tensor(y) for y in ys])
    return out, labels
