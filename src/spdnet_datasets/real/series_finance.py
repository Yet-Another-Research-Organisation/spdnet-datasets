"""
Covariance series of multivariate financial returns (user request, 2/11/2026): each token is the p x p covariance
of the returns of p assets (the covariance axis is the assets), estimated by Ledoit-Wolf over a sliding window of
``window`` time steps; the series is the T windows that end at the sample date, ``stride`` steps apart (position t =
lag from the sample date). The task is a forecast: the regime of the market volatility over the ``horizon`` following
steps relative to its level over the span of the series (ratio of standard deviations; low / medium / high,
terciles of the training block): absolute levels drift over decades, relative ones keep the classes balanced in
every block, and the series sees the whole reference period. Second task (user request, 2/11): regression of
the next covariance, the one of the ``window`` steps that follow the sample date (no overlap with the tokens),
as its log-Euclidean coordinates (isometric vech of log C, off-diagonal x sqrt 2). ``relative``: each series
is whitened by its own log-Euclidean mean M (tokens and target M^-1/2 C M^-1/2): covariance levels drift over the
decades (affine-invariant distance 2.7 between the training and validation means of French, 2.2 between the two
halves of the training block), and the label is relative to the series anyway. Splits are blocks of time; a
sample whose horizon leaves its block is dropped (purge), so no label straddles two splits.

- French: daily value-weighted returns (%) of the Kenneth French industry portfolios, nested classifications
  10 / 17 / 30 / 48, from July 1969 (the 48 industries are complete from then on; the 100 size x book-to-market
  portfolios have empty corners until 2026 and are not used); market = Mkt-RF + RF (Fama-French factors).
- Binance: hourly log returns (%) of the USDT spot pairs with complete hourly klines over the whole period, the
  ``p`` largest by quote volume (nested); market = equal-weighted mean of all the retained pairs.

Dates, windows and labels do not depend on p: only the number of assets changes.
"""

import glob
import io
import os

import numpy as np

from ..estimator.covariance import EstimateCovariance

FRENCH_ROOT = "/home/mgallet/Documents/SPDnet/neurips/cache/raw/french"
BINANCE_ROOT = "/home/mgallet/Documents/SPDnet/neurips/cache/raw/binance/1h"
FRENCH_FILES = {10: "10_Industry_Portfolios_Daily.csv", 17: "17_Industry_Portfolios_Daily.csv",
                30: "30_Industry_Portfolios_Daily.csv", 48: "48_Industry_Portfolios_Daily.csv"}
VOLATILITY_REGIMES = ["basse", "moyenne", "haute"]


def read_french(path: str):
    """The first table (value-weighted daily returns, %) of a Kenneth French CSV file, missing values as NaN."""
    import pandas as pd

    lines = open(path).read().splitlines()
    start = next(i for i, line in enumerate(lines) if "Daily" in line and "Weighted" in line or "Mkt-RF" in line)
    start += "Mkt-RF" not in lines[start]  # the factors file has no title line before the header
    end = next((i for i in range(start + 1, len(lines)) if not lines[i][:1].isdigit()), len(lines))
    table = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0)
    table.index = pd.to_datetime(table.index.astype(str).str.strip(), format="%Y%m%d")
    return table.replace([-99.99, -999], np.nan)


def read_binance(root: str = BINANCE_ROOT, start: str = "2021-01-01", end: str = "2026-10-01", max_gap: int = 24):
    """Hourly log returns (%) of the pairs with a file for every month and no gap longer than ``max_gap`` hours
    (shorter gaps: price carried forward, zero return), and their total quote volume."""
    import pandas as pd

    grid = pd.date_range(start, end, freq="h", inclusive="left")
    months = len(pd.period_range(start, pd.Timestamp(end) - pd.Timedelta(hours=1), freq="M"))
    closes, volumes = {}, {}
    for pair in sorted({os.path.basename(f).split("-")[0] for f in glob.glob(f"{root}/*.zip")}):
        files = sorted(glob.glob(f"{root}/{pair}-1h-*.zip"))
        if len(files) < months:
            continue
        table = pd.concat([pd.read_csv(f, header=None, usecols=[0, 4, 7]) for f in files])
        table = table[pd.to_numeric(table[0], errors="coerce").notna()].astype(float)  # header lines, if any
        stamp = table[0].where(table[0] < 1e14, table[0] / 1000)  # microseconds from 2025 on
        close = pd.Series(table[4].to_numpy(), pd.to_datetime(stamp, unit="ms")).groupby(level=0).last()
        close = close.reindex(grid)
        gaps = close.isna().astype(int).groupby(close.notna().cumsum()).sum().max()
        if close.iloc[0] != close.iloc[0] or gaps > max_gap:
            continue
        closes[pair], volumes[pair] = close.ffill(), table[7].sum()
    prices = pd.DataFrame(closes)
    return 100 * np.log(prices).diff().iloc[1:], pd.Series(volumes)


def volatility_series(
    returns: np.ndarray,
    market: np.ndarray,
    block: np.ndarray,
    window: int,
    stride: int,
    every: int,
    horizon: int,
    n_tokens: int = 8,
    correlation: bool = False,
    load: float = 1e-6,
    next_covariance: bool = False,
    relative: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    returns (n, p), market (n,) and block (n,) = 0 / 1 / 2 (train / validation / test) on one time grid. A sample
    every ``every`` steps, at index d: tokens = Ledoit-Wolf covariances of returns[e - window + 1 : e + 1] for
    e = d - (T - 1 - t) stride (diagonal load * tr / p), or the matching correlations; label = tercile (training
    thresholds) of std(market[d + 1 : d + 1 + horizon]) / std(market[d - span + 1 : d + 1]), span = window + (T - 1)
    stride (the steps covered by the series); or, with ``next_covariance``, the log-Euclidean coordinates of the
    covariance of returns[d + 1 : d + 1 + window] (then horizon = window). ``stride`` must be a multiple of
    ``every``, so each window is estimated once.
    """
    assert stride % every == 0 and (horizon == window or not next_covariance)
    estimator = EstimateCovariance("ledoit_wolf", remove_mean=True, exclude_zero=False)
    span = window + (n_tokens - 1) * stride
    ends = np.arange(span - 1, len(returns) - horizon, every)
    ends = ends[block[ends] == block[ends + horizon]]  # purge: the label stays in the block of the sample
    lags = (np.arange(n_tokens) - n_tokens + 1) * stride
    windows = np.unique(ends[:, None] + lags)
    p = returns.shape[1]

    def estimate(last: np.ndarray) -> np.ndarray:
        covariances = np.stack([estimator(returns[e - window + 1:e + 1]) for e in last])
        if correlation:
            scale = 1 / np.sqrt(np.diagonal(covariances, axis1=-2, axis2=-1))
            covariances = covariances * scale[:, :, None] * scale[:, None, :]
        return covariances + load * np.trace(covariances, axis1=-2, axis2=-1)[:, None, None] / p * np.eye(p)

    data = estimate(windows)[np.searchsorted(windows, ends[:, None] + lags)]
    split = block[ends]
    volatility = np.array([market[d + 1:d + 1 + horizon].std() / market[d - span + 1:d + 1].std() for d in ends])
    target = estimate(ends + window) if next_covariance else None
    if relative:
        eigenvalues, eigenvectors = np.linalg.eigh(data)
        mean_log = ((eigenvectors * np.log(eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)).mean(1)
        w, v = np.linalg.eigh(mean_log)
        whiten = (v * np.exp(-w / 2)[:, None, :]) @ np.swapaxes(v, -1, -2)  # M^-1/2
        data = whiten[:, None] @ data @ whiten[:, None]
        data = (data + np.swapaxes(data, -1, -2)) / 2
        if next_covariance:
            target = whiten @ target @ whiten
            target = (target + np.swapaxes(target, -1, -2)) / 2
    if next_covariance:
        labels = log_coordinates(target)
    else:
        labels = np.digitize(volatility, np.quantile(volatility[split == 0], [1 / 3, 2 / 3]))
    return data, labels, np.full(len(ends), window), {"split": split.astype(np.int64), "volatility": volatility}


def log_coordinates(covariances: np.ndarray) -> np.ndarray:
    """(..., p, p) SPD -> (..., p (p + 1) / 2): upper triangle of log C, off-diagonal entries x sqrt 2 (the
    Frobenius norm of log C is the Euclidean norm of the coordinates)."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    log = (eigenvectors * np.log(eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)
    rows, cols = np.triu_indices(covariances.shape[-1])
    return log[..., rows, cols] * np.where(rows == cols, 1.0, np.sqrt(2))


def french_series(
    p: int = 30,
    root: str = FRENCH_ROOT,
    start: str = "1969-07-01",
    blocks: tuple[str, str] = ("2000-01-01", "2008-01-01"),
    window: int = 200,
    stride: int = 150,
    every: int = 5,
    horizon: int = 21,
    correlation: bool = False,
    next_covariance: bool = False,
    relative: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Daily US equity portfolios from ``start`` (all the files complete); train 1974-1999, validation 2000-2007,
    test 2008-2026 (2008 and 2020 crises)."""
    returns = read_french(os.path.join(root, FRENCH_FILES[p]))
    factors = read_french(os.path.join(root, "F-F_Research_Data_Factors_daily.csv"))
    returns = returns[returns.index >= start]
    assert not returns.isna().any().any(), f"missing returns after {start}"
    market = (factors["Mkt-RF"] + factors["RF"]).reindex(returns.index)
    block = np.searchsorted(np.array(blocks, dtype="datetime64[ns]"), returns.index.values, side="right")
    return volatility_series(returns.to_numpy(np.float64), market.to_numpy(np.float64), block, window, stride,
                             every, window if next_covariance else horizon,
                             correlation=correlation, next_covariance=next_covariance,
                             relative=relative)


def binance_series(
    p: int = 50,
    root: str = BINANCE_ROOT,
    blocks: tuple[str, str] = ("2024-01-01", "2025-01-01"),
    n_market: int = 100,
    window: int = 200,
    stride: int = 72,
    every: int = 12,
    horizon: int = 24,
    correlation: bool = False,
    next_covariance: bool = False,
    relative: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Hourly crypto returns 2021-01 to 2026-09; train 2021-2023, validation 2024, test 2025-2026; market = mean
    of the ``n_market`` pairs with the largest quote volume, of which the series keeps the first ``p``."""
    returns, volume = read_binance(root)
    pairs = volume.sort_values(ascending=False).index[:n_market]
    assert len(pairs) == n_market and p <= n_market
    market = returns[pairs].mean(axis=1)
    block = np.searchsorted(np.array(blocks, dtype="datetime64[ns]"), returns.index.values, side="right")
    return volatility_series(returns[pairs[:p]].to_numpy(np.float64), market.to_numpy(np.float64), block, window,
                             stride, every, window if next_covariance else horizon,
                             correlation=correlation, next_covariance=next_covariance,
                             relative=relative)
