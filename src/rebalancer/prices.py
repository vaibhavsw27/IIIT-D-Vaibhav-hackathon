"""Daily close prices via yfinance, cached to data/raw/prices.csv so demos run offline."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from src.config import load_config, path

log = logging.getLogger(__name__)


def _download(symbols: list[str], start: str, end: str, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    for attempt in range(1, retries + 1):
        try:
            df = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False, threads=True, timeout=30)
            close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
            close = close.dropna(how="all")
            if not close.empty:
                return close
        except Exception as exc:
            log.warning("yfinance attempt %d failed: %s", attempt, exc)
        time.sleep(2 * attempt)
    return pd.DataFrame()


def load_prices(start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame of daily closes (index=date, columns=tickers + benchmark).

    Uses the cache if it covers the window; otherwise downloads and merges into the cache.
    Falls back to whatever is cached if the download fails.
    """
    cfg = load_config()
    symbols = list(cfg["universe"]) + [cfg["benchmark"]]
    cache = Path(path("prices"))
    cached = pd.DataFrame()
    if cache.exists():
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)

    s, e = pd.Timestamp(start), pd.Timestamp(end)
    covered = (
        not cached.empty
        and set(symbols).issubset(cached.columns)
        and cached.index.min() <= s + pd.Timedelta(days=5)
        and cached.index.max() >= e - pd.Timedelta(days=5)
    )
    if refresh or not covered:
        log.info("Downloading prices %s..%s for %d symbols", start, end, len(symbols))
        fresh = _download(symbols, (s - pd.Timedelta(days=10)).date().isoformat(), (e + pd.Timedelta(days=10)).date().isoformat())
        if not fresh.empty:
            fresh.index = pd.to_datetime(fresh.index).tz_localize(None)
            cached = fresh if cached.empty else pd.concat([cached, fresh]).groupby(level=0).last().sort_index()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cached.round(4).to_csv(cache)
        elif cached.empty:
            raise RuntimeError("No price data: download failed and no cache at data/raw/prices.csv")
        else:
            log.warning("Price download failed - using cached prices only")
    out = cached.loc[(cached.index >= s) & (cached.index <= e)]
    return out[[c for c in symbols if c in out.columns]]
