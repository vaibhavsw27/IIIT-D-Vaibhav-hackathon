"""Turn a stream of signals into one decayed sentiment number per ticker.

agg_t = sum(s_i * (impact_i/10) * exp(-lambda * age_i)) / (sum((impact_i/10) * exp(-lambda * age_i)) + prior)

lambda = ln(2) / half_life_hours. The small `prior` weight in the denominator
acts like a neutral pseudo-signal, so with no fresh news the score decays to 0.
Only signals with timestamp <= t are used (no look-ahead).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import load_config


def _params() -> tuple[float, float, float]:
    cfg = load_config()["rebalancer"]
    lam = math.log(2) / float(cfg["half_life_hours"])
    return lam, float(cfg.get("prior_weight", 0.2)), float(cfg.get("lookback_days", 7))


def prepare(signals: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns once (parsed UTC timestamps) and sort by time."""
    df = signals.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def aggregate_at(signals: pd.DataFrame, t: pd.Timestamp, tickers: list[str], top_n: int = 0):
    """Aggregate sentiment for every ticker as of time t.

    Returns (Series agg_sentiment indexed by ticker, dict ticker -> top driver rows).
    `signals` must come from prepare().
    """
    lam, prior, lookback = _params()
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    lo = signals["ts"].searchsorted(t - pd.Timedelta(days=lookback), side="left")
    hi = signals["ts"].searchsorted(t, side="right")          # signals at or before t only
    win = signals.iloc[lo:hi]

    agg = pd.Series(0.0, index=tickers)
    drivers: dict[str, list[dict]] = {}
    if win.empty:
        return agg, drivers

    age_h = (t - win["ts"]).dt.total_seconds().to_numpy() / 3600.0
    w = (win["impact_score"].to_numpy() / 10.0) * np.exp(-lam * age_h)
    contrib = w * win["sentiment_score"].to_numpy()
    tmp = pd.DataFrame({"ticker": win["ticker"].to_numpy(), "w": w, "c": contrib})
    g = tmp.groupby("ticker").sum()
    vals = g["c"] / (g["w"] + prior)
    agg.loc[vals.index.intersection(tickers)] = vals.loc[vals.index.intersection(tickers)]

    if top_n:
        win = win.assign(_c=contrib)
        for tk, grp in win.groupby("ticker"):
            top = grp.reindex(grp["_c"].abs().sort_values(ascending=False).index).head(top_n)
            drivers[tk] = top[["timestamp", "headline", "sentiment_score", "event_type", "impact_score"]].to_dict("records")
    return agg.clip(-1, 1), drivers
