"""Daily backtest: sentiment-weighted index vs equal weight vs S&P 500.

Timing (no look-ahead):
  * On trading day t we aggregate only signals with timestamp <= t 16:00 New York (20:00 UTC).
  * Weights decided at t's close earn the close-to-close return of day t+1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import load_config
from src.rebalancer.aggregator import aggregate_at, prepare
from src.rebalancer.rebalancer import explain, rebalance

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    weights: pd.DataFrame        # long: timestamp, ticker, weight, agg_sentiment
    reasons: pd.DataFrame        # long: timestamp, ticker, old_weight, new_weight, agg_sentiment, reason
    returns: pd.DataFrame        # daily returns: sentiment_index, equal_weight, sp500
    cumulative: pd.DataFrame     # growth of 1
    metrics: pd.DataFrame        # one row per strategy
    turnover: pd.Series


def run_weight_path(signals: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str]):
    """Compute the daily weight path and rebalance reasons."""
    cfg = load_config()["rebalancer"]
    sig = prepare(signals)
    close_h = int(cfg.get("close_hour_utc", 20))
    old = pd.Series(1.0 / len(tickers), index=tickers)
    w_rows, r_rows = [], []
    for d in dates:
        t = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=close_h)
        agg, drivers = aggregate_at(sig, t, tickers, top_n=int(cfg.get("top_reasons", 3)))
        new = rebalance(agg, old, cfg)
        for tk in tickers:
            w_rows.append((d, tk, new[tk], agg[tk]))
            if abs(new[tk] - old[tk]) > 1e-4:
                r_rows.append((d, tk, old[tk], new[tk], agg[tk], explain(tk, old[tk], new[tk], agg[tk], drivers.get(tk, []))))
        old = new
    weights = pd.DataFrame(w_rows, columns=["timestamp", "ticker", "weight", "agg_sentiment"])
    reasons = pd.DataFrame(r_rows, columns=["timestamp", "ticker", "old_weight", "new_weight", "agg_sentiment", "reason"])
    return weights, reasons


def metrics(daily: pd.Series, trading_days: int = 252, rf: float = 0.0) -> dict:
    daily = daily.dropna()
    if daily.empty:
        return {}
    growth = (1 + daily).cumprod()
    years = len(daily) / trading_days
    vol = daily.std(ddof=1) * np.sqrt(trading_days)
    ann_ret = growth.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = ((daily.mean() - rf / trading_days) / daily.std(ddof=1) * np.sqrt(trading_days)) if daily.std() > 0 else np.nan
    dd = (growth / growth.cummax() - 1).min()
    return {
        "cumulative_return": growth.iloc[-1] - 1,
        "annualized_return": ann_ret,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
    }


def run_backtest(signals: pd.DataFrame, prices: pd.DataFrame, start: str, end: str) -> BacktestResult:
    cfg = load_config()
    tickers = list(cfg["universe"])
    bench = cfg["benchmark"]
    bt = cfg["backtest"]

    px = prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]
    px = px.dropna(subset=tickers, how="any")
    rets = px.pct_change()
    dates = px.index

    weights, reasons = run_weight_path(signals, dates, tickers)
    W = weights.pivot(index="timestamp", columns="ticker", values="weight")[tickers]

    # weights decided at close of day t apply to returns of day t+1  -> shift(1)
    w_applied = W.shift(1)
    sent_ret = (w_applied * rets[tickers]).sum(axis=1, min_count=1)
    eq_ret = rets[tickers].mean(axis=1)
    sp_ret = rets[bench] if bench in rets else pd.Series(np.nan, index=rets.index)

    daily = pd.DataFrame({"sentiment_index": sent_ret, "equal_weight": eq_ret, "sp500": sp_ret}).iloc[1:]
    cumulative = (1 + daily.fillna(0)).cumprod()
    turnover = (W.diff().abs().sum(axis=1) / 2).iloc[1:]

    rows = []
    for col, label in [("sentiment_index", "Sentiment Index"), ("equal_weight", "Equal Weight"), ("sp500", "S&P 500")]:
        m = metrics(daily[col], bt["trading_days"], bt["risk_free_rate"])
        m["avg_daily_turnover"] = float(turnover.mean()) if col == "sentiment_index" else 0.0
        rows.append({"strategy": label, **m})
    return BacktestResult(weights, reasons, daily, cumulative, pd.DataFrame(rows), turnover)
