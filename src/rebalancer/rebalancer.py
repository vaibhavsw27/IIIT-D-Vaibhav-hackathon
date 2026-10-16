"""Sentiment-tilted index weights.

target_i = w_base_i * exp(k * agg_sentiment_i)
Then find the closest feasible weights: each weight inside
[max(w_min, w_old - cap), min(w_max, w_old + cap)] and sum(w) = 1.
We solve this exactly with a 1-D bisection on a scale factor c:
    w_i(c) = clip(c * target_i, lower_i, upper_i),   choose c so that sum w_i(c) = 1.
Because the previous weights are feasible, a solution always exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config


def _solve(target: np.ndarray, lower: np.ndarray, upper: np.ndarray, iters: int = 200) -> np.ndarray:
    lower = np.minimum(lower, upper)
    if lower.sum() >= 1:
        return lower / lower.sum()
    if upper.sum() <= 1:
        return upper / upper.sum()
    lo, hi = 0.0, 1.0
    while np.clip(hi * target, lower, upper).sum() < 1:
        hi *= 2
    for _ in range(iters):
        mid = (lo + hi) / 2
        if np.clip(mid * target, lower, upper).sum() < 1:
            lo = mid
        else:
            hi = mid
    w = np.clip(hi * target, lower, upper)
    return w / w.sum()   # tiny numerical correction only


def rebalance(agg_sentiment: pd.Series, old_weights: pd.Series | None = None, cfg: dict | None = None) -> pd.Series:
    """Return new weights (Series indexed like agg_sentiment) that sum to 1 and respect all bounds."""
    cfg = cfg or load_config()["rebalancer"]
    n = len(agg_sentiment)
    k = float(cfg["sensitivity_k"])
    w_min, w_max = float(cfg["w_min"]), float(cfg["w_max"])
    cap = cfg.get("turnover_cap")

    base = np.full(n, 1.0 / n)
    target = base * np.exp(k * agg_sentiment.fillna(0).to_numpy(dtype=float))
    lower = np.full(n, w_min)
    upper = np.full(n, w_max)
    if old_weights is not None and cap:
        old = old_weights.reindex(agg_sentiment.index).fillna(1.0 / n).to_numpy(dtype=float)
        lower = np.maximum(lower, old - float(cap))
        upper = np.minimum(upper, old + float(cap))
    return pd.Series(_solve(target, lower, upper), index=agg_sentiment.index)


def explain(ticker: str, old_w: float, new_w: float, agg: float, drivers: list[dict]) -> str:
    """Human-readable reason for a weight change (used in the dashboard)."""
    direction = "raised" if new_w > old_w + 1e-6 else "cut" if new_w < old_w - 1e-6 else "held"
    head = f"{ticker} weight {direction} {old_w:.1%} -> {new_w:.1%} (sentiment {agg:+.2f})."
    if not drivers:
        return head + " No recent signals; sentiment decaying to neutral."
    parts = [
        f"[{d['event_type']}, s={d['sentiment_score']:+.2f}, impact {d['impact_score']:.1f}] {str(d['headline'])[:110]}"
        for d in drivers
    ]
    return head + " Drivers: " + " | ".join(parts)
