"""Unit tests for aggregation, rebalancing and the backtest timing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.rebalancer.aggregator import aggregate_at, prepare
from src.rebalancer.backtest import run_backtest
from src.rebalancer.rebalancer import rebalance

CFG = load_config()
TICKERS = list(CFG["universe"])
RB = CFG["rebalancer"]


def _signals(rows):
    return pd.DataFrame(rows, columns=["timestamp", "ticker", "sentiment_score", "impact_score", "headline", "event_type"])


# ---------------- weights ----------------
@pytest.mark.parametrize("seed", range(25))
def test_weights_sum_to_one_and_respect_bounds(seed):
    rng = np.random.default_rng(seed)
    agg = pd.Series(rng.uniform(-1, 1, len(TICKERS)), index=TICKERS)
    old = rebalance(pd.Series(0.0, index=TICKERS), None)  # feasible starting point
    for _ in range(10):
        new = rebalance(agg, old)
        assert new.sum() == pytest.approx(1.0, abs=1e-9)
        assert (new >= RB["w_min"] - 1e-9).all() and (new <= RB["w_max"] + 1e-9).all()
        assert ((new - old).abs() <= RB["turnover_cap"] + 1e-9).all()
        old = new
        agg = pd.Series(rng.uniform(-1, 1, len(TICKERS)), index=TICKERS)


def test_extreme_sentiment_hits_bounds_without_cap():
    cfg = dict(RB, turnover_cap=None, sensitivity_k=10)
    agg = pd.Series(-1.0, index=TICKERS)
    agg.iloc[0] = 1.0
    w = rebalance(agg, None, cfg)
    assert w.iloc[0] == pytest.approx(RB["w_max"])
    assert w.sum() == pytest.approx(1.0)


def test_neutral_sentiment_gives_equal_weight():
    w = rebalance(pd.Series(0.0, index=TICKERS))
    assert np.allclose(w.values, 1 / len(TICKERS))


def test_positive_sentiment_increases_weight():
    agg = pd.Series(0.0, index=TICKERS)
    agg["AAPL"], agg["XOM"] = 0.8, -0.8
    w = rebalance(agg)
    assert w["AAPL"] > 1 / len(TICKERS) > w["XOM"]


# ---------------- aggregation ----------------
def test_aggregation_ignores_future_and_decays():
    sig = prepare(_signals([
        ("2022-07-01T10:00:00Z", "AAPL", 0.9, 8, "good", "Earnings"),
        ("2022-07-03T10:00:00Z", "AAPL", -0.9, 8, "future bad news", "Earnings"),
    ]))
    t = pd.Timestamp("2022-07-01T20:00:00Z")
    agg, _ = aggregate_at(sig, t, TICKERS)
    assert agg["AAPL"] > 0          # the future negative signal is not visible yet
    later, _ = aggregate_at(sig, pd.Timestamp("2022-07-02T20:00:00Z"), TICKERS)
    assert 0 < later["AAPL"] < agg["AAPL"]   # decays toward neutral with no new news
    assert agg["MSFT"] == 0


# ---------------- backtest: no look-ahead ----------------
def test_no_lookahead_in_backtest():
    """A huge positive signal on day d must not affect returns earned on day d itself."""
    dates = pd.bdate_range("2022-07-01", periods=10)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (10, len(TICKERS) + 1)), axis=0),
                          index=dates, columns=TICKERS + [CFG["benchmark"]])
    d = dates[5]
    sig = _signals([(f"{d.date()}T15:00:00Z", "NVDA", 1.0, 10, "huge news", "Earnings")])
    res = run_backtest(sig, prices, str(dates[0].date()), str(dates[-1].date()))
    W = res.weights.pivot(index="timestamp", columns="ticker", values="weight")
    # weights change on day d (decided at d's close) ...
    assert W.loc[d, "NVDA"] > W.loc[dates[4], "NVDA"]
    # ... but the return of day d still uses the pre-news weights (equal weights)
    rets = prices[TICKERS].pct_change()
    expected_d = (W.loc[dates[4]] * rets.loc[d]).sum()
    assert res.returns.loc[d, "sentiment_index"] == pytest.approx(expected_d)
    assert res.returns.loc[d, "sentiment_index"] == pytest.approx(res.returns.loc[d, "equal_weight"])
    # and the day after uses the new weights
    expected_next = (W.loc[d] * rets.loc[dates[6]]).sum()
    assert res.returns.loc[dates[6], "sentiment_index"] == pytest.approx(expected_next)
