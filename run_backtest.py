"""Entry point: historical replay of the sentiment rebalancer + benchmark comparison.

  python run_backtest.py                      # sample signals (offline, uses cached prices)
  python run_backtest.py --mode historical    # Kaggle tweets window (Sep 2021 - Sep 2022)
"""
from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import ROOT, load_config, path, setup_logging  # noqa: E402
from src.engine.store import SignalStore  # noqa: E402
from src.rebalancer.backtest import run_backtest  # noqa: E402
from src.rebalancer.prices import load_prices  # noqa: E402

log = logging.getLogger("backtest")


def _window(mode: str, signals: pd.DataFrame) -> tuple[str, str]:
    if mode == "historical":
        bt = load_config()["backtest"]
        return bt["start"], bt["end"]
    ts = pd.to_datetime(signals["timestamp"], utc=True)
    return ts.min().date().isoformat(), ts.max().date().isoformat()


def save_charts(res, mode: str) -> None:
    out = ROOT / load_config()["paths"]["screenshots"]
    out.mkdir(parents=True, exist_ok=True)
    colors = {"sentiment_index": "#2563eb", "equal_weight": "#9ca3af", "sp500": "#f59e0b"}
    labels = {"sentiment_index": "Sentiment Index", "equal_weight": "Equal Weight", "sp500": "S&P 500"}

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    for col in res.cumulative:
        ax.plot(res.cumulative.index, (res.cumulative[col] - 1) * 100, label=labels[col], color=colors[col], lw=2 if col == "sentiment_index" else 1.5)
    ax.axhline(0, color="#d1d5db", lw=0.8)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(f"Backtest ({mode}): Sentiment Index vs Equal Weight vs S&P 500")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"backtest_cumulative_{mode}.png")
    plt.close(fig)

    W = res.weights.pivot(index="timestamp", columns="ticker", values="weight")
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.stackplot(W.index, (W * 100).T.values, labels=W.columns, alpha=0.9, colors=plt.cm.tab20.colors[: W.shape[1]])
    ax.set_ylabel("Weight (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Index weights over time ({mode})")
    ax.legend(ncol=8, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"weights_stacked_{mode}.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sample", "historical"], default="sample")
    p.add_argument("--refresh-prices", action="store_true")
    args = p.parse_args()
    setup_logging()

    store = SignalStore()
    signals = store.load_signals(run_mode=args.mode)
    if signals.empty:
        log.error("No '%s' signals in the database. Run: python run_pipeline.py --mode %s", args.mode, args.mode)
        return
    start, end = _window(args.mode, signals)
    log.info("Backtest window %s .. %s using %d signals", start, end, len(signals))

    prices = load_prices(start, end, refresh=args.refresh_prices)
    res = run_backtest(signals, prices, start, end)

    store.save_weights(args.mode, res.weights, res.reasons)
    out = res.returns.join(res.cumulative.add_suffix("_growth"))
    results_path = path("backtest_results")
    if args.mode != "sample":
        results_path = results_path.with_name(f"backtest_results_{args.mode}.csv")
    out.to_csv(results_path)
    res.metrics.to_csv(results_path.with_name(results_path.stem + "_metrics.csv"), index=False)
    save_charts(res, args.mode)

    pd.set_option("display.width", 140)
    fmt = res.metrics.copy()
    for c in ["cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown", "avg_daily_turnover"]:
        fmt[c] = (fmt[c] * 100).map("{:.2f}%".format)
    fmt["sharpe"] = fmt["sharpe"].map("{:.2f}".format)
    print("\n" + fmt.to_string(index=False))
    print(f"\nRebalance steps: {res.weights['timestamp'].nunique()}  |  weight changes logged: {len(res.reasons)}")
    print(f"Results: {results_path.relative_to(ROOT)}  |  charts: docs/screenshots/")


if __name__ == "__main__":
    main()
