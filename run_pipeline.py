"""Entry point: ingest -> analyze -> store risk signals.

Examples (Windows, from the repo root with the venv active):
  python run_pipeline.py                       # sample mode, offline
  python run_pipeline.py --mode historical     # Kaggle CSVs in data/raw/
  python run_pipeline.py --mode live --interval 900
  python run_pipeline.py --fast                # VADER instead of FinBERT
"""
from __future__ import annotations

import argparse
import logging
import time

from src.config import load_config, setup_logging
from src.engine.engine import RiskEngine
from src.engine.store import SignalStore

log = logging.getLogger("pipeline")


def run_sample(engine: RiskEngine) -> int:
    from src.ingestion.sample import load_sample_news, load_sample_tweets

    docs = load_sample_news() + load_sample_tweets()
    return len(engine.analyze(docs))


def run_historical(engine: RiskEngine, max_per_day: int, start: str | None, end: str | None) -> int:
    from src.ingestion.kaggle_tweets import load_kaggle_tweets

    docs = load_kaggle_tweets(max_per_ticker_day=max_per_day, start=start, end=end)
    if not docs:
        log.error("No Kaggle tweets found in data/raw/. Download them first (see data/README.md).")
        return 0
    total = 0
    chunk = 2000  # store progress in chunks so a long CPU run can be resumed
    for i in range(0, len(docs), chunk):
        total += len(engine.analyze(docs[i : i + chunk]))
        log.info("Historical progress: %d / %d docs", min(i + chunk, len(docs)), len(docs))
    return total


def run_live_once(engine: RiskEngine) -> int:
    from src.ingestion.gdelt import fetch_gdelt
    from src.ingestion.newsapi import fetch_newsapi

    docs = fetch_gdelt() + fetch_newsapi()
    if not docs:
        log.warning("Live sources returned nothing this cycle (offline or rate-limited).")
        return 0
    return len(engine.analyze(docs))


def live_rebalance(store: SignalStore) -> None:
    """One rebalancing step on the live signal stream (run_id='live')."""
    import pandas as pd

    from src.ingestion.base import now_utc
    from src.rebalancer.aggregator import aggregate_at, prepare
    from src.rebalancer.rebalancer import explain, rebalance

    cfg = load_config()
    tickers = list(cfg["universe"])
    sig = store.load_signals(run_mode="live")
    if sig.empty:
        return
    prev = store.load_weights("live")
    old = (prev[prev["timestamp"] == prev["timestamp"].max()].set_index("ticker")["weight"].reindex(tickers)
           if not prev.empty else pd.Series(1.0 / len(tickers), index=tickers))
    now = pd.Timestamp(now_utc())
    agg, drivers = aggregate_at(prepare(sig), now, tickers, top_n=cfg["rebalancer"]["top_reasons"])
    new = rebalance(agg, old)
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    w = pd.DataFrame({"timestamp": stamp, "ticker": tickers, "weight": new.values, "agg_sentiment": agg.values})
    logs = pd.DataFrame([
        (stamp, t, old[t], new[t], agg[t], explain(t, old[t], new[t], agg[t], drivers.get(t, [])))
        for t in tickers if abs(new[t] - old[t]) > 1e-4
    ], columns=["timestamp", "ticker", "old_weight", "new_weight", "agg_sentiment", "reason"])
    store.append_weights("live", w, logs)
    top = new.sort_values(ascending=False)
    log.info("Live rebalance: top %s, bottom %s",
             ", ".join(f"{t} {v:.1%}" for t, v in top.head(3).items()),
             ", ".join(f"{t} {v:.1%}" for t, v in top.tail(3).items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sample", "historical", "live"], default="sample")
    p.add_argument("--fast", action="store_true", help="use VADER instead of FinBERT")
    p.add_argument("--interval", type=int, default=None, help="live polling interval, seconds")
    p.add_argument("--once", action="store_true", help="live mode: run a single poll and exit")
    p.add_argument("--max-per-day", type=int, default=25, help="historical: cap tweets per ticker per day")
    p.add_argument("--start", default=None, help="historical: start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="historical: end date YYYY-MM-DD")
    p.add_argument("--reset", action="store_true", help="delete existing signals for this mode first")
    args = p.parse_args()

    setup_logging()
    store = SignalStore()
    if args.reset:
        store.clear(args.mode)
    engine = RiskEngine(store=store, fast=args.fast, run_mode=args.mode)

    if args.mode == "sample":
        n = run_sample(engine)
    elif args.mode == "historical":
        n = run_historical(engine, args.max_per_day, args.start, args.end)
    else:
        interval = args.interval or load_config()["live"]["interval_seconds"]
        n = 0
        while True:
            try:
                added = run_live_once(engine)
                n += added
                live_rebalance(store)
                store.export()
                log.info("Live cycle done: %d new signals (total this session %d)", added, n)
            except Exception:  # never let the live demo crash
                log.exception("Live cycle failed; will retry next interval")
            if args.once:
                break
            log.info("Sleeping %ds (Ctrl+C to stop)", interval)
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                break

    exported = store.export()
    log.info("Done. %d new signals this run; %d signals in database.", n, exported)


if __name__ == "__main__":
    main()
