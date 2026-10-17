"""Loader for Kaggle "Stock Tweets for Sentiment Analysis and Prediction" (equinxx).

File: stock_tweets.csv with columns Date, Tweet, Stock Name, Company Name.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import load_config, path
from src.ingestion.base import Document, make_doc_id

log = logging.getLogger(__name__)


def load_kaggle_tweets(
    csv_path: Path | None = None,
    source: str = "tweets",
    max_per_ticker_day: int | None = None,
    start: str | None = None,
    end: str | None = None,
) -> list[Document]:
    """Load tweets, keep only tickers in our universe, optionally cap volume per ticker per day.

    The cap keeps CPU FinBERT runtime manageable on the full dataset (~80k tweets).
    """
    csv_path = csv_path or path("kaggle_tweets")
    if not Path(csv_path).exists():
        log.warning("Tweets file not found: %s (download it from Kaggle into data/raw/)", csv_path)
        return []

    cfg = load_config()
    remap = cfg.get("ticker_remap", {})
    universe = set(cfg["universe"])

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Date": "date", "Tweet": "tweet", "Stock Name": "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.upper().replace(remap)
    df = df[df["ticker"].isin(universe)].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed")
    if start:
        df = df[df["date"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["date"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    df = df.drop_duplicates(subset=["tweet", "ticker"])

    if max_per_ticker_day:
        df["day"] = df["date"].dt.date
        # random (fixed-seed) subset of at most N tweets per ticker per day
        df = df.sample(frac=1.0, random_state=42).groupby(["ticker", "day"]).head(max_per_ticker_day)
        df = df.sort_values("date")

    docs = [
        Document(
            doc_id=make_doc_id(source, f"{r.ticker}|{r.date.isoformat()}|{r.tweet}"),
            source=source,
            source_type="social",
            timestamp=r.date.to_pydatetime(),
            title=str(r.tweet),
            text="",
            url=None,
            tickers_hint=[r.ticker],
        )
        for r in df.itertuples(index=False)
    ]
    log.info("Loaded %d tweets from %s", len(docs), Path(csv_path).name)
    return docs
