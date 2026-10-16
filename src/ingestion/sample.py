"""Loaders for the offline synthetic sample CSVs in data/sample/."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import path
from src.ingestion.base import Document, make_doc_id, to_utc

log = logging.getLogger(__name__)


def _split_hint(value) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [t.strip().upper() for t in value.split(";") if t.strip()]
    return []


def load_sample_news(csv_path: Path | None = None) -> list[Document]:
    """Columns: timestamp, source, title, text, url, tickers."""
    csv_path = csv_path or path("sample_news")
    df = pd.read_csv(csv_path)
    docs = [
        Document(
            doc_id=make_doc_id("sample_news", str(r.url or r.title)),
            source="sample_news",
            source_type="news",
            timestamp=to_utc(r.timestamp),
            title=str(r.title),
            text=str(r.text) if isinstance(r.text, str) else "",
            url=r.url if isinstance(r.url, str) else None,
            tickers_hint=_split_hint(r.tickers),
        )
        for r in df.itertuples(index=False)
    ]
    log.info("Loaded %d sample news docs from %s", len(docs), csv_path.name)
    return docs


def load_sample_tweets(csv_path: Path | None = None) -> list[Document]:
    """Same column layout as the Kaggle stock-tweets dataset: Date, Tweet, Stock Name, Company Name."""
    from src.ingestion.kaggle_tweets import load_kaggle_tweets

    csv_path = csv_path or path("sample_tweets")
    return load_kaggle_tweets(csv_path, source="sample_tweets")
