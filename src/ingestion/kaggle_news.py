"""Loader for Kaggle "Sentiment Analysis for Financial News" (ankurzing).

File: all-data.csv, no header, latin-1, columns: label, sentence.
The dataset has labels but no dates or tickers, so it is used to EVALUATE the
sentiment model, and its sentences are included as undated news only for the
engine demo (they never enter the dated backtest).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import path
from src.ingestion.base import Document, make_doc_id, to_utc

log = logging.getLogger(__name__)


def load_labelled_news(csv_path: Path | None = None) -> pd.DataFrame:
    """Return DataFrame with columns [label, text] (label in positive/negative/neutral)."""
    csv_path = csv_path or path("kaggle_news")
    if not Path(csv_path).exists():
        log.warning("Financial news file not found: %s (download it from Kaggle into data/raw/)", csv_path)
        return pd.DataFrame(columns=["label", "text"])
    df = pd.read_csv(csv_path, header=None, names=["label", "text"], encoding="latin-1")
    df["label"] = df["label"].str.strip().str.lower()
    return df.dropna().drop_duplicates(subset=["text"]).reset_index(drop=True)


def load_kaggle_news(csv_path: Path | None = None, timestamp: str = "2022-01-01") -> list[Document]:
    """Wrap the labelled sentences as Documents (fixed placeholder timestamp, flagged in source)."""
    df = load_labelled_news(csv_path)
    ts = to_utc(timestamp)
    docs = [
        Document(
            doc_id=make_doc_id("kaggle_news", r.text),
            source="kaggle_news",
            source_type="news",
            timestamp=ts,
            title=r.text,
            text="",
        )
        for r in df.itertuples(index=False)
    ]
    log.info("Loaded %d labelled news sentences", len(docs))
    return docs
