"""Common document schema shared by every ingestion source."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd


@dataclass
class Document:
    """One piece of unstructured text (news article, headline or tweet)."""

    doc_id: str
    source: str           # "gdelt" | "newsapi" | "kaggle_news" | "tweets" | "sample_news" | "sample_tweets"
    source_type: str      # "news" | "social"
    timestamp: datetime   # UTC, timezone-aware
    title: str
    text: str
    url: str | None = None
    tickers_hint: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Title + body, without repeating the title when they are identical."""
        if self.text and self.text.strip() != self.title.strip():
            return f"{self.title}. {self.text}".strip(". ")
        return self.title or self.text


def make_doc_id(source: str, key: str) -> str:
    """Stable id: hash of source + url (or text)."""
    return hashlib.sha1(f"{source}|{key}".encode("utf-8")).hexdigest()[:16]


def to_utc(value) -> datetime:
    """Parse anything date-like into a timezone-aware UTC datetime."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
