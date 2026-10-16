"""RiskEngine: unstructured documents in -> structured risk signals out.

Pipeline per batch:
  preprocess -> dedupe (incl. docs already in the DB) -> ticker extraction
  -> sentiment (FinBERT / VADER) -> event classification -> impact score -> store
"""
from __future__ import annotations

import bisect
import hashlib
import logging
from collections import defaultdict
from datetime import timedelta

import pandas as pd

from src.engine.entities import company_name, find_tickers
from src.engine.events import classify_event
from src.engine.impact import impact_score
from src.engine.preprocess import preprocess
from src.engine.sentiment import score_texts
from src.engine.store import SignalStore
from src.ingestion.base import Document

log = logging.getLogger(__name__)


def _iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


class RiskEngine:
    def __init__(self, store: SignalStore | None = None, fast: bool = False, run_mode: str = "sample") -> None:
        self.store = store
        self.fast = fast
        self.run_mode = run_mode

    def analyze(self, docs: list[Document], persist: bool = True) -> list[dict]:
        """Turn documents into signals. One document can yield several signals (one per ticker)."""
        docs = preprocess(docs)
        if self.store is not None:
            known = self.store.known_doc_ids(d.doc_id for d in docs)
            if known:
                log.info("Skipping %d already-scored documents", len(known))
            docs = [d for d in docs if d.doc_id not in known]

        pairs = [(d, find_tickers(d.full_text, d.tickers_hint)) for d in docs]
        no_ticker = [d for d, t in pairs if not t]
        pairs = [(d, t) for d, t in pairs if t]
        if not pairs:
            self._mark(no_ticker, {})
            return []

        log.info("Scoring %d documents (%s)", len(pairs), "VADER" if self.fast else "FinBERT")
        sentiments = score_texts([d.full_text for d, _ in pairs], fast=self.fast)
        mention_lookup = self._mention_index(pairs)

        signals: list[dict] = []
        per_doc: dict[str, int] = defaultdict(int)
        for (doc, tickers), sent in zip(pairs, sentiments):
            ev = classify_event(doc.full_text)
            for ticker in tickers:
                mentions = mention_lookup(ticker, doc.timestamp)
                signals.append({
                    "signal_id": hashlib.sha1(f"{doc.doc_id}|{ticker}".encode()).hexdigest()[:16],
                    "doc_id": doc.doc_id,
                    "timestamp": _iso(doc.timestamp),
                    "ticker": ticker,
                    "company": company_name(ticker),
                    "source": doc.source,
                    "source_type": doc.source_type,
                    "sentiment_score": sent.score,
                    "sentiment_label": sent.label,
                    "sentiment_confidence": sent.confidence,
                    "sentiment_model": sent.model,
                    "event_type": ev.event_type,
                    "event_confidence": ev.confidence,
                    "impact_score": impact_score(ev.event_type, sent.score, doc.source_type, mentions),
                    "headline": doc.title[:500],
                    "url": doc.url,
                    "run_mode": self.run_mode,
                })
                per_doc[doc.doc_id] += 1

        if persist and self.store is not None:
            added = self.store.save_signals(signals)
            self._mark([d for d, _ in pairs] + no_ticker, per_doc)
            log.info("Stored %d new signals from %d documents", added, len(pairs))
        return signals

    def _mark(self, docs: list[Document], per_doc: dict[str, int]) -> None:
        if self.store is not None and docs:
            self.store.mark_docs([(d.doc_id, d.source, _iso(d.timestamp), per_doc.get(d.doc_id, 0)) for d in docs])

    def _mention_index(self, pairs: list[tuple[Document, list[str]]]):
        """Build a function (ticker, ts) -> # mentions of that ticker in the previous 24h."""
        times: dict[str, list[pd.Timestamp]] = defaultdict(list)
        for d, tickers in pairs:
            for t in tickers:
                times[t].append(pd.Timestamp(d.timestamp))
        if self.store is not None:
            hist = self.store.ticker_timestamps()
            for t, ts in zip(hist["ticker"], pd.to_datetime(hist["timestamp"], utc=True)):
                times[t].append(ts)
        for t in times:
            times[t].sort()

        def lookup(ticker: str, ts) -> int:
            arr = times.get(ticker, [])
            ts = pd.Timestamp(ts)
            lo = bisect.bisect_left(arr, ts - timedelta(hours=24))
            hi = bisect.bisect_left(arr, ts)   # strictly before this doc
            return max(0, hi - lo)

        return lookup
