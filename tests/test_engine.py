"""Unit tests for the risk engine (VADER mode keeps them fast and offline)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.engine.engine import RiskEngine
from src.engine.entities import find_tickers
from src.engine.events import classify_event
from src.engine.impact import impact_score
from src.engine.preprocess import clean_text, preprocess
from src.engine.sentiment import score_texts
from src.engine.store import SignalStore
from src.ingestion.base import Document, make_doc_id

T0 = datetime(2022, 7, 1, 15, tzinfo=timezone.utc)


def _doc(text: str, i: int = 0, source_type: str = "news", hint=None) -> Document:
    return Document(make_doc_id("test", f"{i}|{text}"), "test", source_type, T0 + timedelta(hours=i), text, "", None, hint or [])


# ---------------- sentiment ----------------
@pytest.mark.parametrize("fast", [True])
def test_sentiment_in_range(fast):
    texts = ["Profits soared to a record high", "The company collapsed into bankruptcy", "The meeting is on Tuesday"]
    res = score_texts(texts, fast=fast)
    assert len(res) == 3
    for r in res:
        assert -1.0 <= r.score <= 1.0
        assert r.label in {"positive", "negative", "neutral"}
        assert 0.0 <= r.confidence <= 1.0
    assert res[0].score > res[1].score


def test_finbert_range_and_direction():
    pytest.importorskip("torch")
    from src.engine.sentiment import FinBertScorer, get_scorer

    scorer = get_scorer(False)
    if not isinstance(scorer, FinBertScorer):
        pytest.skip("FinBERT not cached locally")
    pos, neg = scorer.score(["Apple beats earnings estimates and raises guidance",
                             "Apple misses estimates and cuts its outlook"])
    assert -1 <= neg.score < 0 < pos.score <= 1


# ---------------- entities ----------------
def test_ticker_aliases_and_cashtags():
    assert find_tickers("Tim Cook unveils the new iPhone") == ["AAPL"]
    assert find_tickers("$GOOG and Facebook both fell") == ["GOOGL", "META"]
    assert find_tickers("anything", hint=["GOOG"]) == ["GOOGL"]


def test_context_exclusions():
    assert find_tickers("Brazil: Amazon deforestation lowest in decades") == []
    assert find_tickers("NVIDIA Tesla V100 prices climb") == ["NVDA"]
    assert find_tickers("Tesla recalls cars") == ["TSLA"]


def test_no_false_positive_on_short_tickers():
    # 'ko', 'dis' and 'bac' as normal words must not map to KO / DIS / BAC
    assert find_tickers("the dis-agreement was ko'd by bacteria") == []


# ---------------- events ----------------
@pytest.mark.parametrize("text,expected", [
    ("Moody's downgrades the company's credit rating amid default fears", "Credit Event"),
    ("Microsoft agrees to acquire gaming company in $69 billion takeover", "Merger/Acquisition"),
    ("Netflix beats quarterly earnings estimates and raises guidance", "Earnings"),
    ("Apple unveils new iPhone at launch event", "Product Launch"),
    ("Justice Department files antitrust lawsuit against Google", "Regulatory/Legal"),
    ("Disney CEO resigns; board appoints successor", "Management Change"),
    ("Chip shortage forces Tesla to halt production at its factory", "Supply Chain/Operations"),
    ("Fed raises interest rates as inflation hits 40-year high", "Macroeconomic"),
    ("New sanctions on Russia as war escalates", "Geopolitical"),
    ("Shares were little changed on Tuesday", "Other"),
])
def test_event_rules(text, expected):
    res = classify_event(text)
    assert res.event_type == expected
    assert 0 <= res.confidence <= 1


# ---------------- impact ----------------
@pytest.mark.parametrize("event", ["Credit Event", "Earnings", "Other", "Unknown"])
@pytest.mark.parametrize("s", [-1.0, -0.3, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("src", ["news", "social"])
@pytest.mark.parametrize("mentions", [0, 5, 10_000])
def test_impact_bounds(event, s, src, mentions):
    v = impact_score(event, s, src, mentions)
    assert 1 <= v <= 10


def test_impact_monotonic():
    assert impact_score("Credit Event", -0.9, "news") > impact_score("Other", -0.9, "news")
    assert impact_score("Earnings", -0.9, "news") > impact_score("Earnings", -0.1, "news")
    assert impact_score("Earnings", 0.6, "news") > impact_score("Earnings", 0.6, "social")
    assert impact_score("Earnings", 0.6, "news", 50) > impact_score("Earnings", 0.6, "news", 0)


# ---------------- preprocess & engine ----------------
def test_clean_tweet_keeps_cashtags():
    out = clean_text("@bob $TSLA to the moon \U0001F680 https://t.co/xyz #EV", social=True)
    assert "$TSLA" in out and "http" not in out and "@bob" not in out and "bullish" in out


def test_preprocess_dedupes():
    docs = [_doc("Apple beats estimates", 0), _doc("Apple beats estimates!", 1), _doc("Pfizer drug approved", 2)]
    assert len(preprocess(docs)) == 2


def test_engine_end_to_end(tmp_path):
    store = SignalStore(tmp_path / "t.db")
    eng = RiskEngine(store=store, fast=True, run_mode="test")
    docs = [
        _doc("Apple and Microsoft beat quarterly earnings estimates", 0),
        _doc("$TSLA crashing after recall news, terrible", 1, "social"),
        _doc("Weather is nice today", 2),
    ]
    sigs = eng.analyze(docs)
    assert {s["ticker"] for s in sigs} == {"AAPL", "MSFT", "TSLA"}
    for s in sigs:
        assert -1 <= s["sentiment_score"] <= 1 and 1 <= s["impact_score"] <= 10
        assert s["event_type"] in {"Earnings", "Supply Chain/Operations"}
    # second run must not re-score anything
    assert eng.analyze(docs) == []
    assert len(store.load_signals()) == 3
