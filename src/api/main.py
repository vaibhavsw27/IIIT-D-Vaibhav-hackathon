"""FastAPI service exposing risk signals and index weights to downstream modules.

Run:  uvicorn src.api.main:app --reload --port 8000     (Swagger UI at /docs)
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import load_config
from src.engine.engine import RiskEngine
from src.engine.store import SignalStore
from src.ingestion.base import Document, make_doc_id, now_utc
from src.rebalancer.aggregator import aggregate_at, prepare

app = FastAPI(
    title="AI Risk Signal Engine API",
    version="1.0.0",
    description="Structured risk signals (sentiment, event type, impact) from news & social text, "
    "plus sentiment-tilted index weights (Module A). Illustrative prototype - not investment advice.",
)
store = SignalStore()
_engines: dict[bool, RiskEngine] = {}


class Signal(BaseModel):
    signal_id: str
    doc_id: str
    timestamp: str
    ticker: str
    company: str
    source: str
    source_type: str
    sentiment_score: float = Field(ge=-1, le=1)
    sentiment_label: str
    sentiment_confidence: float
    event_type: str
    event_confidence: float
    impact_score: float = Field(ge=1, le=10)
    headline: str
    url: str | None = None


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=3, examples=["Pfizer shares slide after FDA rejects new cancer drug"])
    source_type: Literal["news", "social"] = "news"
    fast: bool = Field(False, description="use VADER instead of FinBERT")


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records")) if not df.empty else []


def _default_mode() -> str | None:
    df = store.load_signals(limit=1)
    return None if df.empty else df.iloc[0]["run_mode"]


@app.get("/health")
def health() -> dict:
    with store.connect() as con:
        n = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    return {"status": "ok", "signals": n, "time": now_utc().isoformat()}


@app.get("/tickers")
def tickers() -> list[dict]:
    return [{"ticker": t, **{k: v for k, v in m.items()}} for t, m in load_config()["universe"].items()]


@app.get("/signals", response_model=list[Signal])
def signals(
    ticker: str | None = None,
    event_type: str | None = None,
    since: datetime | None = Query(None, description="ISO timestamp, e.g. 2022-07-01T00:00:00Z"),
    min_impact: float | None = Query(None, ge=1, le=10),
    mode: str | None = Query(None, description="sample | historical | live"),
    limit: int = Query(100, ge=1, le=5000),
):
    since_s = pd.Timestamp(since).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") if since and since.tzinfo else (
        pd.Timestamp(since).strftime("%Y-%m-%dT%H:%M:%SZ") if since else None)
    df = store.load_signals(ticker=ticker.upper() if ticker else None, event_type=event_type, since=since_s,
                            min_impact=min_impact, run_mode=mode, limit=limit)
    return _records(df)


@app.get("/signals/latest")
def signals_latest(mode: str | None = Query(None, description="defaults to the most recent run")) -> dict:
    """Decayed, impact-weighted sentiment per ticker as of the newest signal in that run."""
    mode = mode or _default_mode()
    df = store.load_signals(run_mode=mode)
    if df.empty:
        raise HTTPException(404, "No signals stored yet - run run_pipeline.py")
    sig = prepare(df)
    as_of = sig["ts"].max()
    tks = list(load_config()["universe"])
    agg, drivers = aggregate_at(sig, as_of, tks, top_n=1)
    counts = sig[sig["ts"] >= as_of - pd.Timedelta(hours=24)].groupby("ticker").size()
    return {
        "mode": mode,
        "as_of": as_of.isoformat(),
        "tickers": [
            {"ticker": t, "agg_sentiment": round(float(agg[t]), 4), "signals_24h": int(counts.get(t, 0)),
             "top_driver": (drivers.get(t) or [{}])[0].get("headline")}
            for t in tks
        ],
    }


@app.post("/analyze", response_model=list[Signal])
def analyze(req: AnalyzeRequest):
    """Run the engine on any text. Not stored. Returns one signal per detected company (may be empty)."""
    eng = _engines.setdefault(req.fast, RiskEngine(store=None, fast=req.fast, run_mode="adhoc"))
    doc = Document(make_doc_id("api", req.text), "api", req.source_type, now_utc(), req.text, "", None, [])
    return eng.analyze([doc], persist=False)


@app.get("/weights/history")
def weights_history(run_id: str | None = Query(None, description="sample | historical | live")) -> dict:
    w = store.load_weights(run_id)
    if w.empty:
        raise HTTPException(404, "No weight history - run run_backtest.py")
    return {"run_id": w["run_id"].iloc[0], "rows": _records(w.drop(columns="run_id"))}


@app.get("/weights/latest")
def weights_latest(run_id: str | None = None) -> dict:
    w = store.load_weights(run_id)
    if w.empty:
        raise HTTPException(404, "No weight history - run run_backtest.py")
    last = w[w["timestamp"] == w["timestamp"].max()]
    return {
        "run_id": w["run_id"].iloc[0],
        "timestamp": last["timestamp"].iloc[0],
        "weights": {r.ticker: {"weight": round(r.weight, 6), "agg_sentiment": round(r.agg_sentiment, 4)}
                    for r in last.itertuples()},
    }
