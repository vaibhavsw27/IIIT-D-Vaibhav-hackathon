"""SQLite storage for signals, processed docs and index weights, plus JSON/CSV export."""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from src.config import path

log = logging.getLogger(__name__)

SIGNAL_COLUMNS = [
    "signal_id", "doc_id", "timestamp", "ticker", "company", "source", "source_type",
    "sentiment_score", "sentiment_label", "sentiment_confidence", "sentiment_model",
    "event_type", "event_confidence", "impact_score", "headline", "url", "run_mode",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    company TEXT,
    source TEXT,
    source_type TEXT,
    sentiment_score REAL,
    sentiment_label TEXT,
    sentiment_confidence REAL,
    sentiment_model TEXT,
    event_type TEXT,
    event_confidence REAL,
    impact_score REAL,
    headline TEXT,
    url TEXT,
    run_mode TEXT
);
CREATE INDEX IF NOT EXISTS ix_signals_ticker_ts ON signals(ticker, timestamp);
CREATE INDEX IF NOT EXISTS ix_signals_mode ON signals(run_mode);

-- every document ever scored, so the pipeline never re-scores the same text
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT,
    timestamp TEXT,
    n_signals INTEGER
);

CREATE TABLE IF NOT EXISTS weights (
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    weight REAL,
    agg_sentiment REAL,
    PRIMARY KEY (run_id, timestamp, ticker)
);

CREATE TABLE IF NOT EXISTS rebalance_log (
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    old_weight REAL,
    new_weight REAL,
    agg_sentiment REAL,
    reason TEXT,
    PRIMARY KEY (run_id, timestamp, ticker)
);
"""


class SignalStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or path("db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ---------- documents ----------
    def known_doc_ids(self, doc_ids: Iterable[str]) -> set[str]:
        ids = list(doc_ids)
        found: set[str] = set()
        with self.connect() as con:
            for i in range(0, len(ids), 900):
                chunk = ids[i : i + 900]
                q = f"SELECT doc_id FROM documents WHERE doc_id IN ({','.join('?' * len(chunk))})"
                found.update(r[0] for r in con.execute(q, chunk))
        return found

    def mark_docs(self, rows: list[tuple[str, str, str, int]]) -> None:
        with self.connect() as con:
            con.executemany("INSERT OR REPLACE INTO documents VALUES (?,?,?,?)", rows)

    # ---------- signals ----------
    def save_signals(self, signals: list[dict]) -> int:
        if not signals:
            return 0
        cols = ",".join(SIGNAL_COLUMNS)
        qs = ",".join("?" * len(SIGNAL_COLUMNS))
        with self.connect() as con:
            cur = con.executemany(
                f"INSERT OR IGNORE INTO signals ({cols}) VALUES ({qs})",
                [tuple(s.get(c) for c in SIGNAL_COLUMNS) for s in signals],
            )
            return cur.rowcount

    def load_signals(
        self,
        ticker: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
        until: str | None = None,
        min_impact: float | None = None,
        run_mode: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        where, args = [], []
        for col, op, val in [
            ("ticker", "=", ticker), ("event_type", "=", event_type), ("timestamp", ">=", since),
            ("timestamp", "<=", until), ("impact_score", ">=", min_impact), ("run_mode", "=", run_mode),
        ]:
            if val is not None and val != "":
                where.append(f"{col} {op} ?")
                args.append(val)
        q = "SELECT * FROM signals"
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY timestamp DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self.connect() as con:
            df = pd.read_sql_query(q, con, params=args)
        return df

    def ticker_timestamps(self) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query("SELECT ticker, timestamp FROM signals", con)

    def clear(self, run_mode: str | None = None) -> None:
        """Remove signals/docs (optionally only for one run_mode)."""
        with self.connect() as con:
            if run_mode:
                con.execute(
                    "DELETE FROM documents WHERE doc_id IN (SELECT doc_id FROM signals WHERE run_mode=?)",
                    (run_mode,),
                )
                con.execute("DELETE FROM signals WHERE run_mode=?", (run_mode,))
            else:
                con.execute("DELETE FROM signals")
                con.execute("DELETE FROM documents")

    # ---------- weights ----------
    def save_weights(self, run_id: str, weights: pd.DataFrame, log_rows: pd.DataFrame | None = None) -> None:
        """weights: columns [timestamp, ticker, weight, agg_sentiment]."""
        with self.connect() as con:
            con.execute("DELETE FROM weights WHERE run_id=?", (run_id,))
            con.execute("DELETE FROM rebalance_log WHERE run_id=?", (run_id,))
            con.executemany(
                "INSERT INTO weights VALUES (?,?,?,?,?)",
                [(run_id, str(r.timestamp), r.ticker, float(r.weight), float(r.agg_sentiment))
                 for r in weights.itertuples(index=False)],
            )
            if log_rows is not None and len(log_rows):
                con.executemany(
                    "INSERT INTO rebalance_log VALUES (?,?,?,?,?,?,?)",
                    [(run_id, str(r.timestamp), r.ticker, float(r.old_weight), float(r.new_weight),
                      float(r.agg_sentiment), r.reason) for r in log_rows.itertuples(index=False)],
                )

    def append_weights(self, run_id: str, weights: pd.DataFrame, log_rows: pd.DataFrame | None = None) -> None:
        """Add one rebalancing step without touching earlier steps (used by live mode)."""
        with self.connect() as con:
            con.executemany(
                "INSERT OR REPLACE INTO weights VALUES (?,?,?,?,?)",
                [(run_id, str(r.timestamp), r.ticker, float(r.weight), float(r.agg_sentiment))
                 for r in weights.itertuples(index=False)],
            )
            if log_rows is not None and len(log_rows):
                con.executemany(
                    "INSERT OR REPLACE INTO rebalance_log VALUES (?,?,?,?,?,?,?)",
                    [(run_id, str(r.timestamp), r.ticker, float(r.old_weight), float(r.new_weight),
                      float(r.agg_sentiment), r.reason) for r in log_rows.itertuples(index=False)],
                )

    def load_weights(self, run_id: str | None = None) -> pd.DataFrame:
        with self.connect() as con:
            if run_id is None:
                row = con.execute("SELECT run_id FROM weights ORDER BY timestamp DESC LIMIT 1").fetchone()
                if not row:
                    return pd.DataFrame(columns=["run_id", "timestamp", "ticker", "weight", "agg_sentiment"])
                run_id = row[0]
            return pd.read_sql_query("SELECT * FROM weights WHERE run_id=? ORDER BY timestamp", con, params=[run_id])

    def load_rebalance_log(self, run_id: str) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query(
                "SELECT * FROM rebalance_log WHERE run_id=? ORDER BY timestamp", con, params=[run_id]
            )

    def weight_runs(self) -> list[str]:
        with self.connect() as con:
            return [r[0] for r in con.execute("SELECT DISTINCT run_id FROM weights")]

    # ---------- export ----------
    def export(self, json_path: Path | None = None, csv_path: Path | None = None) -> int:
        df = self.load_signals()
        df = df.sort_values("timestamp")
        json_path = Path(json_path or path("signals_json"))
        csv_path = Path(csv_path or path("signals_csv"))
        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(json.loads(df.to_json(orient="records")), fh, indent=1, ensure_ascii=False)
        log.info("Exported %d signals to %s and %s", len(df), json_path.name, csv_path.name)
        return len(df)
