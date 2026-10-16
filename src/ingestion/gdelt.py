"""Live news from the GDELT DOC 2.0 API (no key required).

Companies are batched into a few OR-queries (English only) because GDELT
throttles hard (about one request every 5+ seconds per IP). Calls are spaced
out, retried with backoff, and the whole fetch has a time budget. Any failure
returns what was collected so far - the live demo never crashes.
"""
from __future__ import annotations

import logging
import time

import requests

from src.config import load_config
from src.ingestion.base import Document, make_doc_id, to_utc

log = logging.getLogger(__name__)

URL = "https://api.gdeltproject.org/api/v2/doc/doc"
HEADERS = {"User-Agent": "risk-signal-engine/1.0 (hackathon research prototype)"}


def _names(meta: dict) -> list[str]:
    names = [a for a in meta["aliases"] if not a.startswith("$") and not a.isupper() and len(a) > 3][:1]
    return names or [meta["company"]]


def _query_for(metas: list[dict]) -> str:
    names = [n for m in metas for n in _names(m)]
    terms = " OR ".join(f'"{n}"' for n in names)
    return f"({terms}) sourcelang:english" if len(names) > 1 else f"{terms} sourcelang:english"


def _get(params: dict, timeout: int, retries: int) -> dict | None:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(URL, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 429 or "limit requests" in r.text[:200].lower():
                log.warning("GDELT rate limit hit, backing off %ds", 15 * attempt)
                time.sleep(15 * attempt)
                continue
            r.raise_for_status()
            if not r.text.strip():
                return {"articles": []}
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("GDELT request failed (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(2 * attempt)
    return None


def fetch_gdelt(tickers: list[str] | None = None, group_size: int = 5, budget_s: float = 240) -> list[Document]:
    cfg = load_config()
    live = cfg["live"]
    universe = cfg["universe"]
    tickers = tickers or list(universe)
    groups = [tickers[i : i + group_size] for i in range(0, len(tickers), group_size)]
    docs: list[Document] = []
    t0 = time.monotonic()
    for i, group in enumerate(groups):
        if time.monotonic() - t0 > budget_s:
            log.warning("GDELT time budget used up - skipping %d remaining queries", len(groups) - i)
            break
        if i:
            time.sleep(max(5.0, float(live.get("sleep_between_calls", 5))))
        params = {
            "query": _query_for([universe[t] for t in group]),
            "mode": "artlist",
            "format": "json",
            "maxrecords": live.get("gdelt_maxrecords", 250),
            "timespan": live.get("gdelt_timespan", "1d"),
            "sort": "datedesc",
        }
        data = _get(params, live.get("request_timeout", 20), live.get("retries", 3))
        if data is None:
            continue
        arts = data.get("articles", []) or []
        for a in arts:
            title = (a.get("title") or "").strip()
            url = a.get("url")
            if not title or not url:
                continue
            try:
                ts = to_utc(a.get("seendate", "").replace("T", " ").replace("Z", "") or None)
            except Exception:
                continue
            docs.append(Document(
                doc_id=make_doc_id("gdelt", url),
                source="gdelt",
                source_type="news",
                timestamp=ts,
                title=title,
                text="",
                url=url,
                tickers_hint=[],   # the entity matcher decides; the query is only a pre-filter
            ))
        log.info("GDELT %s: %d articles", "/".join(group), len(arts))
    log.info("GDELT total: %d articles in %.0fs", len(docs), time.monotonic() - t0)
    return docs
