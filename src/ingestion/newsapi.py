"""Optional live news from NewsAPI.org (/v2/everything). Skipped cleanly when NEWSAPI_KEY is not set."""
from __future__ import annotations

import logging
import os

import requests

from src.config import load_config
from src.ingestion.base import Document, make_doc_id, to_utc

log = logging.getLogger(__name__)
URL = "https://newsapi.org/v2/everything"


def fetch_newsapi(tickers: list[str] | None = None) -> list[Document]:
    key = os.getenv("NEWSAPI_KEY", "").strip()
    if not key:
        log.info("NEWSAPI_KEY not set - skipping NewsAPI (optional source)")
        return []
    cfg = load_config()
    universe = cfg["universe"]
    tickers = tickers or list(universe)
    # One combined query keeps us well inside the free tier's 100 requests/day.
    names = [universe[t]["aliases"][0] for t in tickers]
    params = {"q": " OR ".join(f'"{n}"' for n in names), "language": "en", "sortBy": "publishedAt", "pageSize": 100}
    try:
        r = requests.get(URL, params=params, headers={"X-Api-Key": key}, timeout=cfg["live"].get("request_timeout", 20))
        r.raise_for_status()
        arts = r.json().get("articles", [])
    except (requests.RequestException, ValueError) as exc:
        log.warning("NewsAPI request failed: %s", exc)
        return []
    docs = [
        Document(
            doc_id=make_doc_id("newsapi", a["url"]),
            source="newsapi",
            source_type="news",
            timestamp=to_utc(a["publishedAt"]),
            title=a.get("title") or "",
            text=a.get("description") or "",
            url=a["url"],
        )
        for a in arts
        if a.get("url") and a.get("title") and a.get("publishedAt")
    ]
    log.info("NewsAPI: %d articles", len(docs))
    return docs
