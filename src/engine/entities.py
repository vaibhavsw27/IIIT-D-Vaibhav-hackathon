"""Map free text to tickers in our 15-stock universe using an alias dictionary."""
from __future__ import annotations

import re
from functools import lru_cache

from src.config import load_config


@lru_cache(maxsize=1)
def _patterns() -> list[tuple[str, re.Pattern]]:
    """One compiled regex per ticker.

    Company names / people are matched case-insensitively on word boundaries.
    Cashtags ($AAPL) and bare upper-case tickers (AAPL) are matched case-sensitively
    so that e.g. the word "ko" or "dis" never matches.
    """
    out = []
    for ticker, meta in load_config()["universe"].items():
        ci, cs = [], []
        for alias in meta["aliases"]:
            esc = re.escape(alias)
            if alias.startswith("$"):
                cs.append(esc + r"\b")
            elif alias.isupper() and len(alias) <= 5:
                cs.append(r"(?<![\w$])" + esc + r"\b")
            else:
                ci.append(r"(?<!\w)" + esc + r"(?!\w)")
        parts = []
        if ci:
            parts.append("(?i:" + "|".join(sorted(ci, key=len, reverse=True)) + ")")
        if cs:
            parts.append("|".join(cs))
        out.append((ticker, re.compile("|".join(parts))))
    return out


def find_tickers(text: str, hint: list[str] | None = None) -> list[str]:
    """Return tickers mentioned in text (plus any dataset-provided hints), order-preserving."""
    cfg = load_config()
    remap = cfg.get("ticker_remap", {})
    universe = cfg["universe"]
    found: list[str] = []
    for t in hint or []:
        t = remap.get(t, t)
        if t in universe and t not in found:
            found.append(t)
    for ticker, pat in _patterns():
        if ticker not in found and pat.search(text or ""):
            found.append(ticker)
    return found


def company_name(ticker: str) -> str:
    return load_config()["universe"].get(ticker, {}).get("company", ticker)
