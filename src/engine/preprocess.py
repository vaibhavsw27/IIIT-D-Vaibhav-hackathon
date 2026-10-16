"""Text cleaning, de-duplication and a light English-language filter."""
from __future__ import annotations

import html
import re
import unicodedata

from src.ingestion.base import Document

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")
HASHTAG_RE = re.compile(r"#(\w+)")

# A few emojis carry clear market tone in finance tweets; map them to words.
EMOJI_MAP = {
    "\U0001F680": " rocket bullish ",   # rocket
    "\U0001F4C8": " rising ",           # chart up
    "\U0001F4C9": " falling ",          # chart down
    "\U0001F525": " hot ",              # fire
    "\U0001F48E": " diamond hands ",    # gem
    "\U0001F43B": " bearish ",          # bear
    "\U0001F402": " bullish ",          # ox
    "\U0001F480": " terrible ",         # skull
    "\U0001F62D": " sad ",              # crying
    "\U0001F911": " profit ",           # money face
}


def clean_text(text: str, social: bool = False) -> str:
    """Normalise whitespace/HTML; for tweets strip URLs and @mentions, map emojis.

    $cashtags are kept so the entity matcher can use them.
    """
    if not text:
        return ""
    text = html.unescape(str(text))
    text = URL_RE.sub(" ", text)
    if social:
        text = MENTION_RE.sub(" ", text)
        for emo, word in EMOJI_MAP.items():
            text = text.replace(emo, word)
        text = HASHTAG_RE.sub(r"\1", text)
        # drop any remaining emoji / pictographs / variation selectors
        text = "".join(ch for ch in text if unicodedata.category(ch) not in ("So", "Cs", "Mn"))
    return WS_RE.sub(" ", text).strip()


def is_english(text: str, min_ascii_ratio: float = 0.85) -> bool:
    """Cheap language filter: mostly-ASCII letters => treat as English."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(c.isascii() for c in letters) / len(letters) >= min_ascii_ratio


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())[:200]


def preprocess(docs: list[Document]) -> list[Document]:
    """Clean text in place, drop empties / non-English, dedupe by doc_id and normalised text."""
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    out: list[Document] = []
    for d in docs:
        social = d.source_type == "social"
        d.title = clean_text(d.title, social)
        d.text = clean_text(d.text, social)
        body = d.full_text
        if len(body) < 8 or not is_english(body):
            continue
        key = _norm_key(body) + "|" + ",".join(d.tickers_hint)
        if d.doc_id in seen_ids or key in seen_text:
            continue
        seen_ids.add(d.doc_id)
        seen_text.add(key)
        out.append(d)
    return out
