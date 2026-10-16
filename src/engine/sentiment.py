"""Sentiment scoring: FinBERT (primary) with a VADER fallback.

score = P(positive) - P(negative), in [-1, 1].
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from src.config import load_config

log = logging.getLogger(__name__)

LABELS = ("positive", "negative", "neutral")


@dataclass
class SentimentResult:
    score: float        # -1..1
    label: str          # positive | negative | neutral
    confidence: float   # 0..1
    model: str


def _label_from_score(score: float, neutral_band: float = 0.05) -> str:
    if score > neutral_band:
        return "positive"
    if score < -neutral_band:
        return "negative"
    return "neutral"


class VaderScorer:
    """Fast lexicon model, used for --fast runs or when torch is unavailable."""

    name = "vader"

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._sia = SentimentIntensityAnalyzer()

    def score(self, texts: list[str]) -> list[SentimentResult]:
        out = []
        for t in texts:
            c = float(self._sia.polarity_scores(t)["compound"])
            out.append(SentimentResult(round(c, 4), _label_from_score(c), round(abs(c), 4), self.name))
        return out


class FinBertScorer:
    """ProsusAI/finbert on CPU, batched."""

    name = "finbert"

    def __init__(self, model_name: str, batch_size: int = 16, max_length: int = 512) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        log.info("Loading FinBERT model %s (CPU)...", model_name)
        try:  # use the local cache first so the demo works offline
            self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        except OSError:
            log.info("Model not cached yet - downloading (~440 MB, one time)")
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.batch_size = batch_size
        self.max_length = max_length
        idx = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.order = [idx[label] for label in LABELS]

    def score(self, texts: list[str]) -> list[SentimentResult]:
        torch = self._torch
        out: list[SentimentResult] = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for b, i in enumerate(range(0, len(texts), self.batch_size)):
            batch = texts[i : i + self.batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            with torch.inference_mode():
                probs = torch.softmax(self.model(**enc).logits, dim=-1).numpy()
            for p in probs:
                pos, neg, neu = (float(p[j]) for j in self.order)
                trio = (pos, neg, neu)
                best = max(range(3), key=lambda j: trio[j])
                out.append(SentimentResult(round(pos - neg, 4), LABELS[best], round(trio[best], 4), self.name))
            if n_batches > 20 and b % 20 == 0:
                log.info("FinBERT batch %d / %d", b + 1, n_batches)
        return out


@lru_cache(maxsize=2)
def get_scorer(fast: bool = False):
    """Singleton scorer. fast=True, or any FinBERT load failure, gives VADER."""
    if not fast:
        cfg = load_config()["sentiment"]
        try:
            return FinBertScorer(cfg["model"], cfg["batch_size"], cfg["max_length"])
        except Exception as exc:  # torch missing, no internet on first download, etc.
            log.warning("FinBERT unavailable (%s) - falling back to VADER", exc)
    return VaderScorer()


def score_texts(texts: list[str], fast: bool = False) -> list[SentimentResult]:
    if not texts:
        return []
    return get_scorer(fast).score(texts)
