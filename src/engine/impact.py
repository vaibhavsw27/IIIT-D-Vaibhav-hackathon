"""Explainable impact score in [1, 10].

raw = base_severity[event_type]
      * (0.5 + |sentiment_score|)
      * source_weight[source_type]
      * (1 + buzz_coef * log1p(mentions_24h))
impact = clip(round(raw, 1), 1, 10)
"""
from __future__ import annotations

import math

from src.config import load_config


def impact_score(event_type: str, sentiment_score: float, source_type: str, mentions_24h: int = 0) -> float:
    cfg = load_config()["impact"]
    base = cfg["base_severity"].get(event_type, cfg["base_severity"]["Other"])
    src_w = cfg["source_weight"].get(source_type, 1.0)
    buzz = 1 + cfg["buzz_coef"] * math.log1p(max(0, mentions_24h))
    raw = base * (0.5 + abs(sentiment_score)) * src_w * buzz
    return float(min(cfg["max"], max(cfg["min"], round(raw, 1))))
