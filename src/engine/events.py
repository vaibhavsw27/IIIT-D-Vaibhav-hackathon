"""Rule-based event classifier (explainable, fast).

Each category has weighted regex patterns. The category with the highest total
weight wins; confidence = winner's share of all matched weight, scaled down when
the evidence is thin. No match -> "Other".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# (pattern, weight). Patterns are matched case-insensitively on word boundaries.
RULES: dict[str, list[tuple[str, float]]] = {
    "Credit Event": [
        (r"credit rating", 3), (r"(moody'?s|fitch|s&p)\b.{0,40}(downgrad|upgrad|cut|outlook|rating)", 3),
        (r"downgrad\w*", 1.5), (r"default(s|ed)?", 2.5), (r"bankrupt\w*", 3), (r"chapter 11", 3),
        (r"insolven\w*", 3), (r"debt (load|burden|restructuring)", 2), (r"refinanc\w*", 1.5),
        (r"junk (status|bond)", 2.5), (r"bond(s)? (slump|sell-?off|yield)", 1.5), (r"liquidity (crisis|crunch)", 2.5),
        (r"debt", 1), (r"credit", 0.5),
    ],
    "Merger/Acquisition": [
        (r"acquir\w*", 2.5), (r"acquisition", 2.5), (r"merger", 3), (r"merg(e|es|ing)", 2),
        (r"takeover", 3), (r"buyout", 3), (r"deal to buy", 3), (r"to buy\b.{0,30}\b(for|in) \$", 2),
        (r"tender offer", 3), (r"spin-?off", 2), (r"divest\w*", 2), (r"stake in", 1.5),
    ],
    "Earnings": [
        (r"earnings", 2.5), (r"quarterly (results|profit|revenue|loss)", 3), (r"(beats?|miss(es|ed)?) (estimates|expectations)", 3),
        (r"guidance", 2), (r"outlook", 1), (r"revenue", 1.5), (r"profit", 1.5), (r"\beps\b", 2.5),
        (r"(q[1-4]|first|second|third|fourth)[- ]quarter", 2), (r"net (income|loss)", 2), (r"margins?", 1),
        (r"dividend", 1.5), (r"buyback", 1.5), (r"sales (rose|fell|grew|jumped|declined)", 2),
    ],
    "Product Launch": [
        (r"launch\w*", 2.5), (r"unveil\w*", 3), (r"introduc\w*", 1.5), (r"new (product|model|phone|chip|device|feature|service)", 2.5),
        (r"pre-?orders?", 2.5), (r"rolls? out", 2), (r"release[sd]?\b", 1), (r"announce[sd]? new", 2),
        (r"(fda|regulatory) approv\w* (for|of) (its|new)", 1.5), (r"next-generation", 2),
    ],
    "Regulatory/Legal": [
        (r"lawsuit", 3), (r"sue[sd]?\b", 2.5), (r"antitrust", 3), (r"regulator\w*", 2), (r"\bsec\b", 2),
        (r"probe", 2.5), (r"investigation", 2), (r"fine[sd]?\b", 2), (r"penalt\w*", 2), (r"settle(ment|s|d)?", 2),
        (r"court", 2), (r"judge", 2), (r"ftc|doj|justice department", 3), (r"compliance", 1),
        (r"privacy (violation|breach)", 2.5), (r"ban(ned|s)?\b", 1.5), (r"legislation|lawmakers", 1.5),
        (r"fda (approv|reject)\w*", 2),
    ],
    "Management Change": [
        (r"\b(ceo|cfo|coo|cto|chairman|chief executive|chief financial)\b", 2),
        (r"resign\w*", 3), (r"steps? down", 3), (r"appoint\w*", 3), (r"names? new", 2.5), (r"successor", 2.5),
        (r"fired|ousted|departure", 2.5), (r"hires? .{0,20}(as|to lead)", 2), (r"leadership (change|shake-?up)", 3),
        (r"board (of directors )?(member|seat)", 1.5),
    ],
    "Supply Chain/Operations": [
        (r"supply chain", 3), (r"shortage", 2.5), (r"production", 1.5), (r"factory|plant", 1.5),
        (r"(halt|suspend|shut)\w*", 2), (r"shutdown", 2), (r"recall\w*", 3), (r"outage", 2.5),
        (r"strike|walkout|union", 2), (r"layoffs?|job cuts", 2), (r"capacity", 1.5), (r"logistics|shipping delays?", 2),
        (r"cyber ?attack|data breach|hack(ed)?", 2.5), (r"bottleneck\w*", 2.5),
    ],
    "Macroeconomic": [
        (r"inflation", 3), (r"interest rates?", 3), (r"rate (hike|cut)s?", 3), (r"\bfed\b|federal reserve", 3),
        (r"recession", 3), (r"\bgdp\b", 3), (r"unemployment|jobs report|payrolls", 3), (r"\bcpi\b", 3),
        (r"basis points", 1.5), (r"treasury yields?", 2), (r"economy|economic", 1.5), (r"consumer spending", 2),
        (r"dollar (strength|weakness)|strong dollar", 2), (r"oil prices?", 1.5),
    ],
    "Geopolitical": [
        (r"\bwar\b", 3), (r"sanction\w*", 3), (r"tariff\w*", 3), (r"trade (war|tensions?|dispute)", 3),
        (r"geopolitic\w*", 3), (r"taiwan|ukraine|russia|china|middle east|iran|north korea", 1.5),
        (r"export (ban|controls?|restrictions?)", 3), (r"invasion|conflict|military", 2.5), (r"election", 1.5),
        (r"embargo", 3), (r"tensions?", 1),
    ],
}

_COMPILED = {
    cat: [(re.compile(r"(?<!\w)" + p, re.IGNORECASE), w) for p, w in rules] for cat, rules in RULES.items()
}


@dataclass
class EventResult:
    event_type: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    matched: list[str] = field(default_factory=list)


def classify_event(text: str) -> EventResult:
    """Return the best-matching event category for the text."""
    scores: dict[str, float] = {}
    matched: list[str] = []
    for cat, rules in _COMPILED.items():
        s = 0.0
        for pat, w in rules:
            m = pat.search(text or "")
            if m:
                s += w
                matched.append(f"{cat}:{m.group(0).lower()}")
        if s:
            scores[cat] = s
    if not scores:
        return EventResult("Other", 0.3, {}, [])
    best = max(scores, key=scores.get)
    share = scores[best] / sum(scores.values())
    strength = min(1.0, scores[best] / 4.0)          # >= 4 points of evidence = full strength
    conf = round(0.3 + 0.7 * share * strength, 3)
    return EventResult(best, conf, scores, [m for m in matched if m.startswith(best)])
