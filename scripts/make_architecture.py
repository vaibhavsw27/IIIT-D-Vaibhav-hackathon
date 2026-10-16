"""Draw docs/architecture.png with matplotlib (no external tools needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INK, SUB, LINE = "#0b0b0b", "#52514e", "#898781"
FILL = {"src": "#e9f1fc", "eng": "#fdeee7", "store": "#e6f6ef", "out": "#f2effc"}
EDGE = {"src": "#2a78d6", "eng": "#eb6834", "store": "#1baf7a", "out": "#6250d6"}


def box(ax, x, y, w, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.6))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=10.5, weight="bold", color=INK)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.058 - i * 0.03, ln, ha="center", va="top", fontsize=8.3, color=SUB)


def arrow(ax, a, b, label=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=LINE, lw=1.4,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.014, label, ha="center", fontsize=7.6, color=SUB, style="italic")


def main() -> None:
    fig = plt.figure(figsize=(15, 8.2), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.965, "AI Risk Signal Engine + Tactical Index Rebalancer (Module A) - Architecture",
            ha="center", fontsize=15, weight="bold", color=INK)

    # --- sources column
    sx, sw, sh = 0.02, 0.17, 0.125
    sources = [
        ("GDELT DOC 2.0 API", ["live news, no key", "batched OR-queries, backoff"]),
        ("NewsAPI (optional)", ["live news, needs key", "skipped cleanly if absent"]),
        ("Kaggle Financial News", ["4.8k labelled sentences", "model evaluation"]),
        ("Kaggle Stock Tweets", ["social media, 2021-22", "historical backtest"]),
        ("Synthetic sample CSVs", ["news + tweets, offline", "always-works demo"]),
    ]
    ys = [0.79, 0.645, 0.50, 0.355, 0.21]
    for (t, l), y in zip(sources, ys):
        box(ax, sx, y, sw, sh, t, l, "src")
    ax.text(sx + sw / 2, 0.935, "1 . SOURCES", ha="center", fontsize=9, color=EDGE["src"], weight="bold")

    # --- ingestion
    ix, iw = 0.235, 0.13
    box(ax, ix, 0.43, iw, 0.2, "Ingestion", ["common Document", "schema (UTC time,", "source, text, url,", "ticker hints)"], "src")
    for y in ys:
        arrow(ax, (sx + sw, y + sh / 2), (ix, 0.53))

    # --- engine
    ex, ew = 0.41, 0.22
    ax.add_patch(FancyBboxPatch((ex - 0.01, 0.13), ew + 0.02, 0.79, boxstyle="round,pad=0.006,rounding_size=0.015",
                                fc="none", ec=EDGE["eng"], lw=1.2, ls="--"))
    ax.text(ex + ew / 2, 0.935, "2 . RISK ENGINE  (RiskEngine.analyze)", ha="center", fontsize=9, color=EDGE["eng"], weight="bold")
    steps = [
        ("Preprocess", ["clean, emoji map, language filter", "dedupe + skip already-scored docs"]),
        ("Entity linking", ["alias dictionary -> 15 tickers", "cashtags, context exclusions"]),
        ("Sentiment", ["FinBERT (CPU, batched)  P(pos)-P(neg)", "VADER fallback / --fast"]),
        ("Event classification", ["weighted keyword rules, 10 types", "confidence = evidence share"]),
        ("Impact score 1-10", ["severity x (0.5+|s|) x source wt", "x (1 + 0.1 log1p(24h buzz))"]),
    ]
    eys = [0.775, 0.63, 0.485, 0.34, 0.195]
    for (t, l), y in zip(steps, eys):
        box(ax, ex, y, ew, 0.12, t, l, "eng")
    for a, b in zip(eys[:-1], eys[1:]):
        arrow(ax, (ex + ew / 2, a), (ex + ew / 2, b + 0.12))
    arrow(ax, (ix + iw, 0.53), (ex, 0.835))

    # --- store
    stx, stw = 0.675, 0.13
    box(ax, stx, 0.43, stw, 0.2, "Signal store", ["SQLite signals.db", "signals / documents /", "weights / rebalance_log", "+ JSON & CSV export"], "store")
    arrow(ax, (ex + ew, 0.255), (stx, 0.48))
    ax.text(0.66, 0.36, "signal JSON", ha="center", fontsize=7.6, color=SUB, style="italic")
    ax.text(stx + stw / 2, 0.935, "3 . STORAGE", ha="center", fontsize=9, color=EDGE["store"], weight="bold")

    # --- outputs
    ox, ow, oh = 0.845, 0.145, 0.165
    ax.text(ox + ow / 2, 0.935, "4 . CONSUMERS", ha="center", fontsize=9, color=EDGE["out"], weight="bold")
    box(ax, ox, 0.735, ow, oh, "FastAPI", ["/signals  /signals/latest", "/analyze  /weights/*", "Swagger at /docs"], "out")
    box(ax, ox, 0.47, ow, 0.2, "Module A Rebalancer", ["decayed sentiment agg.", "w = w0 * exp(k * s)", "bounds 2-15%, 3% cap", "backtest vs EW, S&P 500"], "out")
    box(ax, ox, 0.235, ow, oh, "Streamlit dashboard", ["weights over time, replay", "why-it-changed, feed,", "heatmap, try-it-live"], "out")
    box(ax, stx, 0.17, stw, 0.13, "yfinance prices", ["daily closes, cached", "data/raw/prices.csv"], "store")
    arrow(ax, (stx + stw, 0.58), (ox, 0.81))
    arrow(ax, (stx + stw, 0.53), (ox, 0.57))
    arrow(ax, (stx + stw, 0.47), (ox, 0.32))
    arrow(ax, (stx + stw, 0.25), (ox, 0.50), "prices")
    arrow(ax, (ox + ow / 2, 0.47), (ox + ow / 2, 0.40), "weights")

    ax.text(0.5, 0.02, "No look-ahead: weights set from signals up to day t's close earn day t+1's return.   "
            "Everything runs offline on cached data; live sources fall back gracefully.",
            ha="center", fontsize=8.5, color=SUB)
    out = ROOT / "docs" / "architecture.png"
    fig.savefig(out, facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
