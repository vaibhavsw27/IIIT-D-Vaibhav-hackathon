"""Evaluate FinBERT and VADER on the labelled Kaggle financial-news set.

  python scripts/evaluate_sentiment.py            # full set (~4.8k sentences, a few minutes on CPU)
  python scripts/evaluate_sentiment.py --n 1000   # quick stratified subsample

Outputs: printed accuracy / macro-F1 / confusion matrix, and
docs/screenshots/sentiment_eval.png + data/sentiment_eval.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402

from src.config import setup_logging  # noqa: E402
from src.engine.sentiment import get_scorer  # noqa: E402
from src.ingestion.kaggle_news import load_labelled_news  # noqa: E402

LABELS = ["negative", "neutral", "positive"]


def vader_label(score: float, band: float = 0.05) -> str:
    return "positive" if score > band else "negative" if score < -band else "neutral"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=0, help="stratified subsample size (0 = all)")
    args = p.parse_args()
    setup_logging()

    df = load_labelled_news()
    if df.empty:
        print("data/raw/all-data.csv not found. Download the Kaggle dataset first (see data/README.md).")
        return
    if args.n and args.n < len(df):
        df = df.groupby("label", group_keys=False).apply(lambda g: g.sample(int(round(args.n * len(g) / len(df))), random_state=42))
    texts = df["text"].tolist()
    print(f"Evaluating on {len(df)} labelled sentences: {df['label'].value_counts().to_dict()}")

    rows, cms = [], {}
    for name, fast in [("FinBERT", False), ("VADER", True)]:
        res = get_scorer(fast).score(texts)
        pred = [r.label for r in res] if not fast else [vader_label(r.score) for r in res]
        acc = accuracy_score(df["label"], pred)
        f1 = f1_score(df["label"], pred, average="macro")
        cms[name] = confusion_matrix(df["label"], pred, labels=LABELS)
        rows.append({"model": name, "n": len(df), "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})
        print(f"\n=== {name} ===  accuracy={acc:.3f}  macro-F1={f1:.3f}")
        print(classification_report(df["label"], pred, labels=LABELS, digits=3))
        print(pd.DataFrame(cms[name], index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS]))

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "sentiment_eval.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    for ax, (name, cm), r in zip(axes, cms.items(), rows):
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.55 else "#0b0b0b", fontsize=9)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}: acc {r['accuracy']:.1%}, macro-F1 {r['macro_f1']:.3f}")
    fig.suptitle(f"Sentiment model evaluation - Kaggle Financial News (n={len(df)})")
    fig.tight_layout()
    path = ROOT / "docs" / "screenshots" / "sentiment_eval.png"
    fig.savefig(path)
    print(f"\nSaved {path.relative_to(ROOT)} and data/sentiment_eval.csv")


if __name__ == "__main__":
    np.random.seed(0)
    main()
