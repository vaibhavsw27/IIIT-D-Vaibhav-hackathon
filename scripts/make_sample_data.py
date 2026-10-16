"""Generate the synthetic offline sample datasets in data/sample/.

All text is synthetic (template-based) and does NOT describe real events.
Each ticker gets a slowly drifting "story" tone so the rebalancer has
something to react to. Run:  python scripts/make_sample_data.py
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402

SEED = 7
START = datetime(2022, 7, 1, tzinfo=timezone.utc)
DAYS = 60
N_NEWS = 210
N_TWEETS = 210

# {c} = company name, {p} = product / person alias
NEWS_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "Earnings": {
        "pos": ["{c} beats quarterly earnings estimates as revenue jumps 18%",
                "{c} raises full-year guidance after record profit",
                "{c} reports strong quarterly results, shares rally after hours"],
        "neg": ["{c} misses earnings expectations, cuts full-year outlook",
                "{c} quarterly profit falls 30% as margins shrink",
                "{c} warns of weak revenue in coming quarter, shares slide"],
    },
    "Credit Event": {
        "pos": ["Moody's upgrades {c} credit rating on strong cash flow",
                "{c} outlook revised to positive by S&P as debt falls"],
        "neg": ["Moody's downgrades {c} credit rating citing rising debt load",
                "S&P cuts {c} outlook to negative amid refinancing concerns",
                "{c} bonds slump as investors fear credit rating downgrade"],
    },
    "Merger/Acquisition": {
        "pos": ["{c} agrees to acquire AI startup in $2 billion deal",
                "{c} completes acquisition that analysts call highly accretive"],
        "neg": ["{c} takeover bid collapses after regulators object to merger",
                "{c} writes down value of failed acquisition"],
    },
    "Regulatory/Legal": {
        "pos": ["Court dismisses antitrust lawsuit against {c}",
                "{c} wins regulatory approval, ending long investigation"],
        "neg": ["{c} faces antitrust lawsuit from Justice Department",
                "EU regulators fine {c} over data privacy violations",
                "{c} hit with SEC probe over accounting practices"],
    },
    "Product Launch": {
        "pos": ["{c} unveils new {p} to strong early demand",
                "{c} launches next-generation {p}, preorders sell out",
                "Analysts praise {c}'s new {p} launch"],
        "neg": ["{c} delays launch of new {p} after testing problems",
                "Reviews of {c}'s new {p} disappoint critics"],
    },
    "Management Change": {
        "pos": ["{c} appoints highly regarded industry veteran as new CFO",
                "{c} names new chief operating officer to drive growth"],
        "neg": ["{c} CEO abruptly resigns, board begins search for successor",
                "{c} CFO steps down amid internal review"],
    },
    "Supply Chain/Operations": {
        "pos": ["{c} says supply chain bottlenecks have eased, production up",
                "{c} opens new factory, boosting capacity by 25%"],
        "neg": ["{c} halts production at key plant due to chip shortage",
                "Factory shutdown and supply chain disruption hit {c} output",
                "{c} recalls products over safety defect"],
    },
    "Macroeconomic": {
        "pos": ["Cooling inflation data lifts {c} and other large caps",
                "{c} gains as Fed signals slower pace of interest rate hikes"],
        "neg": ["{c} falls as Fed raises interest rates by 75 basis points",
                "Recession fears weigh on {c} as inflation stays high"],
    },
    "Geopolitical": {
        "pos": ["Easing trade tensions with China boost {c} outlook",
                "{c} benefits as sanctions relief reopens export markets"],
        "neg": ["{c} exposed as US-China tensions escalate over Taiwan",
                "New sanctions and war risk threaten {c} operations abroad"],
    },
    "Other": {
        "pos": ["{c} shares climb as analysts turn more bullish",
                "{c} named among most admired companies this year"],
        "neg": ["{c} shares drift lower in quiet trading",
                "Analysts trim price target on {c}"],
        "neu": ["{c} to present at industry conference next week",
                "{c} schedules annual shareholder meeting"],
    },
}

TWEET_TEMPLATES = {
    "pos": ["${t} looking strong today, loading more 🚀", "Great quarter from ${t}, holding long 📈",
            "${t} breakout incoming, bullish on {c}", "Just bought more ${t}. {p} is a game changer",
            "{c} crushing it lately. ${t} to the moon", "Love the new {p}, long ${t}",
            "${t} upgrade from analysts, very bullish"],
    "neg": ["${t} dumping hard, sold my position 📉", "Not liking this ${t} chart, bearish",
            "{c} guidance was terrible, ${t} going lower", "${t} lawsuit news is ugly, staying out",
            "Recession + rate hikes = ${t} pain", "Shorting ${t}, {c} is overvalued",
            "${t} supply issues again... disappointing"],
    "neu": ["What do you all think about ${t} this week?", "Watching ${t} into earnings",
            "${t} trading sideways, waiting for direction", "Anyone holding ${t} over the weekend?"],
}

TWEET_TICKER = {"GOOGL": "GOOG"}

PRODUCTS = {
    "AAPL": ["iPhone", "MacBook", "Apple Watch"], "MSFT": ["Azure AI service", "Surface laptop", "Xbox console"],
    "AMZN": ["Echo device", "AWS cloud region", "Prime service"], "GOOGL": ["Pixel phone", "AI search assistant", "cloud TPU"],
    "META": ["VR headset", "Instagram feature", "AI assistant"], "NVDA": ["data-center GPU", "gaming GPU", "AI chip"],
    "TSLA": ["Cybertruck", "Model Y variant", "Full Self-Driving update"], "NFLX": ["ad-supported plan", "gaming service", "original series"],
    "JPM": ["digital banking app", "payments platform", "wealth product"], "BAC": ["mobile banking app", "AI assistant Erica", "credit card"],
    "XOM": ["low-carbon fuel", "carbon capture project", "refinery expansion"], "JNJ": ["cancer drug", "medical device", "vaccine"],
    "PFE": ["RSV vaccine", "oral antiviral pill", "cancer therapy"], "KO": ["zero-sugar drink", "sports drink", "energy drink"],
    "DIS": ["Disney+ bundle", "theme park attraction", "animated film"],
}
SHORT = {"AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "GOOGL": "Alphabet", "META": "Meta Platforms",
         "NVDA": "Nvidia", "TSLA": "Tesla", "NFLX": "Netflix", "JPM": "JPMorgan Chase", "BAC": "Bank of America",
         "XOM": "Exxon Mobil", "JNJ": "Johnson & Johnson", "PFE": "Pfizer", "KO": "Coca-Cola", "DIS": "Disney"}
TWEET_TAILS = ["", "", " imo", " #stocks", " #investing", " today", " this week", " 🤔", " no doubt", " lol",
               " (not advice)", " #markets", " for real", " again"]  # mimic the Kaggle tweets file, which uses GOOG


def _alias(meta: dict, rng: random.Random) -> str:
    plain = [a for a in meta["aliases"] if not a.startswith("$") and not a.isupper()]
    return rng.choice(plain or [meta["company"]])


def _tone(drift: float, rng: random.Random) -> str:
    """Pick pos/neg/neu with a bias given by the ticker's current drift (-1..1)."""
    p_pos = 0.45 + 0.35 * drift
    r = rng.random()
    if r < 0.1:
        return "neu"
    return "pos" if r < 0.1 + 0.9 * p_pos else "neg"


def _drift_path(rng: random.Random) -> list[float]:
    """Random-walk story tone per day, bounded to [-1, 1]."""
    x, out = rng.uniform(-0.3, 0.3), []
    for _ in range(DAYS):
        x = max(-1.0, min(1.0, 0.9 * x + rng.gauss(0, 0.25)))
        out.append(x)
    return out


def main() -> None:
    rng = random.Random(SEED)
    universe = load_config()["universe"]
    tickers = list(universe)
    drifts = {t: _drift_path(rng) for t in tickers}
    events = list(NEWS_TEMPLATES)

    news_rows = []
    for i in range(N_NEWS):
        t = tickers[i % len(tickers)]
        event = events[i % len(events)] if i < 2 * len(events) * 2 else rng.choice(events)
        day = rng.randrange(DAYS)
        ts = START + timedelta(days=day, hours=rng.randint(6, 21), minutes=rng.randint(0, 59))
        tone = _tone(drifts[t][day], rng)
        pool = NEWS_TEMPLATES[event].get(tone) or NEWS_TEMPLATES[event]["pos" if tone != "neg" else "neg"]
        meta = universe[t]
        title = rng.choice(pool).format(c=SHORT[t], p=rng.choice(PRODUCTS[t]))
        hint = t
        # Occasionally mention a second company to exercise multi-ticker extraction
        if rng.random() < 0.08:
            other = rng.choice([x for x in tickers if x != t])
            title += f"; {SHORT[other]} also in focus"
            hint = ""
        news_rows.append({
            "timestamp": ts.isoformat(),
            "source": rng.choice(["Synthetic Wire", "Sample Business News", "Demo Markets Daily"]),
            "title": title,
            "text": "",
            "url": f"https://example.com/sample-news/{i:04d}",
            "tickers": hint,
        })

    tweet_rows = []
    for i in range(N_TWEETS):
        t = tickers[i % len(tickers)]
        day = rng.randrange(DAYS)
        ts = START + timedelta(days=day, hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
        tone = _tone(drifts[t][day], rng)
        meta = universe[t]
        sym = TWEET_TICKER.get(t, t)
        text = rng.choice(TWEET_TEMPLATES[tone]).format(t=sym, c=SHORT[t], p=rng.choice(PRODUCTS[t]))
        text += rng.choice(TWEET_TAILS)
        tweet_rows.append({
            "Date": ts.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "Tweet": text,
            "Stock Name": sym,
            "Company Name": meta["company"],
        })

    out = ROOT / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(news_rows).sort_values("timestamp").to_csv(out / "news_sample.csv", index=False)
    pd.DataFrame(tweet_rows).sort_values("Date").to_csv(out / "tweets_sample.csv", index=False)
    print(f"Wrote {len(news_rows)} news and {len(tweet_rows)} tweets to {out}")


if __name__ == "__main__":
    main()
