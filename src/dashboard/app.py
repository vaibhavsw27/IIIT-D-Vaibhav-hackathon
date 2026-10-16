"""Streamlit dashboard: AI risk signals + tactical index rebalancer (Module A).

Run:  streamlit run src/dashboard/app.py
Reads everything from data/signals.db, so it works offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config, path  # noqa: E402
from src.engine.store import SignalStore  # noqa: E402

# ---------------------------------------------------------------- palette
# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Diverging blue (positive) <-> red (negative), grey midpoint
DIVERGING = [[0.0, "#d03b3b"], [0.25, "#e88a84"], [0.5, "#e7e6e1"], [0.75, "#86b6ef"], [1.0, "#1c5cab"]]
INK, MUTED, GRID = "#52514e", "#898781", "rgba(137,135,129,0.25)"
STRATEGY_COLORS = {"Sentiment Index": CAT[0], "Equal Weight": CAT[1], "S&P 500": CAT[2]}

st.set_page_config(page_title="AI Risk Signals & Tactical Index", page_icon="📊", layout="wide")

CFG = load_config()
UNIVERSE = CFG["universe"]
TICKERS = list(UNIVERSE)
SECTORS = sorted({m["sector"] for m in UNIVERSE.values()})
SECTOR_COLOR = {s: CAT[i] for i, s in enumerate(SECTORS)}
HIGH = CFG["impact"]["high_impact_threshold"]


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, title=None)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=None)
    return fig


# ---------------------------------------------------------------- data
@st.cache_resource
def get_store() -> SignalStore:
    return SignalStore()


@st.cache_data(ttl=30)
def load_signals(mode: str) -> pd.DataFrame:
    df = get_store().load_signals(run_mode=mode)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
        df["day"] = df["ts"].dt.tz_convert(None).dt.normalize()
    return df


@st.cache_data(ttl=30)
def load_weights(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = get_store().load_weights(run_id)
    if not w.empty:
        w["timestamp"] = pd.to_datetime(w["timestamp"])
        w["sector"] = w["ticker"].map(lambda t: UNIVERSE[t]["sector"])
    r = get_store().load_rebalance_log(run_id)
    if not r.empty:
        r["timestamp"] = pd.to_datetime(r["timestamp"])
    return w, r


@st.cache_data(ttl=60)
def load_backtest(mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = path("backtest_results")
    if mode != "sample":
        p = p.with_name(f"backtest_results_{mode}.csv")
    m = p.with_name(p.stem + "_metrics.csv")
    if not p.exists():
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(p, index_col=0, parse_dates=True), (pd.read_csv(m) if m.exists() else pd.DataFrame())


@st.cache_resource(show_spinner="Loading FinBERT...")
def get_engine(fast: bool):
    from src.engine.engine import RiskEngine

    return RiskEngine(store=None, fast=fast, run_mode="adhoc")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
modes = [m for m in ["sample", "historical", "live"] if not load_signals(m).empty] or ["sample"]
mode = st.sidebar.selectbox("Dataset / run", modes, help="sample = synthetic offline demo; historical = Kaggle tweets; live = GDELT")
signals = load_signals(mode)
weight_runs = get_store().weight_runs()
run_id = mode if mode in weight_runs else (weight_runs[0] if weight_runs else None)
weights, reasons = load_weights(run_id) if run_id else (pd.DataFrame(), pd.DataFrame())
st.sidebar.caption(f"Signals: **{len(signals):,}** · weight history: **{run_id or 'none'}**")
if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.caption("Illustrative research prototype. Not investment advice.")

st.title("AI Risk Signal Engine & Tactical Index Rebalancer")
st.caption("Unstructured news & social text → sentiment, event type and impact per company → sentiment-tilted index weights.")

if signals.empty:
    st.warning("No signals yet. Run `python run_pipeline.py` (and `python run_backtest.py`) first.")
    st.stop()

# ---------------------------------------------------------------- 1. KPIs
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Signals processed", f"{len(signals):,}")
k2.metric("Documents", f"{signals['doc_id'].nunique():,}")
k3.metric("Sources", f"{signals['source'].nunique()}", help=", ".join(sorted(signals["source"].unique())))
k4.metric("Avg sentiment", f"{signals['sentiment_score'].mean():+.3f}")
k5.metric(f"High-impact events (≥{HIGH})", f"{(signals['impact_score'] >= HIGH).sum():,}")

tabs = st.tabs(["Index weights", "Performance", "Signal feed", "Sentiment heatmap", "Event breakdown", "Try it live"])

# ---------------------------------------------------------------- 2/3. weights
with tabs[0]:
    if weights.empty:
        st.info("No weight history yet. Run `python run_backtest.py`.")
    else:
        dates = sorted(weights["timestamp"].unique())
        st.subheader("Index weights over time")
        view = st.radio("Group by", ["Sector", "Ticker (select up to 3)"], horizontal=True, label_visibility="collapsed")
        if view == "Sector":
            sec = weights.groupby(["timestamp", "sector"], as_index=False)["weight"].sum()
            fig = px.area(sec, x="timestamp", y="weight", color="sector", color_discrete_map=SECTOR_COLOR,
                          category_orders={"sector": SECTORS})
            fig.update_traces(line=dict(width=0.5, color="#fcfcfb"), hovertemplate="%{fullData.name}: %{y:.1%}<extra></extra>")
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            fig.update_layout(hovermode="x unified")
        else:
            pick = st.multiselect("Tickers", TICKERS, default=["AAPL", "TSLA", "JPM"], max_selections=3)
            sub = weights[weights["ticker"].isin(pick)]
            fig = px.line(sub, x="timestamp", y="weight", color="ticker", color_discrete_sequence=CAT)
            fig.add_hline(y=1 / len(TICKERS), line_dash="dot", line_color=MUTED, annotation_text="equal weight",
                          annotation_font_color=MUTED)
            fig.update_traces(line_width=2, hovertemplate="%{fullData.name}: %{y:.2%}<extra></extra>")
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(hovermode="x unified")
        st.plotly_chart(style(fig, 400), width="stretch")

        st.subheader("Weights on a given day")
        c1, c2 = st.columns([3, 2])
        with c1:
            day = st.select_slider("Rebalance date", options=dates, value=dates[-1],
                                   format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
            cur = weights[weights["timestamp"] == day].sort_values("weight", ascending=False)
            fig = go.Figure(go.Bar(
                x=cur["ticker"], y=cur["weight"],
                marker=dict(color=cur["agg_sentiment"], colorscale=DIVERGING, cmin=-0.6, cmax=0.6,
                            colorbar=dict(title="Sentiment", thickness=10)),
                customdata=cur[["agg_sentiment"]],
                hovertemplate="%{x}: %{y:.2%}<br>sentiment %{customdata[0]:+.2f}<extra></extra>",
            ))
            fig.add_hline(y=1 / len(TICKERS), line_dash="dot", line_color=MUTED, annotation_text="equal weight 6.67%",
                          annotation_font_color=MUTED)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style(fig, 360), width="stretch")
        with c2:
            st.markdown("**Why did this change?**")
            day_r = reasons[reasons["timestamp"] == day] if not reasons.empty else pd.DataFrame()
            if day_r.empty:
                st.caption("No weight changes logged on this date.")
            else:
                day_r = day_r.reindex((day_r["new_weight"] - day_r["old_weight"]).abs().sort_values(ascending=False).index)
                for r in day_r.head(5).itertuples():
                    arrow = "▲" if r.new_weight > r.old_weight else "▼"
                    with st.expander(f"{arrow} {r.ticker}: {r.old_weight:.1%} → {r.new_weight:.1%}  (sentiment {r.agg_sentiment:+.2f})"):
                        st.write(r.reason.replace(" | ", "\n\n• ").replace("Drivers: ", "\n\n• "))

        with st.expander("▶ Replay the rebalancing (animated)"):
            step = max(1, len(dates) // 60)
            frames_days = dates[::step]
            anim = weights[weights["timestamp"].isin(frames_days)].copy()
            anim["date"] = anim["timestamp"].dt.strftime("%Y-%m-%d")
            fig = px.bar(anim, x="ticker", y="weight", color="agg_sentiment", animation_frame="date",
                         color_continuous_scale=DIVERGING, range_color=[-0.6, 0.6],
                         range_y=[0, CFG["rebalancer"]["w_max"] * 1.05], category_orders={"ticker": TICKERS})
            fig.add_hline(y=1 / len(TICKERS), line_dash="dot", line_color=MUTED)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style(fig, 420), width="stretch")

# ---------------------------------------------------------------- 4. performance
with tabs[1]:
    daily, met = load_backtest(mode if mode != "live" else "sample")
    if daily.empty:
        st.info("No backtest results yet. Run `python run_backtest.py`.")
    else:
        st.subheader("Cumulative return")
        cum = daily[[c for c in daily.columns if c.endswith("_growth")]].rename(columns={
            "sentiment_index_growth": "Sentiment Index", "equal_weight_growth": "Equal Weight", "sp500_growth": "S&P 500"})
        long = (cum - 1).reset_index(names="date").melt(id_vars="date", var_name="strategy", value_name="ret")
        fig = px.line(long, x="date", y="ret", color="strategy", color_discrete_map=STRATEGY_COLORS)
        fig.update_traces(line_width=2, hovertemplate="%{fullData.name}: %{y:.2%}<extra></extra>")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(style(fig, 400), width="stretch")
        if not met.empty:
            show = met.set_index("strategy")
            st.dataframe(show.style.format({
                "cumulative_return": "{:.2%}", "annualized_return": "{:.2%}", "annualized_volatility": "{:.2%}",
                "sharpe": "{:.2f}", "max_drawdown": "{:.2%}", "avg_daily_turnover": "{:.2%}"}),
                width="stretch")
        st.caption("Weights set at day t's close earn day t+1's return (no look-ahead). No transaction costs. Illustrative only.")

# ---------------------------------------------------------------- 5. feed
with tabs[2]:
    st.subheader("Signal feed")
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    ft = f1.multiselect("Ticker", TICKERS)
    fe = f2.multiselect("Event type", CFG["event_types"])
    fs = f3.multiselect("Source", sorted(signals["source"].unique()))
    fi = f4.number_input("Min impact", 1.0, 10.0, 1.0, 0.5)
    feed = signals
    if ft:
        feed = feed[feed["ticker"].isin(ft)]
    if fe:
        feed = feed[feed["event_type"].isin(fe)]
    if fs:
        feed = feed[feed["source"].isin(fs)]
    feed = feed[feed["impact_score"] >= fi].sort_values("ts", ascending=False)
    cols = ["timestamp", "ticker", "headline", "sentiment_score", "sentiment_label", "event_type", "impact_score", "source", "url"]

    def _hl(row):
        return ["background-color: rgba(208,59,59,0.12)" if row["impact_score"] >= HIGH else "" for _ in row]

    st.caption(f"{len(feed):,} signals · rows with impact ≥ {HIGH} highlighted")
    st.dataframe(
        feed[cols].head(500).style.apply(_hl, axis=1).format({"sentiment_score": "{:+.2f}", "impact_score": "{:.1f}"}),
        width="stretch", height=460,
        column_config={"url": st.column_config.LinkColumn("url", display_text="open")},
    )

# ---------------------------------------------------------------- 6. heatmap
with tabs[3]:
    st.subheader("Daily average sentiment by ticker")
    heat = signals.pivot_table(index="ticker", columns="day", values="sentiment_score", aggfunc="mean").reindex(TICKERS)
    if heat.shape[1] > 120:   # weekly buckets for long histories
        heat = heat.T.resample("W").mean().T
    fig = go.Figure(go.Heatmap(
        z=heat.values, x=heat.columns, y=heat.index, colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1, xgap=1, ygap=1,
        colorbar=dict(title="Sentiment", thickness=10), hovertemplate="%{y} · %{x|%Y-%m-%d}: %{z:+.2f}<extra></extra>",
    ))
    st.plotly_chart(style(fig, 480), width="stretch")
    st.caption("Blue = positive, red = negative, grey/blank = neutral or no signals.")

# ---------------------------------------------------------------- 8. breakdown
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Signals by event type")
        ev = signals.groupby("event_type").agg(count=("signal_id", "size"), avg_impact=("impact_score", "mean")).reset_index()
        ev = ev.sort_values("count")
        fig = go.Figure(go.Bar(y=ev["event_type"], x=ev["count"], orientation="h", marker_color=CAT[0],
                               customdata=ev[["avg_impact"]],
                               hovertemplate="%{y}: %{x} signals · avg impact %{customdata[0]:.1f}<extra></extra>"))
        st.plotly_chart(style(fig, 400), width="stretch")
    with c2:
        st.subheader("Sentiment label by source")
        src = signals.groupby(["source", "sentiment_label"]).size().reset_index(name="count")
        fig = px.bar(src, x="source", y="count", color="sentiment_label", barmode="stack",
                     color_discrete_map={"positive": "#2a78d6", "neutral": "#c3c2b7", "negative": "#d03b3b"},
                     category_orders={"sentiment_label": ["positive", "neutral", "negative"]})
        fig.update_traces(marker_line_color="#fcfcfb", marker_line_width=1)
        st.plotly_chart(style(fig, 400), width="stretch")
    st.subheader("Impact score distribution by event type")
    fig = px.box(signals, x="event_type", y="impact_score", points=False, color_discrete_sequence=[CAT[0]],
                 category_orders={"event_type": CFG["event_types"]})
    st.plotly_chart(style(fig, 360), width="stretch")

# ---------------------------------------------------------------- 7. try it live
with tabs[5]:
    st.subheader("Try it live: paste any headline or tweet")
    c1, c2 = st.columns([3, 1])
    text = c1.text_area("Text", "Moody's downgrades Tesla's credit rating as debt load rises and Elon Musk sells shares", height=100)
    src_type = c2.radio("Source type", ["news", "social"])
    fast = c2.toggle("Fast mode (VADER)", value=False)
    if st.button("Analyze", type="primary"):
        from src.ingestion.base import Document, make_doc_id, now_utc

        doc = Document(make_doc_id("adhoc", text), "adhoc", src_type, now_utc(), text, "", None, [])
        out = get_engine(fast).analyze([doc], persist=False)
        if not out:
            from src.engine.events import classify_event
            from src.engine.sentiment import score_texts

            s = score_texts([text], fast=fast)[0]
            e = classify_event(text)
            st.warning("No company from the 15-stock universe was detected, so no signal is emitted. Raw scores:")
            st.json({"sentiment_score": s.score, "sentiment_label": s.label, "event_type": e.event_type,
                     "event_confidence": e.confidence})
        else:
            for sig in out:
                a, b, c, d = st.columns(4)
                a.metric("Ticker", sig["ticker"], sig["company"], delta_color="off")
                b.metric("Sentiment", f"{sig['sentiment_score']:+.2f}", sig["sentiment_label"],
                         delta_color="normal" if sig["sentiment_score"] >= 0 else "inverse")
                c.metric("Event", sig["event_type"], f"conf {sig['event_confidence']:.2f}", delta_color="off")
                d.metric("Impact (1-10)", f"{sig['impact_score']:.1f}")
            with st.expander("Signal JSON"):
                st.json(out)
