"""Build docs/presentation.pptx (7 slides) and export docs/presentation.pdf.

  python scripts/make_presentation.py

PDF export uses PowerPoint (COM) if installed, else LibreOffice headless.
Slides pull live numbers from data/*.csv, so re-run after new results.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SHOTS = DOCS / "screenshots"

NAME, COLLEGE, EMAIL = "Vaibhav", "IIIT Delhi", "vaibhav24597@iiitd.ac.in"
TITLE = "Real-Time AI Risk Signal Engine & Tactical Index Rebalancer"

INK = RGBColor(0x0B, 0x0B, 0x0B)
SUB = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
AQUA = RGBColor(0x1B, 0xAF, 0x7A)
VIOLET = RGBColor(0x62, 0x50, 0xD6)
PANEL = RGBColor(0xF4, 0xF6, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Segoe UI"

W, H = Inches(13.333), Inches(7.5)


# ------------------------------------------------------------------ helpers
def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, spacing=1.1):
    """runs: str or list of paragraphs; a paragraph may be (text, dict(size=, color=, bold=))."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        t, opt = (p if isinstance(p, tuple) else (p, {}))
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        para.space_after = Pt(opt.get("after", 6))
        r = para.add_run()
        r.text = t
        r.font.name = FONT
        r.font.size = Pt(opt.get("size", size))
        r.font.bold = opt.get("bold", bold)
        r.font.color.rgb = opt.get("color", color)
    return tb


def rect(slide, x, y, w, h, fill=PANEL, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.06
    s.shadow.inherit = False
    return s


def header(slide, n, title, kicker):
    rect(slide, 0, 0, W, Inches(0.12), fill=BLUE, shape=MSO_SHAPE.RECTANGLE)
    text(slide, Inches(0.6), Inches(0.38), Inches(10), Inches(0.35), kicker.upper(), size=12, color=BLUE, bold=True)
    text(slide, Inches(0.6), Inches(0.68), Inches(12), Inches(0.8), title, size=30, bold=True)
    text(slide, Inches(11.9), Inches(7.0), Inches(0.9), Inches(0.3), f"{n} / 7", size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    text(slide, Inches(0.6), Inches(7.0), Inches(8), Inches(0.3), f"{NAME} · {COLLEGE} · S&P Global & Crisil Campus Hackathon 2026",
         size=11, color=MUTED)


def card(slide, x, y, w, h, title, body, accent=BLUE, size=14):
    rect(slide, x, y, w, h)
    rect(slide, x, y, Inches(0.08), h, fill=accent, shape=MSO_SHAPE.RECTANGLE)
    text(slide, x + Inches(0.28), y + Inches(0.18), w - Inches(0.45), Inches(0.4), title, size=size + 2, bold=True)
    text(slide, x + Inches(0.28), y + Inches(0.62), w - Inches(0.45), h - Inches(0.7), body, size=size, color=SUB)


def picture(slide, path: Path, x, y, w=None, h=None):
    if path.exists():
        return slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    text(slide, x, y, w or Inches(4), Inches(0.4), f"[missing {path.name}]", size=12, color=MUTED)


def crop_top(pic, frac):
    if pic is not None:
        pic.crop_top = frac


def pct(v):
    return f"{v * 100:.1f}%"


# ------------------------------------------------------------------ data
def load_numbers() -> dict:
    out: dict = {"mode": "sample"}
    hist = ROOT / "data" / "backtest_results_historical_metrics.csv"
    sample = ROOT / "data" / "backtest_results_metrics.csv"
    if hist.exists():
        out["mode"] = "historical"
        out["bt"] = pd.read_csv(hist).set_index("strategy")
    elif sample.exists():
        out["bt"] = pd.read_csv(sample).set_index("strategy")
    ev = ROOT / "data" / "sentiment_eval.csv"
    out["eval"] = pd.read_csv(ev).set_index("model") if ev.exists() else None
    db_csv = ROOT / "data" / "signals.csv"
    if db_csv.exists():
        s = pd.read_csv(db_csv)
        out["n_signals"] = len(s)
        out["n_sources"] = s["source"].nunique()
    return out


# ------------------------------------------------------------------ slides
def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, fill=RGBColor(0x0F, 0x1B, 0x2D), shape=MSO_SHAPE.RECTANGLE)
    rect(s, Inches(0.8), Inches(2.0), Inches(0.12), Inches(2.75), fill=BLUE, shape=MSO_SHAPE.RECTANGLE)
    text(s, Inches(1.15), Inches(1.95), Inches(11), Inches(0.4), "S&P GLOBAL & CRISIL CAMPUS HACKATHON 2026", size=14,
         color=RGBColor(0x86, 0xB6, 0xEF), bold=True)
    text(s, Inches(1.15), Inches(2.4), Inches(11.2), Inches(1.6), TITLE, size=40, color=WHITE, bold=True)
    text(s, Inches(1.15), Inches(4.05), Inches(11), Inches(0.6),
         "From unstructured news & social text to structured risk signals, and a sentiment-tilted index (Module A)",
         size=18, color=RGBColor(0xC3, 0xC2, 0xB7))
    text(s, Inches(1.15), Inches(5.3), Inches(11), Inches(1.0),
         [(NAME, {"size": 22, "bold": True, "color": WHITE}), (f"{COLLEGE}  ·  {EMAIL}", {"size": 15, "color": RGBColor(0xC3, 0xC2, 0xB7)})])


def slide_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, 2, "Too much text, too little structure", "Problem & approach")
    text(s, Inches(0.6), Inches(1.55), Inches(12), Inches(0.9),
         "Market-moving information arrives as headlines and posts. Analysts can't read it all, and keyword alerts are noisy "
         "and not comparable across companies. Goal: turn the stream into explainable numbers a portfolio can act on.",
         size=16, color=SUB)
    y, h, w, gap = Inches(2.75), Inches(3.55), Inches(3.9), Inches(0.27)
    card(s, Inches(0.6), y, w, h, "1  Read",
         ["News: GDELT live API + Kaggle financial news", "Social: Kaggle stock tweets",
          "Synthetic sample so the demo always runs offline", "Clean, dedupe, link to 15 S&P 100 tickers"], BLUE)
    card(s, Inches(0.6) + w + gap, y, w, h, "2  Score",
         ["Sentiment  -1..1   (FinBERT)", "Event type  10 classes   (rules)", "Impact  1..10   (explainable formula)",
          "One JSON signal per company per document"], ORANGE)
    card(s, Inches(0.6) + 2 * (w + gap), y, w, h, "3  Act - Module A",
         ["Decayed, impact-weighted sentiment per stock", "Tilt equal weights: w = w0 · e^(k·s)",
          "Bounds 2-15%, max 3 pts change / day", "Backtest vs equal weight & S&P 500"], AQUA)


def slide_design(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, 3, "System design and data flow", "Architecture")
    ph = Inches(5.35)
    pw = int(ph * 3000 / 1640)
    picture(s, DOCS / "architecture.png", int((W - pw) / 2), Inches(1.55), h=ph)


def slide_impl(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, 4, "Implementation highlights", "Key modules & why")
    items = [
        ("FinBERT sentiment", "Finance-tuned BERT on CPU, batched; score = P(pos) − P(neg). Cached so no text is scored twice. VADER fallback (--fast).", BLUE),
        ("Rule-based events", "Weighted regex per class (credit rating, takeover, antitrust, CEO resigns…). Auditable; confidence = share of evidence.", ORANGE),
        ("Explainable impact", "severity[event] × (0.5+|s|) × source weight × (1+0.1·ln(1+24h buzz)), clipped 1-10. All weights in config.yaml.", AQUA),
        ("Bounded rebalancer", "Exact solver: bisection keeps Σw = 1 with 2-15% bounds and a 3-pt turnover cap. Each change logged with its headlines.", VIOLET),
        ("No look-ahead", "Signals up to 4pm NY on day t set weights that earn day t+1's return. Enforced by a unit test (169 tests).", BLUE),
        ("Serving", "SQLite + JSON/CSV export · FastAPI (/signals, /analyze, /weights) · Streamlit dashboard · live GDELT polling.", ORANGE),
    ]
    cw, ch, gx, gy = Inches(3.95), Inches(2.45), Inches(0.22), Inches(0.25)
    for i, (t, b, c) in enumerate(items):
        x = Inches(0.6) + (i % 3) * (cw + gx)
        y = Inches(1.65) + (i // 3) * (ch + gy)
        card(s, x, y, cw, ch, t, b, c, size=13)


def slide_results(prs, nums):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    mode = nums["mode"]
    header(s, 5, "Key results", "Outputs & metrics")
    # left: metrics
    ev = nums.get("eval")
    y = Inches(1.6)
    text(s, Inches(0.6), y, Inches(5.6), Inches(0.4), "Sentiment model (labelled financial news)", size=15, bold=True)
    if ev is not None and "FinBERT" in ev.index:
        fb, vd = ev.loc["FinBERT"], ev.loc["VADER"]
        rows = [("FinBERT", f"{fb.accuracy:.1%}", f"{fb.macro_f1:.3f}"), ("VADER baseline", f"{vd.accuracy:.1%}", f"{vd.macro_f1:.3f}")]
        n_eval = int(fb["n"])
    else:
        rows, n_eval = [("FinBERT", "pending", "pending"), ("VADER baseline", "pending", "pending")], 0
    tbl(s, Inches(0.6), y + Inches(0.45), Inches(5.6), ["Model", "Accuracy", "Macro-F1"], rows)
    if n_eval:
        text(s, Inches(0.6), y + Inches(1.75), Inches(5.6), Inches(0.3), f"n = {n_eval:,} sentences (Kaggle / Financial PhraseBank)", size=11, color=MUTED)

    y2 = Inches(4.0)
    label = "Kaggle tweets, Sep 2021 – Sep 2022" if mode == "historical" else "synthetic sample, Jul – Aug 2022"
    text(s, Inches(0.6), y2, Inches(5.6), Inches(0.4), f"Backtest ({label})", size=15, bold=True)
    bt = nums.get("bt")
    if bt is not None:
        rows = [(k, pct(bt.loc[k, "cumulative_return"]), f"{bt.loc[k, 'sharpe']:.2f}", pct(bt.loc[k, "max_drawdown"]))
                for k in ["Sentiment Index", "Equal Weight", "S&P 500"] if k in bt.index]
        tbl(s, Inches(0.6), y2 + Inches(0.45), Inches(5.6), ["Strategy", "Return", "Sharpe", "Max DD"], rows)
        to = bt.loc["Sentiment Index", "avg_daily_turnover"]
        text(s, Inches(0.6), y2 + Inches(2.05), Inches(5.6), Inches(0.5),
             f"Avg daily turnover {pct(to)}; no transaction costs. Illustrative only.", size=11, color=MUTED)
    # right: chart + dashboard
    picture(s, SHOTS / f"backtest_cumulative_{mode}.png", Inches(6.7), Inches(1.5), w=Inches(5.95))
    picture(s, SHOTS / f"weights_stacked_{mode}.png", Inches(6.7), Inches(4.2), w=Inches(5.95))


def tbl(slide, x, y, w, head, rows):
    shape = slide.shapes.add_table(len(rows) + 1, len(head), x, y, w, Inches(0.36) * (len(rows) + 1))
    t = shape.table
    for j, hname in enumerate(head):
        c = t.cell(0, j)
        c.text = hname
        c.fill.solid()
        c.fill.fore_color.rgb = BLUE
    for i, r in enumerate(rows, start=1):
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.text = str(v)
            c.fill.solid()
            c.fill.fore_color.rgb = PANEL if i % 2 else WHITE
    for i in range(len(rows) + 1):
        for j in range(len(head)):
            for p in t.cell(i, j).text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
                for r in p.runs:
                    r.font.name = FONT
                    r.font.size = Pt(12)
                    r.font.bold = i == 0 or (i == 1 and j == 0)
                    r.font.color.rgb = WHITE if i == 0 else INK
    t.columns[0].width = int(w * 0.4)
    for j in range(1, len(head)):
        t.columns[j].width = int(w * 0.6 / (len(head) - 1))


def slide_impact(prs, nums):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, 6, "Why it matters", "Domain impact")
    pts = [
        ("Faster reaction", "A headline becomes a scored, company-tagged signal in well under a second on a laptop CPU, and live mode refreshes every 15 minutes.", BLUE),
        ("Explainable by design", "Class probabilities, matched event keywords, the impact formula and the headlines behind every weight change are all visible. That fits risk and index governance.", ORANGE),
        ("Controlled risk", "Weight bounds, a turnover cap and decay to neutral stop the index from chasing one headline.", AQUA),
        ("Reusable engine", "The same API feed can drive watch-lists, credit early-warning, or stress-testing scenarios (Module B), not just this index.", VIOLET),
    ]
    cw, ch = Inches(6.0), Inches(2.3)
    for i, (t, b, c) in enumerate(pts):
        x = Inches(0.6) + (i % 2) * (cw + Inches(0.13))
        y = Inches(1.65) + (i // 2) * (ch + Inches(0.25))
        card(s, x, y, cw, ch, t, b, c, size=14)
    n = nums.get("n_signals")
    if n:
        text(s, Inches(0.6), Inches(6.55), Inches(12), Inches(0.4),
             f"{n:,} signals from {nums['n_sources']} sources currently in data/signals.json, all served by FastAPI and the dashboard.",
             size=13, color=SUB)


def slide_limits(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, 7, "Limitations & next steps", "Honest scope")
    card(s, Inches(0.6), Inches(1.65), Inches(6.0), Inches(3.9), "Limitations", [
        "One sentiment per document: two companies in a headline share it",
        "Dictionary entity linking; ambiguous names need exclusion lists",
        "Keyword event rules miss unusual phrasing (falls to “Other”)",
        "FinBERT is trained on news; weaker on slang-heavy tweets",
        "Backtest: one year of tweets, no transaction costs",
        "GDELT throttling: a live poll takes 2-4 minutes",
    ], ORANGE, size=14)
    card(s, Inches(6.73), Inches(1.65), Inches(6.0), Inches(3.9), "Next steps", [
        "Entity-level (aspect) sentiment",
        "Fine-tune FinBERT on labelled finance tweets",
        "Zero-shot classifier as a second opinion on events",
        "Transaction-cost and liquidity-aware rebalancing",
        "Longer, multi-source history (news archives)",
        "Module B: feed signals into stress-testing scenarios",
    ], AQUA, size=14)


def export_pdf(pptx: Path, pdf: Path) -> bool:
    try:  # PowerPoint COM automation
        import win32com.client  # type: ignore

        app = win32com.client.Dispatch("PowerPoint.Application")
        deck = app.Presentations.Open(str(pptx), WithWindow=False)
        deck.SaveAs(str(pdf), 32)  # ppSaveAsPDF
        deck.Close()
        app.Quit()
        return True
    except Exception as exc:
        print("PowerPoint COM export unavailable:", exc)
    ps = (
        "$app = New-Object -ComObject PowerPoint.Application; "
        f"$p = $app.Presentations.Open('{pptx}', $true, $false, $false); "
        f"$p.SaveAs('{pdf}', 32); $p.Close(); $app.Quit()"
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=180)
        if pdf.exists() and r.returncode == 0:
            return True
        print("PowerShell export failed:", r.stderr[:300])
    except Exception as exc:
        print("PowerShell export failed:", exc)
    soffice = shutil.which("soffice") or r"C:\Program Files\LibreOffice\program\soffice.exe"
    if Path(soffice).exists():
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(pdf.parent), str(pptx)], timeout=180)
        return pdf.exists()
    return False


def main() -> None:
    nums = load_numbers()
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    slide_title(prs)
    slide_problem(prs)
    slide_design(prs)
    slide_impl(prs)
    slide_results(prs, nums)
    slide_impact(prs, nums)
    slide_limits(prs)
    out = DOCS / "presentation.pptx"
    prs.save(out)
    print("Saved", out)
    pdf = DOCS / "presentation.pdf"
    if "--no-pdf" not in sys.argv:
        print("PDF:", "saved " + str(pdf) if export_pdf(out, pdf) else "export failed - open the .pptx and Save As PDF")


if __name__ == "__main__":
    main()
