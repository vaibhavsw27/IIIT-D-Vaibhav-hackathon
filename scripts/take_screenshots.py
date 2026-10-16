"""Capture dashboard screenshots into docs/screenshots/ (dev helper).

Needs: pip install playwright   (uses your installed Chrome, no browser download)
Start the dashboard first:  streamlit run src/dashboard/app.py
Then:  python scripts/take_screenshots.py [--url http://localhost:8501] [--mode sample]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"

TABS = [
    ("Index weights", "dashboard_weights"),
    ("Performance", "dashboard_performance"),
    ("Signal feed", "dashboard_feed"),
    ("Sentiment heatmap", "dashboard_heatmap"),
    ("Event breakdown", "dashboard_events"),
    ("Try it live", "dashboard_try_it_live"),
]


def wait_idle(page, ms: int = 2500) -> None:
    page.wait_for_selector("[data-testid='stApp']", timeout=60_000)
    try:
        page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=60_000)
    except Exception:
        pass
    page.wait_for_timeout(ms)


def trim(png: Path, pad: int = 40) -> None:
    """Crop empty space below the last row of content."""
    from PIL import Image, ImageChops

    im = Image.open(png).convert("RGB")
    main_area = im.crop((int(im.width * 0.22), 0, im.width, im.height))
    bg = Image.new("RGB", main_area.size, main_area.getpixel((main_area.width - 5, main_area.height - 5)))
    bbox = ImageChops.difference(main_area, bg).getbbox()
    if bbox:
        im.crop((0, 0, im.width, min(im.height, bbox[3] + pad))).save(png)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8501")
    p.add_argument("--mode", default=None, help="select this dataset in the sidebar first")
    p.add_argument("--suffix", default="")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1900}, device_scale_factor=1.5)
        page.goto(args.url)
        wait_idle(page, 5000)
        if args.mode:
            page.locator("[data-testid='stSidebar'] [data-testid='stSelectbox']").first.click()
            page.get_by_role("option", name=args.mode, exact=True).click()
            wait_idle(page, 4000)
        page.screenshot(path=OUT / f"dashboard_overview{args.suffix}.png")
        trim(OUT / f"dashboard_overview{args.suffix}.png")

        for label, name in TABS:
            page.get_by_role("tab", name=label).click()
            wait_idle(page, 3000)
            if label == "Try it live":
                page.get_by_role("button", name="Analyze").click()
                wait_idle(page, 8000)
            page.screenshot(path=OUT / f"{name}{args.suffix}.png")
            trim(OUT / f"{name}{args.suffix}.png")
            print("saved", name)
        browser.close()


if __name__ == "__main__":
    main()
