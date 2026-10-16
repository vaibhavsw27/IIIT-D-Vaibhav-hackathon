"""One-command demo: sample pipeline -> backtest -> dashboard.

  python run_demo.py            # FinBERT (downloads ~440 MB once)
  python run_demo.py --fast     # VADER, no model download
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(args: list[str]) -> None:
    print("\n>>", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def main() -> None:
    fast = ["--fast"] if "--fast" in sys.argv else []
    run(["run_pipeline.py", "--mode", "sample", *fast])
    run(["run_backtest.py", "--mode", "sample"])
    print("\nStarting dashboard at http://localhost:8501  (Ctrl+C to stop)", flush=True)
    subprocess.run([PY, "-m", "streamlit", "run", "src/dashboard/app.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
