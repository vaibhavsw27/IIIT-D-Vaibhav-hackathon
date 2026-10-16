"""Config loading helpers shared by every module."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load config.yaml once and cache it."""
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(key: str) -> Path:
    """Resolve a path from the `paths` section relative to the repo root."""
    return ROOT / load_config()["paths"][key]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "transformers", "filelock", "yfinance", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
