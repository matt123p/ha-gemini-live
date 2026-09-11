"""Pytest configuration: test against the local Home Assistant checkout."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scratch" / "ha-core"))
sys.path.insert(0, str(ROOT / "custom_components"))
