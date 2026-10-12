"""Paths and case configuration."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"


def load_cases() -> dict[str, dict]:
    cases = yaml.safe_load((ROOT / "config" / "cases.yaml").read_text())["cases"]
    return {c["id"]: c for c in cases}
