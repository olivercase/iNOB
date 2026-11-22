"""The SQUID floor used by the spine figures is the one in the paper (Section 3.7): 280 fT."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "spine_figures.py"


def _floor_ft() -> float:
    spec = importlib.util.spec_from_file_location("spine_figures_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SQUID_FLOOR_FT


def test_squid_floor_matches_its_derivation() -> None:
    # < 4 fT/sqrt(Hz), flat response (no roll-off), recorded 100-5000 Hz.
    density, lo, hi = 4.0, 100.0, 5000.0
    assert abs(density * math.sqrt(hi - lo) - _floor_ft()) < 1.0
