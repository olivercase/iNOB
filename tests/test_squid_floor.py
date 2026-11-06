"""The SQUID alternative floor used by the spine figures is the one in the paper's Table 5."""

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
    # < 4 fT/sqrt(Hz), single pole at the 5 kHz band edge, recorded 100-5000 Hz.
    density, f_c, lo, hi, f_cap = 4.0, 5000.0, 100.0, 5000.0, 227.0
    noise_bw = f_c * (math.atan(hi / f_c) - math.atan(lo / f_c))
    gain = 1.0 / math.sqrt(1.0 + (f_cap / f_c) ** 2)
    assert abs(density * math.sqrt(noise_bw) / gain - _floor_ft()) < 1.0
