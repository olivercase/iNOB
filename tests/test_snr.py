"""SNR + noise-floor utility tests (no DUNEuro required)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inob.analysis.snr import (
    array_summary,
    compute_noise_floors,
    per_source_amplitude,
    snr_per_source,
)
from inob.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_noise_floors_have_expected_orders() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    nf = compute_noise_floors(cfg)
    # OPM 15 fT/√Hz × √1000 ≈ 474 fT
    assert 200 < nf.meg_per_channel_fT < 1000
    # EEG amplifier 0.5 µV/√Hz dominates over Johnson@50kΩ ≈ 0.93 µV/√Hz
    # → ~1.06 µV/√Hz × √1000 ≈ 33 µV
    assert 10 < nf.eeg_per_channel_uV < 80


def test_per_source_amplitude_shape_and_units() -> None:
    rng = np.random.default_rng(0)
    L = rng.standard_normal((50, 30)) * 1e-6
    amps = per_source_amplitude(L)
    assert amps.shape == (10,)   # 30/3 = 10 sources
    assert (amps > 0).all()


def test_snr_per_source_scales_with_root_n() -> None:
    rng = np.random.default_rng(0)
    L = rng.standard_normal((20, 12))
    s1 = snr_per_source(L, 1.0, n_averages=1)
    s100 = snr_per_source(L, 1.0, n_averages=100)
    np.testing.assert_allclose(s100, s1 * 10.0, rtol=1e-6)


def test_array_summary_keys() -> None:
    arr = np.linspace(1, 10, 100)
    s = array_summary(arr)
    assert set(s) == {"n_sources", "min", "p5", "p50", "p95", "max"}
    assert s["min"] == pytest.approx(1.0)
    assert s["max"] == pytest.approx(10.0)


def test_per_source_invalid_moment_raises() -> None:
    L = np.zeros((4, 9))
    with pytest.raises(ValueError, match="moment="):
        per_source_amplitude(L, moment="bogus")


def test_per_source_bad_columns_raises() -> None:
    L = np.zeros((4, 7))      # not divisible by 3
    with pytest.raises(ValueError, match="not divisible"):
        per_source_amplitude(L)
