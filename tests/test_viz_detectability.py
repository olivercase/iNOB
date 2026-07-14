"""Tests for inob.viz.detectability: pure math + render/summary smoke tests."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from inob.viz.detectability import (
    DetectabilityScenario,
    detectability_summary,
    per_source_peak_amplitude,
    per_source_rms_amplitude,
    render_detectability,
    required_trials,
    snr_after_n_trials,
)

from tests.viz_pipeline_helpers import build_pipeline_cfg


def test_per_source_peak_amplitude_shape() -> None:
    rng = np.random.default_rng(0)
    L = rng.standard_normal((10, 12))
    peak = per_source_peak_amplitude(L)
    assert peak.shape == (4,)
    assert (peak > 0).all()


def test_per_source_rms_amplitude_shape_and_le_peak() -> None:
    rng = np.random.default_rng(0)
    L = rng.standard_normal((10, 12))
    rms = per_source_rms_amplitude(L)
    peak = per_source_peak_amplitude(L)
    assert rms.shape == (4,)
    assert (rms <= peak + 1e-12).all()


def test_required_trials_scales_inverse_square_with_signal() -> None:
    n1 = required_trials(1.0, 1.0, snr_target=3.0)
    n2 = required_trials(2.0, 1.0, snr_target=3.0)
    assert n1 == pytest.approx(9.0)
    assert n2 == pytest.approx(9.0 / 4.0)


def test_required_trials_nonpositive_signal_is_inf() -> None:
    assert math.isinf(required_trials(0.0, 1.0))
    assert math.isinf(required_trials(-1.0, 1.0))
    assert math.isinf(required_trials(1.0, 0.0))


def test_snr_after_n_trials_scales_with_sqrt_n() -> None:
    s1 = snr_after_n_trials(1.0, 1.0, n_trials=1)
    s100 = snr_after_n_trials(1.0, 1.0, n_trials=100)
    assert s100 == pytest.approx(s1 * 10.0)


def test_snr_after_n_trials_clamps_below_one_trial() -> None:
    s = snr_after_n_trials(1.0, 1.0, n_trials=0.1)
    assert s == pytest.approx(1.0)


def test_snr_after_n_trials_zero_sigma_is_inf() -> None:
    assert math.isinf(snr_after_n_trials(1.0, 0.0, n_trials=10))


def test_required_and_snr_after_are_consistent() -> None:
    sig, sigma = 0.5, 2.0
    n = required_trials(sig, sigma, snr_target=3.0)
    snr = snr_after_n_trials(sig, sigma, n)
    assert snr == pytest.approx(3.0, rel=1e-6)


def test_detectability_scenario_dataclass_defaults() -> None:
    sc = DetectabilityScenario(label="x", Q_nAm=5.0)
    assert sc.description == ""
    assert sc.Q_nAm == 5.0


def test_render_detectability_smoke(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    out = render_detectability(cfg, out_path=tmp_path / "det.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_detectability_returns_requested_path(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    custom = tmp_path / "custom_name.png"
    out = render_detectability(cfg, out_path=custom, source_idx=1)
    assert out == custom
    assert out.exists()


def test_detectability_summary_structure(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    summary = detectability_summary(cfg)
    assert set(summary) == {
        "source_idx", "source_z_mm", "noise_meg_fT", "noise_eeg_uV",
        "bandwidth_hz", "scenarios",
    }
    assert len(summary["scenarios"]) == 4
    for row in summary["scenarios"].values():
        assert set(row) == {
            "Q_nAm", "MEG_per_trial_fT", "EEG_per_trial_uV",
            "MEG_single_trial_SNR", "EEG_single_trial_SNR",
            "MEG_trials_for_SNR3", "EEG_trials_for_SNR3",
        }
        assert row["MEG_trials_for_SNR3"] > 0
