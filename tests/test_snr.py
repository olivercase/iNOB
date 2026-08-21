"""SNR + noise-floor utility tests (no DUNEuro required)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inob.analysis.snr import (
    array_summary,
    compute_noise_floors,
    per_source_amplitude,
    per_source_best_bipolar,
    per_source_peak,
    snr_per_source,
)
from inob.config import ConfigError, load_config
from dataclasses import replace

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_noise_floors_have_expected_orders() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    nf = compute_noise_floors(cfg)
    # OPM: 7 fT/√Hz over the sensor-weighted band (a 135 Hz pole turns the
    # nominal 470 Hz into ~147 Hz) ≈ 85 fT, then divided by the 0.39 gain the
    # sensor has at the 318 Hz CAP peak → ~217 fT effective.
    assert 50 < nf.meg_per_channel_fT < 500
    # EEG amplifier 0.1 µV/√Hz dominates over Johnson@10 kΩ ≈ 0.013 µV/√Hz
    # → ~0.1 µV/√Hz × √470 ≈ 2.2 µV
    assert 0.5 < nf.eeg_per_channel_uV < 10


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


def test_best_bipolar_is_reference_invariant() -> None:
    """The whole point of the bipolar observable: it survives re-referencing.

    A surface potential is defined only up to a per-source constant, so the
    single-channel peak moves when the reference changes and the bipolar
    amplitude does not.
    """
    rng = np.random.default_rng(3)
    L = rng.standard_normal((16, 3 * 7))
    L_car = L - L.mean(axis=0, keepdims=True)
    L_ref0 = L - L[0][None, :]

    bipolar = per_source_best_bipolar(L)
    assert np.allclose(per_source_best_bipolar(L_car), bipolar)
    assert np.allclose(per_source_best_bipolar(L_ref0), bipolar)
    # ... whereas the per-channel peak does not survive it.
    assert not np.allclose(per_source_peak(L_car), per_source_peak(L))


def test_best_bipolar_matches_all_pairs_brute_force() -> None:
    """The O(C·S) max-minus-min form equals the O(C²·S) all-pairs maximum."""
    rng = np.random.default_rng(4)
    L = rng.standard_normal((9, 3 * 4))
    C, three_S = L.shape
    S = three_S // 3
    L3 = L.reshape(C, S, 3)
    brute = np.array([
        max(np.abs(L3[:, s, k][:, None] - L3[:, s, k][None, :]).max()
            for k in range(3))
        for s in range(S)
    ])
    assert np.allclose(per_source_best_bipolar(L), brute)


def test_best_bipolar_at_least_peak_for_zero_mean_array() -> None:
    """A common-average array's bipolar span bounds its single-channel peak."""
    rng = np.random.default_rng(5)
    L = rng.standard_normal((12, 3 * 6))
    L = L - L.mean(axis=0, keepdims=True)
    assert (per_source_best_bipolar(L) >= per_source_peak(L) - 1e-12).all()


def test_opm_bandwidth_rolls_off_both_signal_and_noise() -> None:
    """A narrow-band sensor must not profit from the noise its pole removes.

    This is the bug Gareth flagged: pairing a QZFM-3 with a 30–500 Hz band
    integrated noise the sensor cannot deliver, while charging nothing for the
    CAP energy it cannot pass. Both effects are now modelled, so shrinking the
    pole always makes the effective floor worse, never better.
    """
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    base = compute_noise_floors(cfg)

    narrow = replace(cfg, noise=replace(cfg.noise, opm_bandwidth_hz=30.0))
    wide = replace(cfg, noise=replace(cfg.noise, opm_bandwidth_hz=5000.0))
    sigma_narrow = compute_noise_floors(narrow).meg_per_channel_fT
    sigma_wide = compute_noise_floors(wide).meg_per_channel_fT
    assert sigma_narrow > base.meg_per_channel_fT > sigma_wide

    # A pole far above the band is the old flat-window arithmetic.
    n = wide.noise
    flat = float(n.opm_intrinsic_fT_sqrtHz) * np.sqrt(n.effective_bandwidth_hz)
    assert sigma_wide == pytest.approx(flat, rel=0.05)

    # The QZFM-3 default passes well under half of a 0.5 ms vagal CAP.
    assert 0.2 < base.meg_sensor_gain < 0.5
    assert base.signal_hz == pytest.approx(318.3, rel=0.01)


def test_wideband_preset_swaps_noise_and_bandwidth_together() -> None:
    from inob.sensors.opm_presets import OPM_SENSORS

    cfg = load_config(REPO_ROOT / "configs" / "default.yaml",
                      overrides=["noise.opm_sensor=he4_wideband"])
    assert cfg.noise.opm_intrinsic_fT_sqrtHz == OPM_SENSORS["he4_wideband"].noise_fT_sqrtHz
    assert cfg.noise.opm_bandwidth_hz == OPM_SENSORS["he4_wideband"].bandwidth_hz
    nf = compute_noise_floors(cfg)
    # Wide enough to pass the CAP essentially intact...
    assert nf.meg_sensor_gain > 0.95
    # ...but its noise density costs more than the QZFM-3's roll-off does.
    assert nf.meg_per_channel_fT > compute_noise_floors(
        load_config(REPO_ROOT / "configs" / "default.yaml")
    ).meg_per_channel_fT


def test_unknown_opm_preset_is_rejected() -> None:
    with pytest.raises((KeyError, ConfigError)):
        load_config(REPO_ROOT / "configs" / "default.yaml",
                    overrides=["noise.opm_sensor=no_such_sensor"])
