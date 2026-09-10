"""Cross-modality coupling tests."""

from __future__ import annotations

import numpy as np
import pytest

from inob.analysis.cross_modality import (
    CrossModalityStats,
    amplitude_correlation,
    fit_moment_from_eeg,
    per_source_amplitude,
    predict_meg_from_eeg,
    shared_singular_modes,
)


def test_per_source_amplitude_shape() -> None:
    rng = np.random.default_rng(0)
    L = rng.standard_normal((20, 12))
    a = per_source_amplitude(L)
    assert a.shape == (4,)
    assert (a > 0).all()


def test_per_source_amplitude_invalid_columns() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        per_source_amplitude(np.zeros((4, 7)))


def test_predict_meg_from_eeg_perfect_recovery() -> None:
    """No-noise round trip: predict MEG from EEG and recover identically."""
    rng = np.random.default_rng(0)
    L_e = rng.standard_normal((32, 3))
    L_m = rng.standard_normal((100, 3))
    q_true = np.array([1.0, 2.0, -3.0])
    V_obs = L_e @ q_true
    B_pred = predict_meg_from_eeg(L_m, L_e, V_obs)
    np.testing.assert_allclose(B_pred, L_m @ q_true, rtol=1e-9, atol=1e-9)


def test_fit_moment_from_eeg_least_squares() -> None:
    rng = np.random.default_rng(1)
    L_e = rng.standard_normal((32, 3))
    q_true = np.array([0.5, -1.0, 2.0])
    V = L_e @ q_true
    q_fit = fit_moment_from_eeg(L_e, V)
    np.testing.assert_allclose(q_fit, q_true, atol=1e-9)


def test_amplitude_correlation_perfect_when_meg_proportional_to_eeg() -> None:
    rng = np.random.default_rng(2)
    base = rng.standard_normal((20, 30))
    stats = amplitude_correlation(base, 5.0 * base)
    assert isinstance(stats, CrossModalityStats)
    assert stats.pearson_r == pytest.approx(1.0, abs=1e-9)
    assert stats.spearman_rho == pytest.approx(1.0, abs=1e-9)


def test_amplitude_correlation_size_mismatch() -> None:
    with pytest.raises(ValueError, match="source-space size mismatch"):
        amplitude_correlation(np.zeros((4, 6)), np.zeros((4, 9)))


def test_shared_singular_modes_shape() -> None:
    rng = np.random.default_rng(3)
    out = shared_singular_modes(
        rng.standard_normal((30, 12)), rng.standard_normal((20, 12)), n_modes=4
    )
    assert out["meg_singular_values"].shape == (4,)
    assert out["eeg_singular_values"].shape == (4,)
    assert out["mode_overlap"].shape == (4, 4)
    assert (out["mode_overlap"] >= 0).all()
    assert (out["mode_overlap"] <= 1.0 + 1e-9).all()
