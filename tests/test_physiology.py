"""Smoke tests for the moving-dipole physiology simulator."""
from __future__ import annotations

import numpy as np
import pytest

from vagus_fm.io.npz import Leadfield
from vagus_fm.physiology.scenarios import (
    baroreceptor_scenario,
    respiratory_scenario,
)
from vagus_fm.physiology.simulate import (
    SimulatedSignal,
    best_channel_index,
    channel_snr,
    simulate_train,
)


def _make_tiny_leadfield(n_chans: int = 8, n_src: int = 5) -> Leadfield:
    rng = np.random.default_rng(0)
    L = rng.standard_normal((n_chans, 3 * n_src)) * 1e-4
    src_pos = np.zeros((n_src, 3))
    src_pos[:, 2] = np.linspace(1100.0, 1500.0, n_src)
    return Leadfield(
        L=L,
        L_fT_per_nAm=L * 1e6,
        source_pos=src_pos,
        coil_pos=rng.standard_normal((n_chans, 3)) * 100,
        coil_orient=np.tile([0.0, 0.0, 1.0], (n_chans, 1)),
        channel_names=tuple(f"ch{i}" for i in range(n_chans)),
        conductivities=np.array([3e-4, 3e-4, 4.2e-6, 4.3e-4]),
        tissue_labels=("vagus_left", "vagus_right", "bone", "skin"),
    )


def test_baroreceptor_scenario_event_count() -> None:
    sc = baroreceptor_scenario(duration_s=6.0, hr_bpm=60.0)
    assert sc.rate_hz == pytest.approx(1.0)
    # 6 s × 1 Hz + 1 = 7 beats
    assert 6 <= len(sc.events) <= 8
    assert all(ev.n_fibres > 0 for ev in sc.events)
    assert sc.physiology_trace is not None


def test_respiratory_scenario_has_phasic_and_tonic() -> None:
    sc = respiratory_scenario(duration_s=10.0, breath_bpm=6.0)
    rar = [ev for ev in sc.events if ev.label.startswith("resp_RAR_")]
    sar = [ev for ev in sc.events if ev.label.startswith("resp_SAR_")]
    assert len(rar) >= 1, "missing RAR phasic burst"
    assert len(sar) >= 30, "missing SAR tonic train"
    # Tonic events have fewer fibres than phasic
    assert sar[0].n_fibres < rar[0].n_fibres


def test_simulate_train_shape_and_amplitude() -> None:
    lf = _make_tiny_leadfield()
    sc = baroreceptor_scenario(duration_s=2.0, hr_bpm=60.0,
                                n_fibres_per_burst=200, jitter_ms=0.0)
    sim = simulate_train(lf, sc, fs_hz=5000.0)
    assert isinstance(sim, SimulatedSignal)
    assert sim.signal.shape[0] == lf.L.shape[0]
    assert sim.signal.shape[1] == int(2.0 * 5000)
    # signal should have spikes (non-zero)
    assert np.abs(sim.signal).max() > 0.0
    # Q_total trace has peaks corresponding to events
    assert np.abs(sim.Q_total_nAm).max() > 0.5


def test_simulate_train_zero_for_empty_scenario() -> None:
    lf = _make_tiny_leadfield()
    from vagus_fm.physiology.scenarios import Scenario
    empty = Scenario(name="empty", description="", duration_s=1.0,
                     events=[], rate_hz=0.0)
    with pytest.raises(ValueError, match="no events"):
        simulate_train(lf, empty, fs_hz=1000.0)


def test_best_channel_and_snr_helpers() -> None:
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((10, 100))
    # Boost channel 7 — should become best
    sig[7] *= 100
    assert best_channel_index(sig) == 7
    snrs = channel_snr(sig, sigma=1.0, n_trials=4)
    assert snrs.shape == (10,)
    assert snrs[7] > snrs[0]
