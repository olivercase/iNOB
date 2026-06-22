"""Time-domain simulation of vagal CAP trains at every sensor.

For each event in a :class:`~inob.physiology.scenarios.Scenario`, builds a
biphasic compound action potential (Hämäläinen-Q × n_fibres × biphasic
shape) and projects it through the leadfield at the cervical-source hot
spot. The output is a per-channel time series in **the leadfield's native
units** (Tesla for MEG, Volt for EEG; convert to fT/µV for plotting).

Why a stationary-source approximation
-------------------------------------
Conduction-velocity propagation along the nerve happens on ~1 ms × cervical
length, which is comparable to the action-potential duration itself, so
*propagation along the cervical vagus polyline only smears each event by a
fraction of an AP width*. For a 1 nA·m – 1 µA·m source train at 1–10 Hz
event rates, the dominant phenomenon is the temporal pattern (cardiac
locking, respiratory locking), not the millimetre-scale propagation. This
keeps the simulator clean and fast — :func:`inob.sources.cap.cap_signal`
remains available for the propagation-resolved view.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from inob.io.npz import Leadfield
from inob.physiology.scenarios import Scenario
from inob.sources.cap import (
    biphasic_waveform,
    hamalainen_per_fibre_nAm,
    longitudinal_leadfield,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulatedSignal:
    t_s: np.ndarray             # (T,)
    signal: np.ndarray          # (C, T) — leadfield-native units
    Q_total_nAm: np.ndarray     # (T,) total dipole moment vs time, summed over events
    scenario: Scenario
    fs_hz: float
    source_idx: int             # the cervical source position used


def _pick_cervical_source(source_pos_mm: np.ndarray) -> int:
    """Pick the highest-Z source on the polyline (most cervical, closest to neck patch)."""
    return int(np.argmax(source_pos_mm[:, 2]))


def simulate_train(
    leadfield: Leadfield, scenario: Scenario,
    *,
    fs_hz: float = 10_000.0,
    ap_width_ms: float = 0.5,
    source_idx: int | None = None,
) -> SimulatedSignal:
    """Simulate one scenario at every sensor channel.

    Each event contributes a biphasic-Gaussian CAP at its scheduled time
    with total moment ``n_fibres × Q_Hämäläinen``. Per-channel response is
    the longitudinal leadfield column for the chosen source position
    multiplied by the time-varying moment.

    Returns a :class:`SimulatedSignal` whose ``signal`` array is in the
    leadfield's native units (T for MEG, V for EEG); the caller multiplies
    by 1e15 (MEG → fT) or 1e6 (EEG → µV) for human-readable plotting.
    """
    if not scenario.events:
        raise ValueError(f"scenario {scenario.name!r} has no events")

    L_long, _arc, _tan = longitudinal_leadfield(leadfield.L, leadfield.source_pos)
    if source_idx is None:
        source_idx = _pick_cervical_source(leadfield.source_pos)
    L_col = L_long[:, source_idx]          # (C,)  T or V per A·m of moment

    n_total = round(scenario.duration_s * fs_hz)
    t_s = np.arange(n_total, dtype=np.float64) / fs_hz

    # Build the time-varying moment trace M(t) — sum of biphasic CAPs at
    # each event time, scaled by per-event total dipole moment.
    M_Am = np.zeros(n_total, dtype=np.float64)
    for ev in scenario.events:
        q_per_fibre_nAm = hamalainen_per_fibre_nAm(
            ev.fibres, action_potential_mV=ev.ap_amplitude_mV,
        )
        q_event_Am = q_per_fibre_nAm * ev.n_fibres * 1e-9
        if q_event_Am <= 0:
            continue
        # AP centred at ev.t_start_s (window is ±5σ of the gaussian)
        t_rel_ms = (t_s - ev.t_start_s) * 1000.0
        # Crop window for speed
        within = np.abs(t_rel_ms) <= 5.0 * ap_width_ms * 4.0   # ±20 ms-ish
        if not within.any():
            continue
        shape = biphasic_waveform(t_rel_ms[within], ap_width_ms=ap_width_ms)
        M_Am[within] += q_event_Am * shape

    # signal = L_col × M_Am  →  (C, T)
    signal = L_col[:, None] * M_Am[None, :]

    logger.info(
        "[simulate] %s  ·  n_events=%d  ·  fs=%g Hz  ·  source #%d (z=%.0f mm)  ·  "
        "peak |signal|=%.3e (raw units), peak |M|=%.3f nA·m",
        scenario.name, len(scenario.events), fs_hz, source_idx,
        leadfield.source_pos[source_idx, 2],
        float(np.abs(signal).max()),
        float(np.abs(M_Am).max() * 1e9),
    )
    return SimulatedSignal(
        t_s=t_s, signal=signal,
        Q_total_nAm=M_Am * 1e9,
        scenario=scenario, fs_hz=float(fs_hz), source_idx=int(source_idx),
    )


def best_channel_index(sig: np.ndarray) -> int:
    """Index of the highest-RMS channel."""
    return int(np.argmax(np.sqrt(np.mean(sig ** 2, axis=1))))


def channel_snr(
    sig: np.ndarray, sigma: float, *, n_trials: int = 1,
) -> np.ndarray:
    """Predicted post-averaging peak SNR per channel."""
    return np.abs(sig).max(axis=1) / max(sigma, 1e-30) * np.sqrt(max(n_trials, 1))
