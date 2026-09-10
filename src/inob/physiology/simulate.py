"""Time-domain simulation of CAP event trains at every sensor.

For each event in a :class:`~inob.physiology.scenarios.Scenario`, builds a
biphasic compound action potential (Hämäläinen-Q × n_fibres × biphasic
shape) and projects it through the leadfield column for the scenario's
generator position. The output is a per-channel time series in **the
leadfield's native units** (Tesla for MEG, Volt for EEG; convert to fT/µV
for plotting).

The stationary-source approximation, and when it holds
------------------------------------------------------
This simulator lumps each event into one stationary dipole. That is a
*target-dependent* approximation, not a universal one, and each target's
:class:`~inob.physiology.profiles.PhysiologyProfile` records whether it is
expected to hold via ``stationary_ok``:

  * **Vagus** (``stationary_ok=True``) — baroreceptor afferents are localised
    to the ~50 mm cervical bundle, crossed in ~1 ms at 47 m/s, comparable to
    the 0.5 ms AP width. Propagation smears each event by a fraction of an AP
    width, and the dominant phenomenon is the temporal pattern (cardiac,
    respiratory locking) rather than millimetre-scale propagation.
  * **Spine** (``stationary_ok=False``) — the volley ascends the whole imaged
    cord, ~450 mm at 59 m/s ≈ 7.6 ms, an order of magnitude longer than its
    0.7 ms AP width. The lumped model overestimates the peak roughly threefold.
    Use it for event-train timing and rate structure; for single-event
    amplitude or waveform use the propagation-resolved path.

:mod:`inob.viz.cap_compare` renders and *measures* that comparison for the
selected target, and :func:`inob.sources.cap.cap_signal` provides the
propagation-resolved signal directly.
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
    t_s: np.ndarray  # (T,)
    signal: np.ndarray  # (C, T) — leadfield-native units
    Q_total_nAm: np.ndarray  # (T,) total dipole moment vs time, summed over events
    scenario: Scenario
    fs_hz: float
    source_idx: int  # the cervical source position used


def _pick_rostral_source(source_pos_mm: np.ndarray) -> int:
    """Pick the highest-Z source on the polyline (most rostral, nearest the neck)."""
    return int(np.argmax(source_pos_mm[:, 2]))


def _pick_source_at_z(source_pos_mm: np.ndarray, z_mm: float) -> int:
    """Index of the source closest to axial position ``z_mm``."""
    return int(np.argmin(np.abs(source_pos_mm[:, 2] - z_mm)))


def resolve_source_index(
    source_pos_mm: np.ndarray,
    scenario: Scenario,
) -> int:
    """Which source position generates ``scenario``'s events.

    A scenario that names its generator (``generator_z_mm`` — the spinal SSEP
    scenarios do, because a median-nerve volley enters the cord ~330 mm rostral
    to a tibial-nerve one) is placed there. Otherwise we fall back to the most
    rostral source, which is the right default for a cervical vagal generator.
    """
    if scenario.generator_z_mm is None:
        return _pick_rostral_source(source_pos_mm)
    idx = _pick_source_at_z(source_pos_mm, scenario.generator_z_mm)
    err = abs(float(source_pos_mm[idx, 2]) - scenario.generator_z_mm)
    if err > 25.0:
        logger.warning(
            "scenario %r wants a generator at z=%.0f mm but the nearest modelled "
            "source is at z=%.0f mm (%.0f mm away) — the solved region may not "
            "cover this generator",
            scenario.name,
            scenario.generator_z_mm,
            float(source_pos_mm[idx, 2]),
            err,
        )
    return idx


def simulate_train(
    leadfield: Leadfield,
    scenario: Scenario,
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
        source_idx = resolve_source_index(leadfield.source_pos, scenario)
    L_col = L_long[:, source_idx]  # (C,)  T or V per A·m of moment
    # A scenario may carry its own AP width (spinal volleys are broader than
    # vagal ones, and a tibial volley is broader still than a median one).
    if scenario.ap_width_ms is not None:
        ap_width_ms = scenario.ap_width_ms

    n_total = round(scenario.duration_s * fs_hz)
    t_s = np.arange(n_total, dtype=np.float64) / fs_hz

    # Build the time-varying moment trace M(t) — sum of biphasic CAPs at
    # each event time, scaled by per-event total dipole moment.
    M_Am = np.zeros(n_total, dtype=np.float64)
    for ev in scenario.events:
        q_per_fibre_nAm = hamalainen_per_fibre_nAm(
            ev.fibres,
            action_potential_mV=ev.ap_amplitude_mV,
        )
        q_event_Am = q_per_fibre_nAm * ev.n_fibres * 1e-9
        if q_event_Am <= 0:
            continue
        # AP centred at ev.t_start_s (window is ±5σ of the gaussian)
        t_rel_ms = (t_s - ev.t_start_s) * 1000.0
        # Crop window for speed
        within = np.abs(t_rel_ms) <= 5.0 * ap_width_ms * 4.0  # ±20 ms-ish
        if not within.any():
            continue
        shape = biphasic_waveform(t_rel_ms[within], ap_width_ms=ap_width_ms)
        M_Am[within] += q_event_Am * shape

    # signal = L_col × M_Am  →  (C, T)
    signal = L_col[:, None] * M_Am[None, :]

    logger.info(
        "[simulate] %s  ·  n_events=%d  ·  fs=%g Hz  ·  source #%d (z=%.0f mm)  ·  "
        "peak |signal|=%.3e (raw units), peak |M|=%.3f nA·m",
        scenario.name,
        len(scenario.events),
        fs_hz,
        source_idx,
        leadfield.source_pos[source_idx, 2],
        float(np.abs(signal).max()),
        float(np.abs(M_Am).max() * 1e9),
    )
    return SimulatedSignal(
        t_s=t_s,
        signal=signal,
        Q_total_nAm=M_Am * 1e9,
        scenario=scenario,
        fs_hz=float(fs_hz),
        source_idx=int(source_idx),
    )


def best_channel_index(sig: np.ndarray) -> int:
    """Index of the highest-RMS channel."""
    return int(np.argmax(np.sqrt(np.mean(sig**2, axis=1))))


def channel_snr(
    sig: np.ndarray,
    sigma: float,
    *,
    n_trials: int = 1,
) -> np.ndarray:
    """Predicted post-averaging peak SNR per channel."""
    return np.abs(sig).max(axis=1) / max(sigma, 1e-30) * np.sqrt(max(n_trials, 1))
