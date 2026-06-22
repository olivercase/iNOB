"""Physiologically-grounded vagal CAP event-train scenarios.

Two scenarios that produce structured cervical-vagus afferent activity:

  * **Baroreceptor**: the carotid-sinus / aortic-arch baroreceptor afferents
    fire in cardiac-locked bursts (one burst per R-wave, ≈ 100–200 ms after
    systole onset). Predominantly large-myelinated A-fibres (Aronson 1980;
    Pelot et al. 2020; Bu et al. 2024 cite this fibre class explicitly).
    Resting heart rate ≈ 60–80 bpm; we use 70 bpm = 1.17 Hz.

  * **Deep breathing (respiratory)**: pulmonary stretch-receptor afferents
    in the lungs project up the vagus. Two populations
    (Kubin et al. 2006 *Anat Rec*, Pelot et al. 2020):

      RAR  rapidly-adapting receptors fire a phasic burst at inspiration
           onset, bigger volume change → larger burst (deep breathing).
      SAR  slowly-adapting receptors fire a tonic train throughout
           inspiration, modulated by lung volume.

    Slow / deep-breathing rate ≈ 6 breaths/min = 0.10 Hz, with longer
    inspiration (≈ 2 s) and stronger phasic component.

Each scenario generates a list of :class:`CapEvent`. The simulator in
:mod:`inob.physiology.simulate` superposes the per-event CAP responses
to produce a time-domain signal at every MEG / EEG sensor.

References
----------
* Bu Y *et al.* 2024 *Comm Biol* 7:893 — measurement of cervical-vagus
  baroreceptor signature with OPMs.
* Pelot NA *et al.* 2017 *Front Neurosci* 12:601 — vagal fibre populations.
* Kubin L *et al.* 2006 *Anat Rec* 288A:961 — pulmonary RAR/SAR projections.
* Hämäläinen M *et al.* 1993 *Rev Mod Phys* 65:413 — Q = π·d²·σ_in·ΔV/4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from inob.sources.cap import (
    FibreDistribution,
    lognormal_fibre_distribution,
)


@dataclass(frozen=True)
class CapEvent:
    """A single afferent firing event.

    The simulator turns each event into a propagating CAP scaled by
    ``n_fibres`` × the per-fibre Hämäläinen dipole moment derived from
    ``fibres`` and ``ap_amplitude_mV``.
    """
    t_start_s: float
    n_fibres: int
    fibres: FibreDistribution
    ap_amplitude_mV: float = 70.0
    label: str = ""


@dataclass(frozen=True)
class Scenario:
    """A named, plottable event train."""
    name: str
    description: str
    duration_s: float
    events: list[CapEvent] = field(default_factory=list)
    rate_hz: float = 0.0
    physiology_trace_label: str = ""    # "ECG (a.u.)" or "Lung volume (a.u.)"
    physiology_trace: np.ndarray | None = None    # (T,) optional context plot


# ── A-fibre and mixed-fibre helpers ────────────────────────────────────────

def a_fibre_population(
    *, mean_um: float = 8.0, sigma_log: float = 0.30, n_bins: int = 20,
) -> FibreDistribution:
    """Lognormal fibre-diameter distribution centred on the A-myelinated
    range (mean diameter ~8 µm). Conduction velocity ~k·d ≈ 50 m/s mean."""
    return lognormal_fibre_distribution(
        mean_um=mean_um, sigma_log=sigma_log,
        n_bins=n_bins, lo_um=2.0, hi_um=15.0,
    )


def mixed_pulmonary_population(
    *, mean_um: float = 5.0, sigma_log: float = 0.45, n_bins: int = 25,
) -> FibreDistribution:
    """Mixed Aδ + small-A diameter distribution typical of pulmonary
    afferents. Slower mean CV than baroreceptor A-fibres."""
    return lognormal_fibre_distribution(
        mean_um=mean_um, sigma_log=sigma_log,
        n_bins=n_bins, lo_um=1.0, hi_um=12.0,
    )


# ── scenario constructors ──────────────────────────────────────────────────

def baroreceptor_scenario(
    *,
    duration_s: float = 6.0,
    hr_bpm: float = 70.0,
    n_fibres_per_burst: int = 200,
    jitter_ms: float = 5.0,
    seed: int = 0,
) -> Scenario:
    """Cardiac-locked baroreceptor train.

    Each R-wave triggers one A-fibre burst ~150 ms later with ~200 fibres
    (Bu 2024 baroreceptor estimate). The simulated ECG trace is a
    placeholder (gaussian-derivative R-waves) for visual context only.
    """
    hr_hz = hr_bpm / 60.0
    rng = np.random.default_rng(seed)
    n_beats = int(duration_s * hr_hz) + 1
    fibres = a_fibre_population()

    events: list[CapEvent] = []
    for k in range(n_beats):
        t_R = k / hr_hz
        # baroreceptor latency ≈ 150 ms after R-wave
        t_burst = t_R + 0.15 + float(rng.normal(0.0, jitter_ms * 1e-3))
        n = int(n_fibres_per_burst + rng.integers(-20, 21))
        events.append(CapEvent(
            t_start_s=float(t_burst), n_fibres=max(n, 1), fibres=fibres,
            ap_amplitude_mV=70.0, label=f"baro_beat_{k:02d}",
        ))

    # Synthetic ECG-like trace: gaussian-derivative R-waves at hr_hz
    fs = 1000
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs
    ecg = np.zeros(n_samples)
    for k in range(n_beats):
        t_R = k / hr_hz
        sigma = 0.02
        ecg += -((t - t_R) / sigma) * np.exp(-((t - t_R) ** 2) / (2 * sigma ** 2))
    if np.abs(ecg).max() > 0:
        ecg /= np.abs(ecg).max()

    return Scenario(
        name="baroreceptor",
        description=(
            f"Carotid-sinus baroreceptor afferents at HR = {hr_bpm:g} bpm "
            f"({hr_hz:.2f} Hz), {n_fibres_per_burst} A-fibres per burst, "
            f"150 ms after R-wave, ±{jitter_ms:g} ms jitter."
        ),
        duration_s=duration_s,
        events=events,
        rate_hz=hr_hz,
        physiology_trace_label="ECG R-waves (a.u.)",
        physiology_trace=ecg,
    )


def respiratory_scenario(
    *,
    duration_s: float = 12.0,
    breath_bpm: float = 6.0,
    inspiration_fraction: float = 0.45,
    n_phasic_fibres_RAR: int = 600,
    n_tonic_fibres_SAR: int = 80,
    sar_rate_hz: float = 50.0,
    seed: int = 0,
) -> Scenario:
    """Slow / deep breathing — RAR phasic burst + SAR tonic train per inspiration.

    Defaults: 6 breaths / minute (deep breathing), inspiration 45% of cycle,
    one large RAR burst at insp-onset, then SAR tonic at 50 Hz throughout
    inspiration. Expiration is silent.
    """
    breath_hz = breath_bpm / 60.0
    period_s = 1.0 / breath_hz
    insp_dur_s = inspiration_fraction * period_s
    rng = np.random.default_rng(seed)
    n_breaths = int(duration_s * breath_hz) + 1
    fibres = mixed_pulmonary_population()

    events: list[CapEvent] = []
    for k in range(n_breaths):
        t0 = k * period_s
        # RAR phasic burst right at inspiration onset
        events.append(CapEvent(
            t_start_s=float(t0), n_fibres=n_phasic_fibres_RAR,
            fibres=fibres, ap_amplitude_mV=80.0,
            label=f"resp_RAR_{k:02d}",
        ))
        # SAR tonic: events evenly spaced through inspiration
        n_sar = int(insp_dur_s * sar_rate_hz)
        for j in range(n_sar):
            t_event = t0 + (j + 0.5) / sar_rate_hz
            jitter = rng.normal(0.0, 1e-3)
            events.append(CapEvent(
                t_start_s=float(t_event + jitter),
                n_fibres=n_tonic_fibres_SAR + int(rng.integers(-10, 11)),
                fibres=fibres, ap_amplitude_mV=80.0,
                label=f"resp_SAR_{k:02d}_{j:03d}",
            ))

    # Lung-volume trace: rising during inspiration, falling during expiration
    fs = 1000
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs
    lv = np.zeros(n_samples)
    for k in range(n_breaths):
        t0 = k * period_s
        in_insp = (t >= t0) & (t < t0 + insp_dur_s)
        in_exp = (t >= t0 + insp_dur_s) & (t < t0 + period_s)
        # Smooth ramp up during inspiration
        if in_insp.any():
            lv[in_insp] = np.sin(np.pi * (t[in_insp] - t0) / (2 * insp_dur_s))
        if in_exp.any():
            decay_t = (t[in_exp] - t0 - insp_dur_s) / (period_s - insp_dur_s)
            lv[in_exp] = np.cos(np.pi * decay_t / 2)

    return Scenario(
        name="respiratory_deep",
        description=(
            f"Deep breathing at {breath_bpm:g} bpm ({breath_hz:.2f} Hz). "
            f"Each cycle: RAR burst ({n_phasic_fibres_RAR} fibres) at insp-onset "
            f"+ SAR tonic ({n_tonic_fibres_SAR} fibres @ {sar_rate_hz:g} Hz) "
            f"through {inspiration_fraction:.0%} of cycle."
        ),
        duration_s=duration_s,
        events=events,
        rate_hz=breath_hz,
        physiology_trace_label="Lung volume (a.u.)",
        physiology_trace=lv,
    )
