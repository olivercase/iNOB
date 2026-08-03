"""Physiologically-grounded CAP event-train scenarios, per target.

Which scenarios apply to a run is decided by the target's
:class:`~inob.physiology.profiles.PhysiologyProfile`, not by this module.

**Vagus** — two scenarios of spontaneous cervical-vagus afferent activity
(baroreceptor, deep breathing), described below.

**Spine** — two evoked scenarios at the bottom of this file
(:func:`median_nerve_ssep_scenario`, :func:`tibial_nerve_ssep_scenario`). The
cord's counterpart to spontaneous vagal traffic is the stimulus-evoked
dorsal-column volley, which is what SSEP and magnetospinography actually
record. Unlike the vagal scenarios these carry a ``generator_z_mm``, because
the entry segment — cervical for median nerve, lumbosacral for tibial — is
what determines both the depth under the sensors and the distance the volley
travels through the imaged cord.

The vagal scenarios:

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

.. PHYSIOLOGY-TODO: muscle (magnetomyography) dynamics — NOT IMPLEMENTED.
   Both scenarios above are *vagal afferent* event trains. The muscle
   source-target has a **static** forward model (fibre-aligned anisotropic
   conductivity, ``forward.muscle_anisotropy``) and, since the muscle fibre/
   CV/AP-amplitude fix, a muscle-specific single-event physiology in
   :data:`inob.physiology.profiles.MUSCLE_PROFILE` (d ≈ 40–80 µm fibres,
   CV ≈ 4 m/s, matched to :data:`inob.viz.detectability.MUSCLE_SCENARIOS`).
   That single-event model can represent one synchronous fibre volley (the
   evoked-M-wave case) but nothing beyond it — there is still no MUAP/motor-
   unit/recruitment layer, so any *time-domain, multi-event* muscle output
   (an event train from cap_compare/physiology_plot/simulate, i.e. voluntary
   or spontaneous EMG rather than a single stimulated volley) remains
   PROVISIONAL and un-buildable (``scenario_builder=None``).

   The deferred muscle-dynamics module would need:
     * a MUAP waveform + motor-unit model (100s–1000s of fibres per MU,
       innervation-zone origin, propagation to both tendons);
     * recruitment / rate-coding (size principle, 8–30 Hz firing, asynchronous
       MUs → interference EMG rather than a synchronous compound AP);
     * an activation scenario (isometric hold, twitch, evoked M-wave) built
       as a proper event train, the way the SSEP scenarios are for spine.
   Grep ``PHYSIOLOGY-TODO`` before publishing any time-domain muscle figure.

   A second, separate gap: :mod:`inob.sources.muscle` places sources as a
   3-D volume-fill of the whole muscle belly, not a 1-D ordered path along
   a fibre axis. :mod:`inob.sources.cap`'s propagating-wavefront model
   (used by ``cap_compare``) assumes source positions are arc-length-
   ordered along a single path, as they are for the vagus bundle and the
   cord — that assumption does not hold for muscle's point cloud, so any
   "propagating vs stationary" comparison for muscle is only as meaningful
   as :data:`inob.physiology.profiles.MUSCLE_PROFILE.propagation_span_mm`,
   bounded small specifically to avoid computing over the (physically
   meaningless) cumulative arc length of an unordered point cloud. Fixing
   this properly needs a fibre-ordered source path for muscle, not just a
   profile-parameter change.

References
----------
* Bu Y *et al.* 2024 *Comm Biol* 7:893 — measurement of cervical-vagus
  baroreceptor signature with OPMs.
* Pelot NA *et al.* 2017 *Front Neurosci* 12:601 — vagal fibre populations.
* Kubin L *et al.* 2006 *Anat Rec* 288A:961 — pulmonary RAR/SAR projections.
* Hämäläinen M *et al.* 1993 *Rev Mod Phys* 65:413 — Q = π·d²·σ_in·ΔV/4.
* Cruccu G *et al.* 2008 *Clin Neurophysiol* 119:1705 — SSEP recording standards.
* Kawabata S *et al.* 2002 *Clin Neurophysiol* 113:1874 — magnetospinography of
  the ascending cord volley.
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
    generator_z_mm: float | None = None
    """Axial position of the generator, mm in the atlas frame.

    ``None`` keeps the historical behaviour of using the most rostral source on
    the polyline (right for a cervical vagal generator). Spinal SSEP scenarios
    set it to the entry segment, because a median-nerve volley enters the cord
    at C6–T1 and a tibial-nerve volley at the conus — ~400 mm apart, and at
    very different depths under the sensor array.
    """
    ap_width_ms: float | None = None
    """Per-scenario AP width override, ms. ``None`` uses the caller's default."""


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


def dorsal_column_population(
    *, mean_um: float = 10.0, sigma_log: float = 0.25, n_bins: int = 20,
) -> FibreDistribution:
    """Dorsal-column ascending afferents — see
    :func:`inob.physiology.profiles.dorsal_column_population`."""
    from inob.physiology.profiles import (
        dorsal_column_population as _dc,
    )
    return _dc(mean_um=mean_um, sigma_log=sigma_log, n_bins=n_bins)


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


# ── spinal somatosensory evoked scenarios ──────────────────────────────────
#
# The spinal cord's analogue of the vagal CAP is not spontaneous traffic but
# the evoked ascending volley: stimulate a peripheral nerve, and a synchronous
# large-myelinated discharge enters the cord at that nerve's root segment and
# ascends the dorsal columns. This is the paradigm behind clinical SSEP and
# behind magnetospinography, and it maps directly onto the package's planning
# question, because both already work by averaging many stimulus repetitions.
#
# Entry-segment Z values are for the BodyParts3D cord (z = 1031..1482 mm, most
# rostral at the top). They are defaults, not constants — override per subject.
MEDIAN_NERVE_ENTRY_Z_MM: float = 1390.0   # C6-T1, cervical enlargement
TIBIAL_NERVE_ENTRY_Z_MM: float = 1060.0   # L4-S1, lumbosacral enlargement / conus


def _stimulus_marker_trace(
    duration_s: float, rate_hz: float, n_stim: int, fs: int = 1000,
) -> np.ndarray:
    """Unit impulse at each stimulus time, for the context panel."""
    n_samples = int(duration_s * fs)
    trace = np.zeros(n_samples)
    for k in range(n_stim):
        idx = int(k / rate_hz * fs)
        if 0 <= idx < n_samples:
            trace[idx] = 1.0
    return trace


def _ssep_scenario(
    *, name: str, nerve: str, entry_z_mm: float, segment: str,
    duration_s: float, rate_hz: float, n_fibres: int,
    ap_amplitude_mV: float, ap_width_ms: float, jitter_ms: float, seed: int,
) -> Scenario:
    """Shared builder for peripheral-nerve SSEP volleys."""
    rng = np.random.default_rng(seed)
    n_stim = int(duration_s * rate_hz) + 1
    fibres = dorsal_column_population()

    events: list[CapEvent] = []
    for k in range(n_stim):
        t_stim = k / rate_hz
        # Peripheral conduction delay from stimulator to cord entry is absorbed
        # into the event time; what matters downstream is the spacing and the
        # trial-to-trial jitter, not the absolute latency.
        t_entry = t_stim + float(rng.normal(0.0, jitter_ms * 1e-3))
        events.append(CapEvent(
            t_start_s=float(t_entry),
            n_fibres=n_fibres,
            fibres=fibres,
            ap_amplitude_mV=ap_amplitude_mV,
            label=f"{name}_stim_{k:03d}",
        ))

    return Scenario(
        name=name,
        description=(
            f"{nerve} stimulation at {rate_hz:g} Hz. Each stimulus evokes a "
            f"synchronous dorsal-column volley of {n_fibres} large-myelinated "
            f"afferents entering the cord at {segment} (z = {entry_z_mm:.0f} mm) "
            f"and ascending rostrally."
        ),
        duration_s=duration_s,
        events=events,
        rate_hz=rate_hz,
        physiology_trace_label=f"{nerve} stimulus",
        physiology_trace=_stimulus_marker_trace(duration_s, rate_hz, n_stim),
        generator_z_mm=entry_z_mm,
        ap_width_ms=ap_width_ms,
    )


def median_nerve_ssep_scenario(
    *,
    duration_s: float = 4.0,
    rate_hz: float = 4.7,
    n_fibres: int = 800,
    entry_z_mm: float = MEDIAN_NERVE_ENTRY_Z_MM,
    ap_amplitude_mV: float = 80.0,
    ap_width_ms: float = 0.7,
    jitter_ms: float = 0.2,
    seed: int = 0,
) -> Scenario:
    """Median-nerve SSEP — cervical entry, the standard clinical montage.

    The 4.7 Hz default is the usual non-integer clinical stimulation rate
    (Cruccu et al. 2008): it avoids locking to 50/60 Hz mains harmonics, so
    line noise averages out across trials instead of summing coherently.
    The evoked volley enters at C6-T1 and ascends the dorsal columns — the
    response that dominates cervical magnetospinography.
    """
    return _ssep_scenario(
        name="ssep_median", nerve="Median nerve", entry_z_mm=entry_z_mm,
        segment="C6-T1", duration_s=duration_s, rate_hz=rate_hz,
        n_fibres=n_fibres, ap_amplitude_mV=ap_amplitude_mV,
        ap_width_ms=ap_width_ms, jitter_ms=jitter_ms, seed=seed,
    )


def tibial_nerve_ssep_scenario(
    *,
    duration_s: float = 6.0,
    rate_hz: float = 3.1,
    n_fibres: int = 600,
    entry_z_mm: float = TIBIAL_NERVE_ENTRY_Z_MM,
    ap_amplitude_mV: float = 80.0,
    ap_width_ms: float = 0.9,
    jitter_ms: float = 0.4,
    seed: int = 0,
) -> Scenario:
    """Tibial-nerve SSEP — lumbosacral entry, the hard case for detection.

    Differs from the median-nerve scenario in three ways that all reduce
    detectability, which is exactly why it is worth simulating: the volley
    enters ~330 mm more caudally (deeper under thicker tissue and further from
    the cervical sensors), fewer afferents are recruited, and the longer
    peripheral path disperses the volley across conduction velocities, giving
    a broader and lower-amplitude response. Clinical practice compensates with
    a slower rate and more averages.
    """
    return _ssep_scenario(
        name="ssep_tibial", nerve="Tibial nerve", entry_z_mm=entry_z_mm,
        segment="L4-S1", duration_s=duration_s, rate_hz=rate_hz,
        n_fibres=n_fibres, ap_amplitude_mV=ap_amplitude_mV,
        ap_width_ms=ap_width_ms, jitter_ms=jitter_ms, seed=seed,
    )
