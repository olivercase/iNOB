"""Detectability analysis — given N trials and a noise floor, can we see it?

Predicted signal-to-noise ratio for the target's compound action potential,
plotted as a function of:

  * source dipole moment  Q  — each target planned against the Q range its own
    literature supports, never another target's (see
    :func:`scenarios_for_target`);
  * number of averaged trials  N  (white-noise SNR scales as √N);
  * per-modality noise floor (OPM intrinsic + EEG amplifier + Johnson,
    integrated over the recording passband in ``cfg.noise``).

Detection threshold convention: SNR ≥ 3 (Rose criterion / standard
post-averaging visibility). Other thresholds are easy to read off.

What counts as "signal"
-----------------------
MEG uses the best-channel peak: each channel measures a field directly. EEG
uses the best *bipolar pair*, because a surface potential exists only as a
difference between contacts — see
:func:`inob.analysis.snr.per_source_best_bipolar` for why the single-channel
peak understates what an electrode array measures, and by how much.

For evoked targets the trials axis also carries the clinical averaging budget
(:data:`CLINICAL_AVERAGE_BUDGET`): landing inside it means the measurement fits
a protocol that already exists, which is the actual feasibility question.

Trial-count floor: the required-trials figure is ``(SNR_target·σ/signal)²``,
floored at 1. Strong sources (e.g. a 20–70 nA·m CAP on the best channel) can
clear the threshold in a single trial, where the raw formula returns a
fractional trial count that has no physical meaning — you cannot average a
fraction of a trial. Such sources are reported as "detectable in one trial"
(N = 1), and the recording-time panel bottoms out at one trial period (1/rate).
"""
from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from inob.analysis.propagation import is_ordered_polyline
from inob.analysis.snr import (
    compute_noise_floors,
    per_source_best_bipolar,
    per_source_peak,
)
from inob.config import (
    Config,
    source_region_label,
    source_target_tag,
    target_output,
)
from inob.io.npz import load_leadfield
from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
    save_figure,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectabilityScenario:
    """A named source-strength + per-trial-event-rate scenario."""
    label: str
    Q_nAm: float
    description: str = ""


# Vagus source-strength ladder, deliberately 1-20 nA·m — the same rungs the
# cord takes (see spine_scenarios), so the two targets can be read on one axis
# and any difference between them is geometry rather than a different assumed
# source.
#
# The ladder used to top out at 70 nA·m, Bu et al. 2024's full A+C-fibre
# summation. That figure is real but it is an upper bound on a maximally
# synchronous nerve-wide event, an order of magnitude above the cord's
# magnetospinography anchor and two orders above one 200-fibre baroreceptor
# burst (0.74 nA·m, VAGUS_PROFILE). Planning against it flattered every vagus
# result. It is kept in the top rung's description rather than as a rung, so
# the reference is not lost; `--q-nAm 70` still runs it explicitly.
DEFAULT_SCENARIOS: tuple[DetectabilityScenario, ...] = (
    DetectabilityScenario("Q = 1 nA·m  (calibration unit)", 1.0,
                           "Reference scale, leadfield calibration."),
    DetectabilityScenario("Q = 5.11 nA·m  (cord anchor)", 5.11,
                           "The cervical-cord magnetospinography figure, run "
                           "here so vagus and spine share a rung exactly."),
    DetectabilityScenario("Q = 10 nA·m  (modest CAP)", 10.0,
                           "Reflex / mild evoked activation."),
    DetectabilityScenario("Q = 20 nA·m  (strong CAP)", 20.0,
                           "Top of the planning range. Full A+C summation "
                           "(70 nA·m, Bu et al. 2024) is 3.5x higher again and "
                           "is an upper bound, not a planning figure."),
)

# Muscle (magnetomyography) source-strength range. Muscle fibres are ~60 µm —
# an order of magnitude fatter than the ~8 µm vagal A-fibres — so the per-fibre
# Hämäläinen moment (Q ∝ d²) is far larger, and a motor unit fires 100s of
# fibres near-synchronously. Static equivalent-current-dipole magnitudes only —
# recruitment/firing dynamics are deliberately NOT modelled here; see the
# PHYSIOLOGY-TODO block in :mod:`inob.physiology.scenarios` for the deferred
# muscle dynamics work. Fibre/AP constants below match
# :data:`inob.physiology.profiles.MUSCLE_PROFILE` (mean d=60 um,
# sigma_in=1 S/m — the same Hamalainen convention used for vagus/spine,
# not the peripheral-axon value — ap_amplitude=90 mV), so the two figures'
# magnitude assumptions agree:
#   per-fibre Q ≈ π·(60 µm)²·σ_in(1.0)·ΔV(0.09 V)/4 ≈ 0.25 nA·m
#   single MUAP  ≈ 40 fibres near-synchronous (MUSCLE_PROFILE.n_fibres) → ~10 nA·m
#   weak voluntary (few MUs recruited) → ~50 nA·m
#   moderate voluntary contraction     → ~200 nA·m
#   evoked compound M-wave (whole-muscle synchronous) → ~1000 nA·m
# Refs: Hämäläinen 1993 (Q = π d² σ ΔV/4); Cohen & Givler 1972; Broser 2018/2021
# (OPM-MMG single-MU fields); Farina & Merletti 2004 (MU / MUAP physiology).
MUSCLE_SCENARIOS: tuple[DetectabilityScenario, ...] = (
    DetectabilityScenario("Q = 10 nA·m  (single MUAP)", 10.0,
                           "One motor unit, ~10²–10³ fibres near-synchronous."),
    DetectabilityScenario("Q = 50 nA·m  (weak contraction)", 50.0,
                           "A few motor units recruited (low voluntary force)."),
    DetectabilityScenario("Q = 200 nA·m  (moderate contraction)", 200.0,
                           "Many motor units; moderate voluntary force."),
    DetectabilityScenario("Q = 1000 nA·m  (evoked M-wave)", 1000.0,
                           "Whole-muscle synchronous compound MAP (stimulated)."),
)


# Spinal-cord source-strength range, anchored the same way the vagus range is.
# The vagus ladder tops out at Bu et al. 2024's measured cervical-vagus figure;
# the spine's equivalent literature anchor is the equivalent current dipole of
# the cervical dorsal-column volley measured by magnetospinography (Kawabata
# et al. 2002 Clin Neurophysiol 113:1874; Sasaki et al. 2008 Spine 33:E836),
# which is the single-nA·m range. That anchor already lives in
# :data:`inob.physiology.profiles.SPINE_PROFILE` — it is pulled from there
# rather than restated, so the detectability figure and the time-domain
# figures cannot drift apart on the source strength.
#
# Using DEFAULT_SCENARIOS for the spine (as this did) silently planned the
# spine against the vagus's 70 nA·m full-summation figure, which is an order
# of magnitude above anything reported for the cord.
def spine_scenarios() -> tuple[DetectabilityScenario, ...]:
    """Q ladder for the spinal SSEP, built around the literature anchor."""
    from inob.physiology.profiles import SPINE_PROFILE
    anchor = SPINE_PROFILE.default_strength_nAm
    return (
        DetectabilityScenario("Q = 1 nA·m  (calibration unit)", 1.0,
                              "Reference scale; also the low end of reported "
                              "cervical-cord equivalent dipoles."),
        DetectabilityScenario(f"Q = {anchor:.2f} nA·m  (median-nerve SSEP)", anchor,
                              "Magnetospinography-derived cervical volley "
                              "(Kawabata 2002, Sasaki 2008) — the anchor."),
        DetectabilityScenario("Q = 10 nA·m  (strong volley)", 10.0,
                              "Upper end of the reported cord-ECD range."),
        DetectabilityScenario("Q = 20 nA·m  (optimistic bound)", 20.0,
                              "Above anything reported for the cord; shown as "
                              "a bound, not an expectation."),
    )


def fixed_q_scenarios(*Q_nAm: float) -> tuple[DetectabilityScenario, ...]:
    """A ladder at explicit source strengths, for cross-target comparisons.

    Each target's own ladder is anchored to its own literature (see
    :func:`scenarios_for_target`), so the ladders are not comparable rung for
    rung. Running one target at another's anchor — e.g. the vagus at the
    spine's 5.11 nA·m magnetospinography figure — asks the different question
    "how does this geometry do at *that* source strength?", and needs the Q
    stated explicitly rather than pulled from a profile.

    Several values give several rungs, which is how the vagus gets plotted over
    the cord's 1–20 nA·m range: same rungs, same axes, only the geometry and
    the sensor distance differ.
    """
    if not Q_nAm:
        raise ValueError("fixed_q_scenarios needs at least one source strength")
    return tuple(
        DetectabilityScenario(f"Q = {q:g} nA·m", float(q),
                              "Explicit source strength (--q-nAm), not the "
                              "target's own physiology anchor.")
        for q in sorted(float(q) for q in Q_nAm)
    )


def scenarios_for_target(cfg: Config) -> tuple[DetectabilityScenario, ...]:
    """Pick the source-strength scenario set matching the forward target.

    Each target uses the Q range its own literature supports: muscle the
    magnetomyography range (:data:`MUSCLE_SCENARIOS`), spine the
    magnetospinography range (:func:`spine_scenarios`), vagus the cervical-vagus
    range (:data:`DEFAULT_SCENARIOS`).
    """
    tag = source_target_tag(cfg)
    if tag == "muscle":
        return MUSCLE_SCENARIOS
    if tag.startswith("spine"):
        return spine_scenarios()
    return DEFAULT_SCENARIOS


# Canonical best-channel peak signal lives in inob.analysis.snr so the figure
# and the GUI detect endpoint never disagree. Thin alias kept for readability.
def per_source_peak_amplitude(L: np.ndarray) -> np.ndarray:
    """Peak |L| across (channels × moments) for each source — see
    :func:`inob.analysis.snr.per_source_peak`."""
    return per_source_peak(L)


def per_source_rms_amplitude(L: np.ndarray) -> np.ndarray:
    """RMS |L| across (channels × moments). Conservative typical sensor."""
    C, three_S = L.shape
    S = three_S // 3
    L3 = L.reshape(C, S, 3)
    return np.sqrt(np.mean(L3 ** 2, axis=(0, 2)))


def per_source_eeg_amplitude(L: np.ndarray) -> np.ndarray:
    """The EEG observable: best bipolar pair in the array, per source.

    MEG channels measure a field directly, so the best-channel peak is the
    signal. A surface potential is not a per-channel quantity — it is defined
    only against a reference, and the saved leadfield's common-average
    reference discards most of a deep source's amplitude across a patch-sized
    footprint. What an electrode array measures is a *difference*, so that is
    what the EEG panels use. See :func:`inob.analysis.snr.per_source_best_bipolar`.
    """
    return per_source_best_bipolar(L)


#: Averages used by clinical somatosensory-evoked-potential recording
#: (Cruccu et al. 2008). Drawn on the evoked-paradigm panels as the budget a
#: real session actually has: a trials-to-detect figure landing inside this
#: band means the measurement fits an existing clinical protocol.
CLINICAL_AVERAGE_BUDGET: tuple[int, int] = (500, 2000)


@dataclass(frozen=True)
class PropagationCorrection:
    """How much the propagating source model costs, per modality.

    The detectability panels compute signal as leadfield × Q, which lumps the
    whole event into one stationary dipole. For a target whose volley sweeps
    far compared with its AP width that overestimates: contributions from
    different arc positions partially cancel. These factors — from
    :mod:`inob.analysis.propagation`, the same code behind
    ``cap_compare_<target>.png`` — scale the stationary curves down to the
    propagating case. They are measured against *this* figure's stationary
    reference (the quoted source) on *this* figure's observable, so they are
    numerically different from the best-radial-channel ratio cap_compare prints
    while describing the same phenomenon through one implementation.

    A single scalar per modality is correct here: the propagating model is one
    event traversing the whole structure, so it has no per-source
    decomposition (see that module's Scope note). Applied uniformly across the
    source axis, and the figure says so.
    """
    meg: float
    eeg: float | None
    profile_name: str


def propagation_correction(
    cfg: Config, meg_lf, eeg_lf, *, source_idx: int,
) -> PropagationCorrection | None:
    """Propagating/stationary factors, or ``None`` where lumping is defensible.

    Returns ``None`` when the target's profile has ``stationary_ok=True`` (the
    vagus: a localised cervical generator crossed in ~1 ms against a 0.5 ms AP
    width), because there the correction is ≈1 and a second curve family would
    be visual noise rather than information.
    """
    from inob.analysis.propagation import (
        compute_propagation_signals,
        is_ordered_polyline,
        peak_bipolar,
        peak_over_channels,
        propagation_ratio,
    )
    from inob.physiology.profiles import profile_for_tag

    profile = profile_for_tag(source_target_tag(cfg))
    if profile.stationary_ok:
        return None
    if not is_ordered_polyline(meg_lf.source_pos):
        # A volume-fill source set has no arc length to propagate along, so the
        # ratio would be an artefact of point ordering rather than physics.
        # Better to show the stationary curves alone than a fabricated factor.
        logger.warning(
            "propagation correction skipped: this target's sources are not an "
            "ordered path (volume fill), so arc-length propagation is "
            "undefined. Detectability curves are the stationary upper bound "
            "only. See PHYSIOLOGY-TODO in inob.physiology.scenarios.",
        )
        return None
    # The stationary lump must sit at the source the panels quote, not at the
    # polyline's rostral end. Otherwise the ratio compares a dipole under the
    # array against one 73 mm away, and for the spine patch it comes out
    # ~8x — propagation apparently *helping*, which is an artefact of the
    # mismatched reference, not physics.
    meg_factor = propagation_ratio(
        compute_propagation_signals(meg_lf, profile, stationary_idx=source_idx),
        peak_over_channels,
    )
    eeg_factor = None
    if eeg_lf is not None:
        # Measured on the bipolar observable, matching what the EEG panels
        # plot — a ratio taken on a different quantity would not compose.
        eeg_factor = propagation_ratio(
            compute_propagation_signals(eeg_lf, profile, stationary_idx=source_idx),
            peak_bipolar,
        )
    logger.info(
        "propagation correction (%s profile): MEG ×%.3f, EEG ×%s",
        profile.name, meg_factor,
        "n/a" if eeg_factor is None else f"{eeg_factor:.3f}",
    )
    return PropagationCorrection(meg_factor, eeg_factor, profile.name)


def clinical_average_budget(cfg: Config) -> tuple[int, int] | None:
    """The clinical averaging budget, for targets whose paradigm is evoked.

    Only meaningful where the activity is stimulus-locked and can be averaged
    coherently — the spinal SSEP targets. Spontaneous vagal traffic has no
    equivalent budget, so it gets ``None`` and no band is drawn.
    """
    return (CLINICAL_AVERAGE_BUDGET
            if source_target_tag(cfg).startswith("spine") else None)


# ── core detectability math ────────────────────────────────────────────────

def required_trials(
    signal_per_trial: float, sigma_per_trial: float, snr_target: float = 3.0,
) -> float:
    """How many averaged trials to reach the SNR target.

    Scales as ``(snr_target × sigma / signal)²`` for white, uncorrelated noise,
    floored at 1: you always need at least one measurement. When a single trial
    already clears the target (signal/σ ≥ snr_target) the raw formula returns a
    fractional trial count, which is unphysical — you cannot average a fraction
    of a trial — so the source is reported as "detectable in one trial" (N = 1)
    rather than, e.g., 0.13 trials.
    """
    if signal_per_trial <= 0 or sigma_per_trial <= 0:
        return float("inf")
    return max(1.0, float((snr_target * sigma_per_trial / signal_per_trial) ** 2))


def snr_after_n_trials(
    signal_per_trial: float, sigma_per_trial: float, n_trials: float,
) -> float:
    """SNR after averaging ``n_trials`` repetitions (white noise)."""
    if sigma_per_trial <= 0:
        return float("inf")
    return float(signal_per_trial / sigma_per_trial * np.sqrt(max(n_trials, 1)))


# ── render ─────────────────────────────────────────────────────────────────


def _optional_leadfield(path):
    """Load a leadfield if present, else None — lets MEG-only regions (e.g. the
    spine, which has no EEG run) skip the EEG panels/rows gracefully."""
    return load_leadfield(path) if path.exists() else None


def _compatible_eeg_leadfield(meg_lf, eeg_lf):
    """Drop the EEG leadfield if it was solved against a different source set.

    The MEG and EEG leadfields for one target are supposed to share a source
    list (same ``forward.source_spacing_mm`` / anisotropy run), so every
    per-source panel can index both with the same ``source_idx``. When a
    target has been re-solved on one modality but not the other — e.g. muscle
    after a source-spacing or anisotropy change was re-run for MEG but the EEG
    array job is still queued — the two ``.npz`` files disagree on source
    count and every per-source EEG panel raises a matplotlib shape error deep
    in rendering instead of failing informatively.

    Treat a mismatch as "no EEG yet" (same as the file being absent) rather
    than crashing, and say why in the log so it is not mistaken for the EEG
    solve having failed outright.
    """
    if eeg_lf is None:
        return None
    if eeg_lf.source_pos.shape[0] != meg_lf.source_pos.shape[0]:
        logger.warning(
            "EEG leadfield has %d sources but MEG has %d — solved against a "
            "different source set (stale cache or a re-run that only "
            "regenerated one modality). Rendering MEG-only until a matching "
            "EEG solve is available.",
            eeg_lf.source_pos.shape[0], meg_lf.source_pos.shape[0],
        )
        return None
    return eeg_lf


def default_source_idx(meg_lf, eeg_lf) -> int:
    """Which source the single-source panels (a, c, d, f) should report.

    The electrode patch is deliberately sited over the target — a spine run
    centres it on C7 (see :data:`inob.config.SOURCE_TARGETS`) — so the source
    the figure should quote is the one it was sited over, not an arbitrary
    index into the source list.

    This used to take the midpoint of the source array. On an elongated target
    that is simply the wrong source: the spinal source list spans the whole
    450 mm cord, so its midpoint sits mid-thoracic, 146 mm from a cervical
    patch instead of 57 mm, and the quoted trials-to-detect came out ~700x
    pessimistic. Panels b and e always showed the full z-dependence, so the
    figure was internally inconsistent rather than uniformly wrong.

    Falls back to the midpoint when there is no EEG array to anchor to.
    """
    if eeg_lf is None:
        return meg_lf.source_pos.shape[0] // 2
    centre = np.asarray(eeg_lf.coil_pos).mean(axis=0)
    return int(np.argmin(np.linalg.norm(meg_lf.source_pos - centre, axis=1)))


def render_detectability(
    cfg: Config, *, source_idx: int = -1, out_path: Path | None = None,
    dpi: int = 300,
    scenarios: tuple[DetectabilityScenario, ...] | None = None,
    snr_threshold: float = 3.0, max_trials: int = 1_000_000,
) -> Path:
    """Six-panel detectability figure (3 MEG + 3 EEG).

      a  MEG: SNR vs N trials curves for each Q scenario.
      b  MEG: trials needed for SNR=3 detection across the cervical sources.
      c  MEG: required recording time at typical CAP rates.
      d/e/f  same panels for EEG.

    ``scenarios`` defaults to the target-appropriate source-strength set
    (muscle → magnetomyography Q range; otherwise the vagal-CAP range).
    """
    if scenarios is None:
        scenarios = scenarios_for_target(cfg)
    apply_nature_style()
    floors = compute_noise_floors(cfg)
    sigma_meg = floors.meg_per_channel_fT          # fT per trial, broadband
    sigma_eeg = floors.eeg_per_channel_uV          # µV per trial

    region = source_region_label(cfg)
    meg_lf = load_leadfield(cfg.outputs.forward_npz)
    eeg_lf = _compatible_eeg_leadfield(
        meg_lf, _optional_leadfield(cfg.outputs.forward_eeg_npz))
    has_eeg = eeg_lf is not None
    if source_idx < 0:
        source_idx = default_source_idx(meg_lf, eeg_lf)
    src = meg_lf.source_pos[source_idx]

    # One signal number per Z position. MEG: best-channel peak. EEG: best
    # bipolar pair, since a potential is only measurable as a difference.
    meg_peak = per_source_peak_amplitude(meg_lf.L_fT_per_nAm)   # fT/nAm
    eeg_peak = (per_source_eeg_amplitude(eeg_lf.L_fT_per_nAm)   # µV/nAm
                if has_eeg else None)
    z = meg_lf.source_pos[:, 2]
    # A volume-fill source set (muscle) has no meaningful order along its
    # source list — connecting points by list order in a line plot draws
    # zigzags between sources that share a z but sit in different muscles, or
    # different sides of the body. The per-source panels switch to a scatter
    # for these targets; ordered-polyline targets (vagus, spine) are
    # unaffected and keep the line plot they have always had.
    ordered_sources = is_ordered_polyline(meg_lf.source_pos)
    budget = clinical_average_budget(cfg)
    # Where the stationary lump is not defensible, every panel carries a second
    # dashed family showing what propagation costs. Solid is then an upper
    # bound, not the expected answer.
    prop = propagation_correction(cfg, meg_lf, eeg_lf, source_idx=source_idx)

    caption = (
        "MEG: best-channel peak |L| per source. EEG: best bipolar pair, which "
        "is reference-independent and is what an electrode array measures "
        "(after Hämäläinen et al. 1993; "
        "OPM noise floor from QuSpin Gen-3 spec, "
        "Malliaras-group PEDOT:PSS textile-electrode noise = amplifier + Johnson "
        f"(R = {cfg.noise.eeg_electrode_skin_kohm:g} kΩ), integrated over the "
        f"{cfg.noise.band_label} recording band). "
        + ("Solid curves lump the event into one stationary dipole at the "
           "quoted source — an upper bound. Dashed curves apply the "
           "propagating-source model (inob.analysis.propagation, the same code "
           "behind cap_compare), measured per modality on that panel's own "
           "observable and against its own stationary reference, so the "
           f"factors ({prop.meg:.2f} MEG"
           + (f", {prop.eeg:.2f} EEG" if prop.eeg is not None else "")
           + ") differ from the best-radial-channel ratio cap_compare prints. "
           "One event sweeps the whole structure, so the correction is a "
           "scalar with no per-source form. " if prop is not None else "")
        + "Trial counts assume independent white noise across averages — "
        "spatially / temporally correlated environmental MEG noise (heartbeat "
        "artefacts, magnetic shielding residual; cf. Boto et al. 2018) inflates "
        "the required N by a factor of 1–10× depending on shielding quality."
    )
    # Wrap explicitly: matplotlib does not wrap fig.text, and save_figure uses
    # bbox_inches="tight", so one very long line silently stretches the saved
    # PNG to that line's width — the figure came out 4:1 instead of its figsize.
    # The wrapped height then sets the bottom margin, so a caption that grows
    # (e.g. when the propagation correction adds its paragraph) reserves the
    # room it needs instead of overlapping the bottom row of panels.
    caption = textwrap.fill(caption, width=170)
    caption_lines = caption.count("\n") + 1

    # ── figure (2 rows MEG+EEG, or 1 row MEG-only) ─────────────────────────
    fig_h = 11.5 if has_eeg else 6.2
    fig = plt.figure(figsize=(15, fig_h))
    # The suptitle, each panel's own title, and the "b"/"e" panel labels all
    # need physical (inch) space above the axes, not a fixed figure-fraction —
    # a fraction sized for the two-row (fig_h=11.5) case leaves only ~0.4 in
    # for the one-row MEG-only case (e.g. muscle, which has no EEG solve yet),
    # so the panel label and suptitle text overlap. Reserve a constant ~0.85 in
    # regardless of row count.
    top = 1.0 - 0.85 / fig_h
    gs = GridSpec(2 if has_eeg else 1, 3, figure=fig,
                  left=0.06, right=0.97, top=top,
                  bottom=0.055 + 0.115 * caption_lines / fig_h,
                  hspace=0.36, wspace=0.30)

    n_grid = np.logspace(0, np.log10(max_trials), 200)

    scenario_colors = [
        NATURE_PALETTE["blue"], NATURE_PALETTE["teal"],
        NATURE_PALETTE["orange"], NATURE_PALETTE["red"],
    ]

    def _model_proxies(ax, factor):
        """Legend entries for the two source models (linestyle, not colour).

        Colour already encodes Q; duplicating every scenario for both models
        would double an already busy legend, so the models are shown once as
        grey proxy handles.
        """
        if factor is None:
            return []
        return [
            Line2D([], [], color=NATURE_PALETTE["axis"], lw=1.4,
                   label="stationary (upper bound)"),
            Line2D([], [], color=NATURE_PALETTE["axis"], lw=1.4, ls=(0, (4, 2)),
                   label=f"propagating (×{factor:.2f})"),
        ]

    def _plot_snr_curves(ax, peak_one_source: float, sigma: float, ylabel: str,
                         factor: float | None = None):
        if budget is not None:
            ax.axvspan(*budget, color=NATURE_PALETTE["axis"], alpha=0.10, lw=0,
                       label=f"clinical SSEP averages ({budget[0]}–{budget[1]})")
        for sc, col in zip(scenarios, scenario_colors, strict=False):
            sig = peak_one_source * sc.Q_nAm
            ax.plot(n_grid, sig / sigma * np.sqrt(n_grid),
                    color=col, lw=1.6, label=sc.label)
            if factor is not None:
                ax.plot(n_grid, sig * factor / sigma * np.sqrt(n_grid),
                        color=col, lw=1.3, ls=(0, (4, 2)), alpha=0.9)
        ax.axhline(snr_threshold, color=NATURE_PALETTE["axis"], lw=0.8,
                   linestyle="--", label=f"SNR = {snr_threshold:g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of averaged trials")
        ax.set_ylabel(ylabel)
        handles, labels = ax.get_legend_handles_labels()
        proxies = _model_proxies(ax, factor)
        ax.legend(handles + proxies, labels + [h.get_label() for h in proxies],
                  loc="lower right", fontsize=7, handlelength=1.6)

    def _line_or_scatter(ax, y, *, color, lw, label=None, ls="-", alpha=1.0):
        """Line for an ordered polyline source set, scatter otherwise.

        See the ``ordered_sources`` note above ``z`` — a list-order line plot
        over muscle's volume-fill sources connects points that are not spatial
        neighbours, drawing zigzags with no physical meaning.
        """
        if ordered_sources:
            ax.plot(z, y, color=color, lw=lw, label=label, ls=ls, alpha=alpha)
        else:
            ax.scatter(z, y, color=color, s=6, label=label, alpha=max(alpha, 0.35),
                       edgecolors="none")

    def _plot_trials_per_source(ax, peak_arr: np.ndarray, sigma: float,
                                  *, ymax: float | None = None,
                                  factor: float | None = None):
        all_n = []
        if budget is not None:
            ax.axhspan(*budget, color=NATURE_PALETTE["axis"], alpha=0.10, lw=0,
                       label=f"clinical SSEP averages ({budget[0]}–{budget[1]})")

        def trials(sig):
            # Floor at 1 trial: a source already above threshold in a single
            # trial needs N = 1, not the fractional N the raw formula gives.
            return np.maximum(
                1.0, (snr_threshold * sigma / np.maximum(sig, 1e-30)) ** 2)

        for sc, col in zip(scenarios, scenario_colors, strict=False):
            sig = peak_arr * sc.Q_nAm
            n = trials(sig)
            all_n.append(n)
            _line_or_scatter(ax, n, color=col, lw=1.4, label=sc.label)
            if factor is not None:
                # One scalar applied across the whole source axis: the
                # propagating model is a single event sweeping the structure,
                # so it has no per-source form to plot.
                n_prop = trials(sig * factor)
                all_n.append(n_prop)
                _line_or_scatter(ax, n_prop, color=col, lw=1.2, ls=(0, (4, 2)),
                                 alpha=0.9)
        ax.set_xlabel("Source z (mm)")
        ax.set_ylabel(f"Trials needed for SNR ≥ {snr_threshold:g}")
        ax.set_yscale("log")
        # Fit the y-range to the actual curves (with a decade of padding) so
        # every scenario is visible, from the N = 1 floor up to ≫ max_trials.
        stacked = np.concatenate(all_n) if all_n else np.array([1.0, max_trials])
        finite = stacked[np.isfinite(stacked) & (stacked > 0)]
        data_lo = float(np.min(finite)) if finite.size else 1.0
        data_hi = float(np.max(finite)) if finite.size else max_trials
        ymin = 10 ** np.floor(np.log10(data_lo * 0.5))
        if ymax is None:
            ymax = 10 ** np.ceil(np.log10(data_hi * 2.0))
        ax.set_ylim(ymin, ymax)
        # Only annotate the max-trials cap if it actually falls in the visible
        # range — for very strong sources (e.g. muscle) required trials never
        # approach it, and an off-axis text position with clip_on=False (the
        # default) blows up the bbox_inches="tight" save to include it.
        if ymin <= max_trials <= ymax:
            ax.axhline(max_trials, color=NATURE_PALETTE["axis"], lw=0.5,
                       linestyle=":", alpha=0.6)
            ax.text(z.min(), max_trials * 1.4, f"{max_trials:.0e} trials",
                    fontsize=6.5, color=NATURE_PALETTE["axis"], alpha=0.8)

    def _floors_everywhere(peak_arr: np.ndarray, sigma: float) -> bool:
        """True when every scenario clears threshold in one trial at every z.

        ``_plot_trials_per_source`` floors at N = 1 (§ module docstring), so
        for a source this strong the whole panel is a flat line at 1 — no
        detection boundary crosses the plotted range, so the panel carries no
        information about *where* along the structure detection succeeds or
        fails (see :func:`_plot_snr_margin_per_source`, used instead for that
        case). Checked per modality/scenario-set combination, since a source
        this strong for MEG need not be for EEG.
        """
        if not scenarios:
            return False
        weakest = min(scenarios, key=lambda sc: sc.Q_nAm)
        snr1 = peak_arr * weakest.Q_nAm / sigma
        return bool(np.all(snr1 >= snr_threshold))

    def _plot_snr_margin_per_source(ax, peak_arr: np.ndarray, sigma: float,
                                     *, factor: float | None = None):
        """Single-trial SNR along the source axis (dB above threshold).

        Used instead of :func:`_plot_trials_per_source` when every scenario
        already clears the detection threshold in a single trial everywhere
        along the structure (muscle's MMG signal does, by 1–3 orders of
        magnitude) — trials-to-detect is then a flat line at N = 1 with no
        information in it. The informative question for a source this strong
        is not "how many trials" but "how much headroom", so this panel plots
        single-trial SNR in dB, which keeps varying with position even when
        every curve sits far above the N = 1 floor.
        """
        for sc, col in zip(scenarios, scenario_colors, strict=False):
            snr = np.maximum(peak_arr * sc.Q_nAm / sigma, 1e-30)
            _line_or_scatter(ax, 20 * np.log10(snr), color=col, lw=1.4,
                             label=sc.label)
            if factor is not None:
                snr_prop = np.maximum(peak_arr * sc.Q_nAm * factor / sigma, 1e-30)
                _line_or_scatter(ax, 20 * np.log10(snr_prop), color=col, lw=1.2,
                                 ls=(0, (4, 2)), alpha=0.9)
        ax.axhline(20 * np.log10(snr_threshold), color=NATURE_PALETTE["axis"],
                   lw=0.8, linestyle="--", label=f"SNR = {snr_threshold:g}")
        ax.set_xlabel("Source z (mm)")
        ax.set_ylabel("Single-trial SNR (dB above 1)")

    # Event-repetition rates for the averaging axis, and their label. Muscle
    # events recur at motor-unit firing rates (~8–30 Hz); vagal CAPs at the
    # cardiac/reflex rates. This is only the trial-repetition rate used for
    # √N averaging — no firing dynamics are modelled (see PHYSIOLOGY-TODO).
    is_muscle = source_target_tag(cfg) == "muscle"
    rate_label = "MU firing rate" if is_muscle else "CAP rate"
    rec_rates = (8.0, 15.0, 30.0) if is_muscle else (1.0, 5.0, 20.0)

    def _plot_recording_time(ax, peak_one_source: float, sigma: float, rates_hz,
                             factor: float | None = None):
        def seconds_for(rate, scale):
            out = []
            for sc in scenarios:
                sig = peak_one_source * sc.Q_nAm * scale
                if sig <= 0:
                    out.append(np.inf)
                    continue
                # Floor at 1 trial → minimum recording is one trial period.
                out.append(max(1.0, (snr_threshold * sigma / sig) ** 2) / rate)
            return out

        Qs = [sc.Q_nAm for sc in scenarios]
        for rate, col in zip(rates_hz, scenario_colors[:len(rates_hz)], strict=False):
            ax.plot(Qs, seconds_for(rate, 1.0), marker="o", lw=1.4, color=col,
                    label=f"{rate:g} Hz {rate_label}")
            if factor is not None:
                ax.plot(Qs, seconds_for(rate, factor), marker="o", ms=3, lw=1.2,
                        ls=(0, (4, 2)), color=col, alpha=0.9)
        ax.set_xlabel("Source dipole moment Q (nA·m)")
        ax.set_ylabel(f"Recording duration for SNR ≥ {snr_threshold:g} (s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        handles, labels = ax.get_legend_handles_labels()
        proxies = _model_proxies(ax, factor)
        ax.legend(handles + proxies, labels + [h.get_label() for h in proxies],
                  loc="upper right", fontsize=7, handlelength=1.6)

    # ── MEG row ────────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_snr_curves(ax_a, meg_peak[source_idx], sigma_meg,
                     "MEG SNR (best-channel)",
                     factor=None if prop is None else prop.meg)
    ax_a.set_title(f"MEG  ·  SNR vs N trials  ·  noise σ = {sigma_meg:.0f} fT")
    add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    meg_factor = None if prop is None else prop.meg
    if _floors_everywhere(meg_peak, sigma_meg):
        _plot_snr_margin_per_source(ax_b, meg_peak, sigma_meg, factor=meg_factor)
        ax_b.set_title(f"MEG  ·  single-trial SNR along the {region} "
                       "(detects in N=1 everywhere)")
    else:
        _plot_trials_per_source(ax_b, meg_peak, sigma_meg, factor=meg_factor)
        ax_b.set_title(f"MEG  ·  trials-to-detect along the {region}")
    ax_b.legend(loc="upper right", fontsize=6.5, handlelength=1.4)
    add_panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[0, 2])
    _plot_recording_time(ax_c, meg_peak[source_idx], sigma_meg,
                         rates_hz=rec_rates,
                         factor=None if prop is None else prop.meg)
    ax_c.set_title(f"MEG  ·  recording time @ source z = {src[2]:.0f} mm")
    add_panel_label(ax_c, "c")

    # ── EEG row (only when an EEG leadfield was computed) ──────────────────
    if has_eeg:
        ax_d = fig.add_subplot(gs[1, 0])
        _plot_snr_curves(ax_d, eeg_peak[source_idx], sigma_eeg,
                         "EEG SNR (best bipolar pair)",
                         factor=None if prop is None else prop.eeg)
        ax_d.set_title(f"EEG  ·  SNR vs N trials  ·  noise σ = {sigma_eeg:.1f} µV")
        add_panel_label(ax_d, "d")

        ax_e = fig.add_subplot(gs[1, 1])
        eeg_factor = None if prop is None else prop.eeg
        if _floors_everywhere(eeg_peak, sigma_eeg):
            _plot_snr_margin_per_source(ax_e, eeg_peak, sigma_eeg, factor=eeg_factor)
            ax_e.set_title(f"EEG  ·  single-trial SNR along the {region} "
                           "(detects in N=1 everywhere)")
        else:
            _plot_trials_per_source(ax_e, eeg_peak, sigma_eeg, factor=eeg_factor)
            ax_e.set_title(f"EEG  ·  trials-to-detect along the {region}")
        ax_e.legend(loc="upper right", fontsize=6.5, handlelength=1.4)
        add_panel_label(ax_e, "e")

        ax_f = fig.add_subplot(gs[1, 2])
        _plot_recording_time(ax_f, eeg_peak[source_idx], sigma_eeg,
                             rates_hz=rec_rates,
                             factor=None if prop is None else prop.eeg)
        ax_f.set_title(f"EEG  ·  recording time @ source z = {src[2]:.0f} mm")
        add_panel_label(ax_f, "f")

    fig.suptitle(
        f"Detectability — N trials × noise floor × source strength  "
        f"(SNR threshold = {snr_threshold:g}, band = {cfg.noise.band_label})",
        fontsize=12, fontweight="bold", y=0.985,
    )
    # Wrap explicitly. Matplotlib does not wrap fig.text, and save_figure uses
    # bbox_inches="tight", so an unwrapped caption silently stretches the saved
    # PNG to the width of one very long line — the figure came out 4:1 instead
    # of its 15:11.5 figsize.
    fig.text(
        0.5, 0.008, caption,
        ha="center", va="bottom", fontsize=7,
        color=NATURE_PALETTE["axis"], style="italic",
    )

    return save_figure(fig, out_path or target_output(cfg, "detectability.png"), dpi=dpi)


# ── headline numbers (printed in the CLI) ──────────────────────────────────

def detectability_summary(
    cfg: Config, *, source_idx: int = -1,
    scenarios: tuple[DetectabilityScenario, ...] | None = None,
) -> dict:
    """Compute headline detectability numbers as a JSON-serialisable dict.

    ``scenarios`` defaults to the target-appropriate ladder, exactly as
    :func:`render_detectability` does — pass the same tuple to both so the
    figure and the JSON never describe different source strengths.
    """
    floors = compute_noise_floors(cfg)
    sigma_meg = floors.meg_per_channel_fT
    sigma_eeg = floors.eeg_per_channel_uV

    meg_lf = load_leadfield(cfg.outputs.forward_npz)
    eeg_lf = _compatible_eeg_leadfield(
        meg_lf, _optional_leadfield(cfg.outputs.forward_eeg_npz))
    if source_idx < 0:
        source_idx = default_source_idx(meg_lf, eeg_lf)

    meg_peak = float(per_source_peak_amplitude(meg_lf.L_fT_per_nAm)[source_idx])
    eeg_peak = (float(per_source_eeg_amplitude(eeg_lf.L_fT_per_nAm)[source_idx])
                if eeg_lf is not None else None)
    src = meg_lf.source_pos[source_idx]
    budget = clinical_average_budget(cfg)
    prop = propagation_correction(cfg, meg_lf, eeg_lf, source_idx=source_idx)

    rows = {}
    for sc in (scenarios if scenarios is not None else scenarios_for_target(cfg)):
        sig_meg = meg_peak * sc.Q_nAm
        row = {
            "Q_nAm": sc.Q_nAm,
            "MEG_per_trial_fT": sig_meg,
            "MEG_single_trial_SNR": sig_meg / sigma_meg if sigma_meg else float("inf"),
            "MEG_trials_for_SNR3": required_trials(sig_meg, sigma_meg, 3.0),
        }
        if prop is not None:
            row["MEG_per_trial_fT_propagating"] = sig_meg * prop.meg
            row["MEG_trials_for_SNR3_propagating"] = required_trials(
                sig_meg * prop.meg, sigma_meg, 3.0)
        if eeg_peak is not None:
            sig_eeg = eeg_peak * sc.Q_nAm
            n_eeg = required_trials(sig_eeg, sigma_eeg, 3.0)
            row.update({
                "EEG_per_trial_uV": sig_eeg,
                "EEG_single_trial_SNR": sig_eeg / sigma_eeg if sigma_eeg else float("inf"),
                "EEG_trials_for_SNR3": n_eeg,
            })
            if prop is not None and prop.eeg is not None:
                # The propagating source model is the honest case wherever the
                # profile says lumping is invalid; report both so the headline
                # number is a choice made in the open, not by default.
                n_eeg_prop = required_trials(sig_eeg * prop.eeg, sigma_eeg, 3.0)
                row["EEG_per_trial_uV_propagating"] = sig_eeg * prop.eeg
                row["EEG_trials_for_SNR3_propagating"] = n_eeg_prop
            if budget is not None:
                # Does the answer fit the averaging budget a clinical SSEP
                # session already spends? That, not the raw µV, is the
                # feasibility question for an evoked paradigm.
                row["EEG_within_clinical_budget"] = bool(n_eeg <= budget[1])
                if prop is not None and prop.eeg is not None:
                    row["EEG_within_clinical_budget_propagating"] = bool(
                        n_eeg_prop <= budget[1])
        rows[sc.label] = row
    return {
        "source_idx": int(source_idx),
        "source_z_mm": float(src[2]),
        "noise_meg_fT": float(sigma_meg),
        "noise_eeg_uV": float(sigma_eeg),
        "bandwidth_hz": float(cfg.noise.effective_bandwidth_hz),
        "band_label": cfg.noise.band_label,
        "clinical_average_budget": list(budget) if budget else None,
        "propagation_factor_meg": None if prop is None else prop.meg,
        "propagation_factor_eeg": None if prop is None else prop.eeg,
        "scenarios": rows,
    }
