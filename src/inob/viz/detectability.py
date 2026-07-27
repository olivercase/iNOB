"""Detectability analysis — given N trials and a noise floor, can we see it?

Predicted signal-to-noise ratio for the cervical-vagus CAP, plotted as a
function of:

  * source dipole moment  Q  (from a single low-fibre-count event up to the
    full A+C summation at ~70 nA·m, with fibre-population assumptions
    discussed in Hämäläinen et al. 1993 and Bu et al. 2024);
  * number of averaged trials  N  (white-noise SNR scales as √N);
  * per-modality noise floor (OPM intrinsic + EEG amplifier + Johnson).

Detection threshold convention: SNR ≥ 3 (Rose criterion / standard
post-averaging visibility). Other thresholds are easy to read off.

Trial-count floor: the required-trials figure is ``(SNR_target·σ/signal)²``,
floored at 1. Strong sources (e.g. a 20–70 nA·m CAP on the best channel) can
clear the threshold in a single trial, where the raw formula returns a
fractional trial count that has no physical meaning — you cannot average a
fraction of a trial. Such sources are reported as "detectable in one trial"
(N = 1), and the recording-time panel bottoms out at one trial period (1/rate).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from inob.analysis.snr import compute_noise_floors, per_source_peak
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


# Default scenarios spanning the realistic vagus-CAP range.
DEFAULT_SCENARIOS: tuple[DetectabilityScenario, ...] = (
    DetectabilityScenario("Q = 1 nA·m  (calibration unit)", 1.0,
                           "Reference scale, leadfield calibration."),
    DetectabilityScenario("Q = 5 nA·m  (sparse activation)", 5.0,
                           "~7% of full summation; spontaneous baseline."),
    DetectabilityScenario("Q = 20 nA·m  (modest CAP)", 20.0,
                           "Reflex / mild evoked activation."),
    DetectabilityScenario("Q = 70 nA·m  (full A+C summation)", 70.0,
                           "Per Bu et al. 2024 Hämäläinen summation."),
)

# Muscle (magnetomyography) source-strength range. Muscle fibres are ~50 µm —
# an order of magnitude fatter than the ~8 µm vagal A-fibres — so the per-fibre
# Hämäläinen moment (Q ∝ d²) is ~40× larger, and a motor unit fires 100s of
# fibres near-synchronously. Static equivalent-current-dipole magnitudes only —
# recruitment/firing dynamics are deliberately NOT modelled here; see the
# PHYSIOLOGY-TODO block in :mod:`inob.physiology.scenarios` for the deferred
# muscle dynamics work:
#   per-fibre Q ≈ π·(50 µm)²·σ_in(0.4)·ΔV(0.1 V)/4 ≈ 0.08 nA·m
#   single MUAP  ≈ 10²–10³ fibres      → ~10 nA·m
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


def scenarios_for_target(cfg: Config) -> tuple[DetectabilityScenario, ...]:
    """Pick the source-strength scenario set matching the forward target.

    Muscle uses the magnetomyography Q range (:data:`MUSCLE_SCENARIOS`); every
    other target keeps the vagal-CAP range (:data:`DEFAULT_SCENARIOS`)."""
    return MUSCLE_SCENARIOS if source_target_tag(cfg) == "muscle" else DEFAULT_SCENARIOS


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
    eeg_lf = _optional_leadfield(cfg.outputs.forward_eeg_npz)
    has_eeg = eeg_lf is not None
    if source_idx < 0:
        source_idx = meg_lf.source_pos.shape[0] // 2
    src = meg_lf.source_pos[source_idx]

    # Best-channel peak per source (one number per Z position)
    meg_peak = per_source_peak_amplitude(meg_lf.L_fT_per_nAm)   # fT/nAm
    eeg_peak = (per_source_peak_amplitude(eeg_lf.L_fT_per_nAm)  # µV/nAm
                if has_eeg else None)
    z = meg_lf.source_pos[:, 2]

    # ── figure (2 rows MEG+EEG, or 1 row MEG-only) ─────────────────────────
    fig = plt.figure(figsize=(15, 11.5 if has_eeg else 6.2))
    gs = GridSpec(2 if has_eeg else 1, 3, figure=fig,
                  left=0.06, right=0.97, top=0.93, bottom=0.07,
                  hspace=0.36, wspace=0.30)

    n_grid = np.logspace(0, np.log10(max_trials), 200)

    scenario_colors = [
        NATURE_PALETTE["blue"], NATURE_PALETTE["teal"],
        NATURE_PALETTE["orange"], NATURE_PALETTE["red"],
    ]

    def _plot_snr_curves(ax, peak_one_source: float, sigma: float, ylabel: str):
        for sc, col in zip(scenarios, scenario_colors, strict=False):
            sig = peak_one_source * sc.Q_nAm
            snrs = sig / sigma * np.sqrt(n_grid)
            ax.plot(n_grid, snrs, color=col, lw=1.6, label=sc.label)
        ax.axhline(snr_threshold, color=NATURE_PALETTE["axis"], lw=0.8,
                   linestyle="--", label=f"SNR = {snr_threshold:g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of averaged trials")
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower right", fontsize=7, handlelength=1.4)

    def _plot_trials_per_source(ax, peak_arr: np.ndarray, sigma: float,
                                  *, ymax: float | None = None):
        all_n = []
        for sc, col in zip(scenarios, scenario_colors, strict=False):
            sig = peak_arr * sc.Q_nAm
            # Floor at 1 trial: a source already above threshold in a single
            # trial needs N = 1, not the fractional N the raw formula gives.
            n = np.maximum(1.0, (snr_threshold * sigma / np.maximum(sig, 1e-30)) ** 2)
            all_n.append(n)
            ax.plot(z, n, color=col, lw=1.4, label=sc.label)
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
        ax.axhline(max_trials, color=NATURE_PALETTE["axis"], lw=0.5,
                   linestyle=":", alpha=0.6)
        ax.text(z.min(), max_trials * 1.4, f"{max_trials:.0e} trials",
                fontsize=6.5, color=NATURE_PALETTE["axis"], alpha=0.8)

    # Event-repetition rates for the averaging axis, and their label. Muscle
    # events recur at motor-unit firing rates (~8–30 Hz); vagal CAPs at the
    # cardiac/reflex rates. This is only the trial-repetition rate used for
    # √N averaging — no firing dynamics are modelled (see PHYSIOLOGY-TODO).
    is_muscle = source_target_tag(cfg) == "muscle"
    rate_label = "MU firing rate" if is_muscle else "CAP rate"
    rec_rates = (8.0, 15.0, 30.0) if is_muscle else (1.0, 5.0, 20.0)

    def _plot_recording_time(ax, peak_one_source: float, sigma: float, rates_hz):
        for rate, col in zip(rates_hz, scenario_colors[:len(rates_hz)], strict=False):
            seconds = []
            for sc in scenarios:
                sig = peak_one_source * sc.Q_nAm
                if sig <= 0:
                    seconds.append(np.inf)
                    continue
                # Floor at 1 trial → minimum recording is one trial period.
                n = max(1.0, (snr_threshold * sigma / sig) ** 2)
                seconds.append(n / rate)
            ax.plot([sc.Q_nAm for sc in scenarios], seconds,
                    marker="o", lw=1.4, color=col, label=f"{rate:g} Hz {rate_label}")
        ax.set_xlabel("Source dipole moment Q (nA·m)")
        ax.set_ylabel(f"Recording duration for SNR ≥ {snr_threshold:g} (s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=7, handlelength=1.4)

    # ── MEG row ────────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_snr_curves(ax_a, meg_peak[source_idx], sigma_meg,
                     "MEG SNR (best-channel)")
    ax_a.set_title(f"MEG  ·  SNR vs N trials  ·  noise σ = {sigma_meg:.0f} fT")
    add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    _plot_trials_per_source(ax_b, meg_peak, sigma_meg)
    ax_b.set_title(f"MEG  ·  trials-to-detect along the {region}")
    ax_b.legend(loc="upper right", fontsize=6.5, handlelength=1.4)
    add_panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[0, 2])
    _plot_recording_time(ax_c, meg_peak[source_idx], sigma_meg,
                         rates_hz=rec_rates)
    ax_c.set_title(f"MEG  ·  recording time @ source z = {src[2]:.0f} mm")
    add_panel_label(ax_c, "c")

    # ── EEG row (only when an EEG leadfield was computed) ──────────────────
    if has_eeg:
        ax_d = fig.add_subplot(gs[1, 0])
        _plot_snr_curves(ax_d, eeg_peak[source_idx], sigma_eeg,
                         "EEG SNR (best-contact)")
        ax_d.set_title(f"EEG  ·  SNR vs N trials  ·  noise σ = {sigma_eeg:.1f} µV")
        add_panel_label(ax_d, "d")

        ax_e = fig.add_subplot(gs[1, 1])
        _plot_trials_per_source(ax_e, eeg_peak, sigma_eeg)
        ax_e.set_title(f"EEG  ·  trials-to-detect along the {region}")
        ax_e.legend(loc="upper right", fontsize=6.5, handlelength=1.4)
        add_panel_label(ax_e, "e")

        ax_f = fig.add_subplot(gs[1, 2])
        _plot_recording_time(ax_f, eeg_peak[source_idx], sigma_eeg,
                             rates_hz=rec_rates)
        ax_f.set_title(f"EEG  ·  recording time @ source z = {src[2]:.0f} mm")
        add_panel_label(ax_f, "f")

    fig.suptitle(
        f"Detectability — N trials × noise floor × source strength  "
        f"(SNR threshold = {snr_threshold:g}, BW = {cfg.noise.bandwidth_hz:g} Hz)",
        fontsize=12, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.005,
        "Best-channel peak |L| per source (after Hämäläinen et al. 1993; "
        "OPM noise floor from QuSpin Gen-3 spec, "
        "Malliaras-group PEDOT:PSS textile-electrode noise = amplifier + Johnson "
        f"(R = {cfg.noise.eeg_electrode_skin_kohm:g} kΩ). "
        "Trial counts assume independent white noise across averages — "
        "spatially / temporally correlated environmental MEG noise (heartbeat "
        "artefacts, magnetic shielding residual; cf. Boto et al. 2018) inflates "
        "the required N by a factor of 1–10× depending on shielding quality.",
        ha="center", va="bottom", fontsize=7,
        color=NATURE_PALETTE["axis"], style="italic",
    )

    return save_figure(fig, out_path or target_output(cfg, "detectability.png"), dpi=dpi)


# ── headline numbers (printed in the CLI) ──────────────────────────────────

def detectability_summary(cfg: Config, *, source_idx: int = -1) -> dict:
    """Compute headline detectability numbers as a JSON-serialisable dict."""
    floors = compute_noise_floors(cfg)
    sigma_meg = floors.meg_per_channel_fT
    sigma_eeg = floors.eeg_per_channel_uV

    meg_lf = load_leadfield(cfg.outputs.forward_npz)
    eeg_lf = _optional_leadfield(cfg.outputs.forward_eeg_npz)
    if source_idx < 0:
        source_idx = meg_lf.source_pos.shape[0] // 2

    meg_peak = float(per_source_peak_amplitude(meg_lf.L_fT_per_nAm)[source_idx])
    eeg_peak = (float(per_source_peak_amplitude(eeg_lf.L_fT_per_nAm)[source_idx])
                if eeg_lf is not None else None)
    src = meg_lf.source_pos[source_idx]

    rows = {}
    for sc in scenarios_for_target(cfg):
        sig_meg = meg_peak * sc.Q_nAm
        row = {
            "Q_nAm": sc.Q_nAm,
            "MEG_per_trial_fT": sig_meg,
            "MEG_single_trial_SNR": sig_meg / sigma_meg if sigma_meg else float("inf"),
            "MEG_trials_for_SNR3": required_trials(sig_meg, sigma_meg, 3.0),
        }
        if eeg_peak is not None:
            sig_eeg = eeg_peak * sc.Q_nAm
            row.update({
                "EEG_per_trial_uV": sig_eeg,
                "EEG_single_trial_SNR": sig_eeg / sigma_eeg if sigma_eeg else float("inf"),
                "EEG_trials_for_SNR3": required_trials(sig_eeg, sigma_eeg, 3.0),
            })
        rows[sc.label] = row
    return {
        "source_idx": int(source_idx),
        "source_z_mm": float(src[2]),
        "noise_meg_fT": float(sigma_meg),
        "noise_eeg_uV": float(sigma_eeg),
        "bandwidth_hz": float(cfg.noise.bandwidth_hz),
        "scenarios": rows,
    }
