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
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from vagus_fm.analysis.snr import compute_noise_floors
from vagus_fm.config import Config
from vagus_fm.io.npz import load_leadfield
from vagus_fm.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
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


def per_source_peak_amplitude(L: np.ndarray) -> np.ndarray:
    """Peak |L| across (channels × moments) for each source.

    Shape (C, 3*S) → (S,). Returns a per-source maximum |L| (across all
    moments and channels). This is the *best-case* sensor for that source —
    what the most-sensitive channel sees if we point the dipole optimally.
    """
    C, three_S = L.shape
    if three_S % 3 != 0:
        raise ValueError(f"L shape {L.shape}: second dim not divisible by 3")
    S = three_S // 3
    L3 = L.reshape(C, S, 3)
    return np.abs(L3).reshape(C, S, 3).max(axis=(0, 2))


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

    Scales as ``(snr_target × sigma / signal)²`` for white, uncorrelated noise.
    """
    if signal_per_trial <= 0 or sigma_per_trial <= 0:
        return float("inf")
    return float((snr_target * sigma_per_trial / signal_per_trial) ** 2)


def snr_after_n_trials(
    signal_per_trial: float, sigma_per_trial: float, n_trials: float,
) -> float:
    """SNR after averaging ``n_trials`` repetitions (white noise)."""
    if sigma_per_trial <= 0:
        return float("inf")
    return float(signal_per_trial / sigma_per_trial * np.sqrt(max(n_trials, 1)))


# ── render ─────────────────────────────────────────────────────────────────

def render_detectability(
    cfg: Config, *, source_idx: int = -1, out_path: Path | None = None,
    dpi: int = 300,
    scenarios: tuple[DetectabilityScenario, ...] = DEFAULT_SCENARIOS,
    snr_threshold: float = 3.0, max_trials: int = 1_000_000,
) -> Path:
    """Six-panel detectability figure (3 MEG + 3 EEG).

      a  MEG: SNR vs N trials curves for each Q scenario.
      b  MEG: trials needed for SNR=3 detection across the cervical sources.
      c  MEG: required recording time at typical CAP rates.
      d/e/f  same panels for EEG.
    """
    apply_nature_style()
    floors = compute_noise_floors(cfg)
    sigma_meg = floors.meg_per_channel_fT          # fT per trial, broadband
    sigma_eeg = floors.eeg_per_channel_uV          # µV per trial

    meg_lf = load_leadfield(cfg.outputs.forward_npz)
    eeg_lf = load_leadfield(cfg.outputs.forward_eeg_npz)
    if source_idx < 0:
        source_idx = meg_lf.source_pos.shape[0] // 2
    src = meg_lf.source_pos[source_idx]

    # Best-channel peak per source (one number per Z position)
    meg_peak = per_source_peak_amplitude(meg_lf.L_fT_per_nAm)   # fT/nAm
    eeg_peak = per_source_peak_amplitude(eeg_lf.L_fT_per_nAm)   # µV/nAm
    z = meg_lf.source_pos[:, 2]

    # ── figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11.5))
    gs = GridSpec(2, 3, figure=fig,
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
            n = (snr_threshold * sigma / np.maximum(sig, 1e-30)) ** 2
            all_n.append(n)
            ax.plot(z, n, color=col, lw=1.4, label=sc.label)
        ax.set_xlabel("Source z (mm)")
        ax.set_ylabel(f"Trials needed for SNR ≥ {snr_threshold:g}")
        ax.set_yscale("log")
        # auto-extend ylim so EEG (≫ max_trials) actually shows up
        if ymax is None:
            data_top = float(np.max(all_n)) if all_n else max_trials
            ymax = max(max_trials, data_top * 1.5)
        ax.set_ylim(1, ymax)
        ax.axhline(max_trials, color=NATURE_PALETTE["axis"], lw=0.5,
                   linestyle=":", alpha=0.6)
        ax.text(z.min(), max_trials * 1.4, f"{max_trials:.0e} trials",
                fontsize=6.5, color=NATURE_PALETTE["axis"], alpha=0.8)

    def _plot_recording_time(ax, peak_one_source: float, sigma: float, rates_hz):
        for rate, col in zip(rates_hz, scenario_colors[:len(rates_hz)], strict=False):
            seconds = []
            for sc in scenarios:
                sig = peak_one_source * sc.Q_nAm
                if sig <= 0:
                    seconds.append(np.inf)
                    continue
                n = (snr_threshold * sigma / sig) ** 2
                seconds.append(n / rate)
            ax.plot([sc.Q_nAm for sc in scenarios], seconds,
                    marker="o", lw=1.4, color=col, label=f"{rate:g} Hz CAP rate")
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
    ax_b.set_title("MEG  ·  trials-to-detect along the vagus")
    ax_b.legend(loc="upper right", fontsize=6.5, handlelength=1.4)
    add_panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[0, 2])
    _plot_recording_time(ax_c, meg_peak[source_idx], sigma_meg,
                         rates_hz=(1.0, 5.0, 20.0))
    ax_c.set_title(f"MEG  ·  recording time @ source z = {src[2]:.0f} mm")
    add_panel_label(ax_c, "c")

    # ── EEG row ────────────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    _plot_snr_curves(ax_d, eeg_peak[source_idx], sigma_eeg,
                     "EEG SNR (best-contact)")
    ax_d.set_title(f"EEG  ·  SNR vs N trials  ·  noise σ = {sigma_eeg:.1f} µV")
    add_panel_label(ax_d, "d")

    ax_e = fig.add_subplot(gs[1, 1])
    _plot_trials_per_source(ax_e, eeg_peak, sigma_eeg)
    ax_e.set_title("EEG  ·  trials-to-detect along the vagus")
    ax_e.legend(loc="upper right", fontsize=6.5, handlelength=1.4)
    add_panel_label(ax_e, "e")

    ax_f = fig.add_subplot(gs[1, 2])
    _plot_recording_time(ax_f, eeg_peak[source_idx], sigma_eeg,
                         rates_hz=(1.0, 5.0, 20.0))
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

    out = out_path or cfg.outputs.base / "detectability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    logger.info("[saved] %s", out)
    plt.close(fig)
    return out


# ── headline numbers (printed in the CLI) ──────────────────────────────────

def detectability_summary(cfg: Config, *, source_idx: int = -1) -> dict:
    """Compute headline detectability numbers as a JSON-serialisable dict."""
    floors = compute_noise_floors(cfg)
    sigma_meg = floors.meg_per_channel_fT
    sigma_eeg = floors.eeg_per_channel_uV

    meg_lf = load_leadfield(cfg.outputs.forward_npz)
    eeg_lf = load_leadfield(cfg.outputs.forward_eeg_npz)
    if source_idx < 0:
        source_idx = meg_lf.source_pos.shape[0] // 2

    meg_peak = float(per_source_peak_amplitude(meg_lf.L_fT_per_nAm)[source_idx])
    eeg_peak = float(per_source_peak_amplitude(eeg_lf.L_fT_per_nAm)[source_idx])
    src = meg_lf.source_pos[source_idx]

    rows = {}
    for sc in DEFAULT_SCENARIOS:
        sig_meg = meg_peak * sc.Q_nAm
        sig_eeg = eeg_peak * sc.Q_nAm
        rows[sc.label] = {
            "Q_nAm": sc.Q_nAm,
            "MEG_per_trial_fT": sig_meg,
            "EEG_per_trial_uV": sig_eeg,
            "MEG_single_trial_SNR": sig_meg / sigma_meg if sigma_meg else float("inf"),
            "EEG_single_trial_SNR": sig_eeg / sigma_eeg if sigma_eeg else float("inf"),
            "MEG_trials_for_SNR3": required_trials(sig_meg, sigma_meg, 3.0),
            "EEG_trials_for_SNR3": required_trials(sig_eeg, sigma_eeg, 3.0),
        }
    return {
        "source_idx": int(source_idx),
        "source_z_mm": float(src[2]),
        "noise_meg_fT": float(sigma_meg),
        "noise_eeg_uV": float(sigma_eeg),
        "bandwidth_hz": float(cfg.noise.bandwidth_hz),
        "scenarios": rows,
    }
