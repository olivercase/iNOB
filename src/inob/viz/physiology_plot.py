"""Nature-styled figure for the moving-dipole physiology simulations.

Six panels per scenario (baroreceptor / deep-breathing) — same layout for
both, allowing a side-by-side or stacked headline figure:

  a  Stimulus context: ECG R-waves (baro) or lung-volume trace (resp).
  b  Best-MEG-channel time trace.
  c  Best-EEG-channel time trace.
  d  Predicted SNR vs N trials at the realistic noise floors.
  e  MEG topoplot at the peak time.
  f  Spectrum of the best MEG channel — does the cardiac / respiratory
     frequency peak emerge above the noise floor?
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from inob.analysis.snr import compute_noise_floors
from inob.config import Config
from inob.io.npz import load_leadfield
from inob.physiology.scenarios import (
    Scenario,
    baroreceptor_scenario,
    respiratory_scenario,
)
from inob.physiology.simulate import (
    SimulatedSignal,
    best_channel_index,
    simulate_train,
)
from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
)
from inob.viz.topoplot import (
    _radial_channel_mask,
)

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────

def _to_human_units(sig_T_or_V: np.ndarray, *, modality: str) -> tuple[np.ndarray, str]:
    """Convert raw forward output into fT (MEG) or µV (EEG)."""
    if modality == "meg":
        return sig_T_or_V * 1e15, "fT"
    if modality == "eeg":
        return sig_T_or_V * 1e6, "µV"
    raise ValueError(f"unknown modality {modality!r}")


def _row_for_modality(
    fig, gs_row, sim: SimulatedSignal, *, cfg: Config, modality: str,
    label_prefix: str, panel_letters: tuple[str, str, str],
):
    """Render the time-trace + topoplot + SNR panels for one modality.

    ``gs_row`` is a 3-column GridSpec slice. Returns the time-trace axis so
    the caller can add a stimulus underlay.
    """
    floors = compute_noise_floors(cfg)
    if modality == "meg":
        sigma = floors.meg_per_channel_fT
    else:
        sigma = floors.eeg_per_channel_uV

    sig_h, unit = _to_human_units(sim.signal, modality=modality)
    best_c = best_channel_index(sig_h)
    trace = sig_h[best_c]

    ax_a = fig.add_subplot(gs_row[0])
    ax_a.plot(sim.t_s, trace, color=(NATURE_PALETTE["blue"]
              if modality == "meg" else NATURE_PALETTE["red"]), lw=0.8)
    ax_a.axhline(sigma, color=NATURE_PALETTE["axis"], lw=0.5, linestyle="--",
                 label=f"σ = {sigma:.1f} {unit}")
    ax_a.axhline(-sigma, color=NATURE_PALETTE["axis"], lw=0.5, linestyle="--")
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel(f"{label_prefix} signal at best channel  ·  {unit}")
    snr_single = float(np.abs(trace).max() / sigma)
    ax_a.set_title(
        f"{label_prefix}  ·  best channel #{best_c}  ·  "
        f"single-trial peak-SNR = {snr_single:.2f}"
    )
    ax_a.legend(loc="upper right", fontsize=7, handlelength=1.2)
    add_panel_label(ax_a, panel_letters[0])

    # Topoplot at peak time (for MEG only — EEG covered by the patch heatmap
    # in the existing topoplot module, here we just show a sensor scatter).
    return ax_a, sig_h, sigma, best_c, unit


def render_physiology(
    cfg: Config,
    *,
    scenarios: tuple[Scenario, ...] | None = None,
    fs_hz: float = 30_000.0,
    out_path: Path | None = None,
    dpi: int = 300,
) -> Path:
    """Render the full Nature-style physiology figure (2 scenarios × 3 panels)."""
    apply_nature_style()
    if scenarios is None:
        scenarios = (
            baroreceptor_scenario(duration_s=6.0),
            respiratory_scenario(duration_s=12.0, breath_bpm=6.0),
        )

    meg_lf = load_leadfield(cfg.outputs.forward_npz)
    eeg_lf = load_leadfield(cfg.outputs.forward_eeg_npz)

    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(
        len(scenarios), 4, figure=fig,
        left=0.05, right=0.97, top=0.93, bottom=0.07,
        hspace=0.45, wspace=0.30,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
    )

    floors = compute_noise_floors(cfg)
    sigma_meg = floors.meg_per_channel_fT
    sigma_eeg = floors.eeg_per_channel_uV

    panel_letters = "abcdefghijkl"
    pl_idx = 0

    for row, sc in enumerate(scenarios):
        logger.info("[physiology] simulating %s (%d events)…", sc.name, len(sc.events))
        meg_sim = simulate_train(meg_lf, sc, fs_hz=fs_hz)
        eeg_sim = simulate_train(eeg_lf, sc, fs_hz=fs_hz)

        meg_h = meg_sim.signal * 1e15        # T → fT
        eeg_h = eeg_sim.signal * 1e6         # V → µV

        # Best channels (MEG: radial subset; EEG: any contact)
        meg_radial = _radial_channel_mask(list(meg_lf.channel_names))
        meg_best = int(np.argmax(
            np.sqrt(np.mean(meg_h ** 2, axis=1)) * meg_radial
        ))
        eeg_best = best_channel_index(eeg_h)

        # ── col 0: stimulus + time traces (overlaid) ─────────────────────────
        ax0 = fig.add_subplot(gs[row, 0])
        if sc.physiology_trace is not None:
            t_phys = np.linspace(0, sc.duration_s, len(sc.physiology_trace))
            ax0.plot(t_phys, sc.physiology_trace,
                     color=NATURE_PALETTE["axis"], lw=0.8, alpha=0.6,
                     label=sc.physiology_trace_label)
        ax_top = ax0.twinx()
        ax_top.plot(meg_sim.t_s, meg_h[meg_best],
                    color=NATURE_PALETTE["blue"], lw=0.7,
                    label=f"MEG #{meg_best}")
        ax_top.set_ylabel("MEG (fT)", color=NATURE_PALETTE["blue"], fontsize=8)
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel(sc.physiology_trace_label, fontsize=7.5)
        ax0.set_title(f"{sc.name}  ·  rate = {sc.rate_hz:.2f} Hz", fontsize=10)
        ax0.legend(loc="upper left", fontsize=6.5, handlelength=1.1)
        ax_top.legend(loc="upper right", fontsize=6.5, handlelength=1.1)
        add_panel_label(ax0, panel_letters[pl_idx])
        pl_idx += 1

        # ── col 1: MEG vs EEG traces (paired axes for comparison) ───────────
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.plot(meg_sim.t_s, meg_h[meg_best] / sigma_meg,
                 color=NATURE_PALETTE["blue"], lw=0.8,
                 label=f"MEG / σ_MEG  (peak {np.abs(meg_h[meg_best]).max() / sigma_meg:.1f})")
        ax1.plot(eeg_sim.t_s, eeg_h[eeg_best] / sigma_eeg,
                 color=NATURE_PALETTE["red"], lw=0.8,
                 label=f"EEG / σ_EEG  (peak {np.abs(eeg_h[eeg_best]).max() / sigma_eeg:.2e})")
        ax1.axhline(3, color=NATURE_PALETTE["axis"], lw=0.6, linestyle="--",
                    label="SNR=3")
        ax1.axhline(-3, color=NATURE_PALETTE["axis"], lw=0.6, linestyle="--")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Signal / single-trial noise σ")
        ax1.set_title("Single-trial SNR  ·  MEG vs EEG")
        ax1.legend(loc="upper right", fontsize=6.5, handlelength=1.1)
        add_panel_label(ax1, panel_letters[pl_idx])
        pl_idx += 1

        # ── col 2: trials-to-detect for each modality ───────────────────────
        ax2 = fig.add_subplot(gs[row, 2])
        peak_meg = float(np.abs(meg_h[meg_best]).max())
        peak_eeg = float(np.abs(eeg_h[eeg_best]).max())
        n_grid = np.logspace(0, 6, 200)
        snr_meg_n = peak_meg / sigma_meg * np.sqrt(n_grid)
        snr_eeg_n = peak_eeg / sigma_eeg * np.sqrt(n_grid)
        ax2.plot(n_grid, snr_meg_n, color=NATURE_PALETTE["blue"], lw=1.4, label="MEG")
        ax2.plot(n_grid, snr_eeg_n, color=NATURE_PALETTE["red"], lw=1.4, label="EEG")
        ax2.axhline(3, color=NATURE_PALETTE["axis"], lw=0.6, linestyle="--",
                    label="SNR = 3")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Averaged trials (cardiac / respiratory cycles)")
        ax2.set_ylabel("Predicted SNR")
        # Number of trials to detect at SNR=3
        n_meg = (3 * sigma_meg / max(peak_meg, 1e-30)) ** 2
        n_eeg = (3 * sigma_eeg / max(peak_eeg, 1e-30)) ** 2
        ax2.set_title(
            f"Trials to SNR=3  ·  MEG: {n_meg:.2g}  ·  EEG: {n_eeg:.2g}",
            fontsize=9,
        )
        ax2.legend(loc="lower right", fontsize=7, handlelength=1.1)
        add_panel_label(ax2, panel_letters[pl_idx])
        pl_idx += 1

        # ── col 3: MEG spectrum (does the cardiac/resp peak emerge?) ────────
        ax3 = fig.add_subplot(gs[row, 3])
        from numpy.fft import rfft, rfftfreq
        sig_for_fft = meg_h[meg_best] - meg_h[meg_best].mean()
        win = np.hanning(len(sig_for_fft))
        fft_mag = np.abs(rfft(sig_for_fft * win))
        fft_freq = rfftfreq(len(sig_for_fft), d=1.0 / fs_hz)
        # Show 0–10 Hz for cardiac/respiratory; cap at peak
        mask = (fft_freq <= 10.0)
        ax3.plot(fft_freq[mask], fft_mag[mask],
                 color=NATURE_PALETTE["blue"], lw=0.8)
        ax3.axvline(sc.rate_hz, color=NATURE_PALETTE["red"], lw=0.8,
                    linestyle="--", label=f"f₀ = {sc.rate_hz:.2f} Hz")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("|FFT(MEG)|  (a.u.)")
        ax3.set_title("Spectrum  ·  best MEG channel")
        ax3.legend(loc="upper right", fontsize=7, handlelength=1.1)
        add_panel_label(ax3, panel_letters[pl_idx])
        pl_idx += 1

    fig.suptitle(
        "Moving-dipole vagal CAP simulation  —  baroreceptor (cardiac-locked)  "
        "and slow / deep-breathing (respiratory)",
        fontsize=12, fontweight="bold", y=0.985,
    )
    out = out_path or cfg.outputs.base / "physiology.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    logger.info("[saved] %s", out)
    plt.close(fig)
    return out
