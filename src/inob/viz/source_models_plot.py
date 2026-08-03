"""Figure: OPM vs electrodes across cord source models.

Three panels, in the order the argument runs:

  a  Single-trial SNR per source model, both modalities, against the SNR = 3
     detection line. The absolute heights depend on the assumed Q; the *shape*
     — which bars clear the line and which do not — is the result.
  b  Trials to reach SNR = 3, log scale, with the clinical 500–2,000 average
     budget shaded for the evoked targets that have one.
  c  The Q-independent ratios: the modality gap for each source model, and the
     propagation and coherence penalties. Nothing in this panel moves if the
     true source strength turns out to be ten times what was assumed, which is
     why it is the panel to read first.

See :mod:`inob.analysis.source_models` for what is held fixed to make the
comparison like-for-like.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from inob.analysis.snr import compute_noise_floors
from inob.analysis.source_models import (
    SNR_TARGET,
    SourceModelComparison,
    compare_source_models,
)
from inob.config import Config, source_region_label, source_target_tag, target_output
from inob.io.npz import load_leadfield
from inob.viz.style import (
    NATURE_PALETTE,
    apply_nature_style,
    save_figure,
)

logger = logging.getLogger(__name__)

MEG_COLOUR = NATURE_PALETTE["blue"]
EEG_COLOUR = NATURE_PALETTE["red"]

#: Pretty names for the axis; the analysis keys stay machine-readable.
MODEL_LABELS: dict[str, str] = {
    "synchronous": "synchronous\n(whole cord in phase)",
    "stationary": "stationary\n(one segment)",
    "ascending": "ascending\n(travelling volley)",
}


def _bar_pair(ax, values_meg, values_eeg, labels, *, log: bool = False):
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, values_meg, w, color=MEG_COLOUR, label="OPM (best channel)")
    ax.bar(x + w / 2, values_eeg, w, color=EEG_COLOUR,
           label="electrodes (best bipolar)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    if log:
        ax.set_yscale("log")
    return x


def render_source_models(
    cfg: Config, *, Q_nAm: float | None = None, source_idx: int | None = None,
    out_path: Path | None = None, dpi: int = 300,
) -> Path:
    """Render the like-for-like modality comparison."""
    from inob.physiology.profiles import profile_for_tag
    from inob.viz.detectability import clinical_average_budget, default_source_idx

    apply_nature_style()
    meg = load_leadfield(cfg.outputs.forward_npz)
    eeg = load_leadfield(cfg.outputs.forward_eeg_npz)
    profile = profile_for_tag(source_target_tag(cfg))
    if source_idx is None:
        source_idx = default_source_idx(meg, eeg)
    if Q_nAm is None:
        Q_nAm = profile.default_strength_nAm
    floors = compute_noise_floors(cfg)
    cmp: SourceModelComparison = compare_source_models(
        meg, eeg, profile, Q_nAm=Q_nAm, source_idx=source_idx,
        meg_sigma_fT=floors.meg_per_channel_fT,
        eeg_sigma_uV=floors.eeg_per_channel_uV,
    )
    labels = [MODEL_LABELS.get(r.name, r.name) for r in cmp.rows]

    fig = plt.figure(figsize=(13, 4.4))
    gs = GridSpec(1, 3, figure=fig, left=0.06, right=0.985, top=0.82,
                  bottom=0.20, wspace=0.28)

    # ── a: single-trial SNR ────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _bar_pair(ax_a, [r.meg_snr for r in cmp.rows],
              [r.eeg_snr for r in cmp.rows], labels, log=True)
    ax_a.axhline(SNR_TARGET, color=NATURE_PALETTE["axis"], lw=1.0, ls="--")
    # Right-aligned: the tallest bar is the leftmost one, so a left-aligned
    # annotation lands on top of it.
    ax_a.text(0.98, SNR_TARGET * 1.2, f"SNR = {SNR_TARGET:g} (Rose criterion)",
              transform=ax_a.get_yaxis_transform(), fontsize=7, ha="right",
              color=NATURE_PALETTE["axis"])
    ax_a.set_ylabel("single-trial SNR")
    ax_a.set_title("a  single-trial SNR", fontsize=9, loc="left")
    ax_a.legend(fontsize=7, frameon=False, loc="lower left")

    # ── b: trials to detect ────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _bar_pair(ax_b, [r.meg_trials for r in cmp.rows],
              [r.eeg_trials for r in cmp.rows], labels, log=True)
    budget = clinical_average_budget(cfg)
    if budget is not None:
        ax_b.axhspan(*budget, color=NATURE_PALETTE["glow"], alpha=0.22, zorder=0)
        ax_b.text(0.02, budget[1], f"  clinical budget {budget[0]}–{budget[1]}",
                  transform=ax_b.get_yaxis_transform(), fontsize=7, va="bottom",
                  color=NATURE_PALETTE["axis"])
    ax_b.set_ylabel(f"trials to reach SNR {SNR_TARGET:g}")
    ax_b.set_title("b  averaging cost", fontsize=9, loc="left")

    # ── c: the ratios that do not depend on Q ──────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    ratio_labels = [f"OPM/elec\n{r.name}" for r in cmp.rows] + [
        "propagation\npenalty (OPM)", "propagation\npenalty (elec)",
        "coherence\npenalty (OPM)",
    ]
    ratio_values = [r.modality_gap for r in cmp.rows] + [
        cmp.meg_propagation_factor, cmp.eeg_propagation_factor,
        cmp.coherence_penalty,
    ]
    colours = ([NATURE_PALETTE["purple"]] * len(cmp.rows)
               + [MEG_COLOUR, EEG_COLOUR, NATURE_PALETTE["teal"]])
    ax_c.bar(np.arange(len(ratio_values)), ratio_values, 0.6, color=colours)
    ax_c.axhline(1.0, color=NATURE_PALETTE["axis"], lw=1.0, ls=":")
    ax_c.set_yscale("log")
    ax_c.set_xticks(np.arange(len(ratio_labels)))
    ax_c.set_xticklabels(ratio_labels, fontsize=6.5, rotation=30, ha="right")
    ax_c.set_ylabel("ratio (×)")
    ax_c.set_title("c  independent of source strength", fontsize=9, loc="left")

    region = source_region_label(cfg)
    fig.suptitle(
        f"{region}: OPM vs surface electrodes on the same cord activity — "
        f"Q = {cmp.Q_nAm:g} nA·m per active source, source #{cmp.source_idx} "
        f"(z = {cmp.source_z_mm:.0f} mm)",
        fontsize=11, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.015,
        f"{cmp.n_sources} sources over {cmp.span_mm:.0f} mm; transit "
        f"{cmp.transit_ms:.1f} ms vs {cmp.ap_width_ms:.1f} ms AP width. "
        "Panels a–b scale linearly with Q; panel c does not depend on it.",
        ha="center", va="bottom", fontsize=7, style="italic",
        color=NATURE_PALETTE["axis"],
    )
    out = out_path or target_output(cfg, "source_models.png")
    return save_figure(fig, out, dpi=dpi)
