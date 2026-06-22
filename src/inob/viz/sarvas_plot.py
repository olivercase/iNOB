"""Sarvas-vs-FEM comparison figure (Nature Reviews-styled)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from inob.analysis.sarvas_compare import SarvasVsFemResult
from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
    divergent_cmap,
)

logger = logging.getLogger(__name__)


def render_sarvas_vs_fem(
    result: SarvasVsFemResult, *, source_idx: int = -1,
    out_path: Path, dpi: int = 300,
) -> Path:
    """Four-panel benchmark figure:

      a  scatter: FEM vs Sarvas per coil (single source)
      b  per-coil residual vs sensor-axis distance
      c  Sarvas + FEM peak |B| along the cervical vagus
      d  FEM/Sarvas amplitude-ratio histogram

    Y-axis units follow the leadfield convention used throughout this repo:
    **fT per 1 nA·m source** when ``result.Q_nAm == 1.0``. For physiological
    rescaling (e.g. ``Q_nAm = 70``), the same numbers can be read as pT —
    the figure annotates this in the suptitle.
    """
    apply_nature_style()
    if source_idx < 0:
        source_idx = result.source_pos_mm.shape[0] // 2
    src = result.source_pos_mm[source_idx]
    # Display in fT (1 nA·m convention). Result fields are in Tesla for the
    # given Q_nAm — multiplying by 1e15 / Q_nAm divides out the source.
    scale_fT = 1.0e15 / result.Q_nAm
    sarvas_disp = result.sarvas_T * scale_fT
    fem_disp = result.fem_T * scale_fT
    unit_label = f"fT  (per {result.Q_nAm:g} nA·m source)" if result.Q_nAm != 1.0 \
                  else "fT  (per 1 nA·m source)"

    fig = plt.figure(figsize=(13.5, 11.5))
    gs = GridSpec(2, 2, figure=fig, left=0.07, right=0.97, top=0.93, bottom=0.07,
                  hspace=0.32, wspace=0.28)

    # ── panel a: per-coil scatter (single source) ──────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    s_d = sarvas_disp[:, source_idx]
    f_d = fem_disp[:, source_idx]
    ax_a.scatter(s_d, f_d, s=18, alpha=0.7,
                 color=NATURE_PALETTE["blue"], edgecolor="none")
    lim = max(float(np.abs(s_d).max()), float(np.abs(f_d).max())) * 1.05
    ax_a.plot([-lim, lim], [-lim, lim], lw=0.7, color=NATURE_PALETTE["axis"],
              linestyle="--", label="y = x")
    ax_a.set_xlabel(f"Sarvas analytic  ·  {unit_label}")
    ax_a.set_ylabel(f"FEM (DUNEuro)  ·  {unit_label}")
    ax_a.set_xlim(-lim, lim)
    ax_a.set_ylim(-lim, lim)
    ax_a.set_aspect("equal")
    ax_a.set_title(f"Per-coil agreement  ·  source z = {src[2]:.0f} mm")
    ax_a.legend(loc="upper left", handlelength=1.2)
    add_panel_label(ax_a, "a")

    # ── panel b: per-coil residual vs distance ─────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    res_d = fem_disp[:, source_idx] - sarvas_disp[:, source_idx]
    # distance_to_axis_mm is now (C, S) — pull this source's column.
    dist = result.distance_to_axis_mm
    if dist.ndim == 2:
        dist = dist[:, source_idx]
    sc = ax_b.scatter(
        dist, res_d,
        c=sarvas_disp[:, source_idx], cmap=divergent_cmap(),
        vmin=-np.abs(s_d).max(), vmax=np.abs(s_d).max(),
        s=24, edgecolor=NATURE_PALETTE["axis"], linewidths=0.2,
    )
    ax_b.axhline(0, lw=0.6, color=NATURE_PALETTE["axis"], linestyle="--")
    ax_b.axvline(58.5, lw=0.6, color=NATURE_PALETTE["red"], linestyle=":",
                 label="58.5 mm (literature sensor-axis)")
    ax_b.set_xlabel("Coil distance to cervical axis  ·  mm")
    ax_b.set_ylabel(f"FEM − Sarvas residual  ·  {unit_label}")
    ax_b.set_title("Residual vs sensor distance")
    ax_b.legend(loc="upper right", fontsize=7, handlelength=1.0)
    cb = fig.colorbar(sc, ax=ax_b, shrink=0.85, fraction=0.04, pad=0.02)
    cb.set_label(f"Sarvas  ·  {unit_label}", fontsize=8)
    cb.outline.set_visible(False)
    add_panel_label(ax_b, "b")

    # ── panel c: peak amplitude along the vagus ────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    z = result.source_pos_mm[:, 2]
    sarvas_peak = np.max(np.abs(sarvas_disp), axis=0)
    fem_peak = np.max(np.abs(fem_disp), axis=0)
    ax_c.plot(z, sarvas_peak, color=NATURE_PALETTE["blue"], lw=1.6,
              label="Sarvas analytic")
    ax_c.plot(z, fem_peak, color=NATURE_PALETTE["red"], lw=1.6,
              label="FEM (DUNEuro)")
    ax_c.fill_between(z, np.minimum(sarvas_peak, fem_peak),
                      np.maximum(sarvas_peak, fem_peak),
                      color=NATURE_PALETTE["stone"], alpha=0.45,
                      label="Difference band")
    ax_c.set_xlabel("Source z position along cervical vagus  ·  mm")
    ax_c.set_ylabel(f"Peak |B| over OPM array  ·  {unit_label}")
    ax_c.set_title("Peak field along the vagus")
    ax_c.legend(loc="upper right", handlelength=1.4)
    add_panel_label(ax_c, "c")

    # ── panel d: amplitude ratio histogram ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    flat_s = sarvas_disp.ravel()
    flat_f = fem_disp.ravel()
    keep = np.abs(flat_s) > 0.01 * np.abs(flat_s).max()
    ratio = flat_f[keep] / flat_s[keep]
    ax_d.hist(ratio, bins=80, color=NATURE_PALETTE["blue"], alpha=0.75)
    median = float(np.nanmedian(ratio))
    ax_d.axvline(1.0, color=NATURE_PALETTE["axis"], lw=0.8, linestyle="--",
                 label="ratio = 1")
    ax_d.axvline(median, color=NATURE_PALETTE["red"], lw=1.2,
                 label=f"median = {median:.2f}")
    ax_d.set_xlabel("FEM / Sarvas amplitude ratio  (unitless)")
    ax_d.set_ylabel("Channel-source pair count")
    ax_d.set_title("Amplitude-ratio distribution")
    ax_d.set_xlim(np.percentile(ratio, 1), np.percentile(ratio, 99))
    ax_d.legend(loc="upper right", handlelength=1.2)
    add_panel_label(ax_d, "d")

    fig.suptitle(
        f"Analytic Sarvas (single-sphere) vs full multi-tissue FEM forward  "
        f"·  source–axis 40 mm, sensor–axis 58.5 mm  "
        f"·  Q = {result.Q_nAm:g} nA·m",
        fontsize=11, fontweight="bold", y=0.985,
    )
    # Footer: explain the FEM/Sarvas amplification physics in one line.
    fig.text(
        0.5, 0.005,
        "FEM ≠ Sarvas reflects secondary (volume) currents in the multi-tissue "
        "conductor (Geselowitz 1970; Sarvas 1987; Hämäläinen et al. 1993). For "
        "cervical/spinal MEG, bone strongly attenuates lateral currents while "
        "longitudinal currents are nearly conductor-invariant (O'Neill et al. "
        "2025 Sci Rep). Sarvas is exact only for a homogeneous sphere.",
        ha="center", va="bottom", fontsize=7,
        color=NATURE_PALETTE["axis"], style="italic",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    logger.info("[saved] %s", out_path)
    plt.close(fig)
    return out_path
