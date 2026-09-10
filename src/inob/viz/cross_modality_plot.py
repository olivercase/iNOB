"""Cross-modality MEG↔EEG coupling figure (Nature Reviews-styled).

Four panels, telling the "if you see X on the EEG you should see Y on the
MEG" story:

  a  Per-source amplitude scatter: MEG_RMS vs EEG_RMS across the cervical
     source line. Pearson r quantifies cross-modality predictability.
  b  Observed EEG topo for a representative source.
  c  *Predicted* MEG topo, derived purely from the EEG observation in (b)
     by least-squares-fitting the 3-moment dipole and applying L_MEG.
  d  Actual (FEM) MEG topo for the same source — the ground truth that (c)
     is trying to recover.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from inob.analysis.cross_modality import (
    amplitude_correlation,
    per_source_amplitude,
    predict_meg_from_eeg,
)
from inob.config import Config, target_output
from inob.io.hdf5 import load_geometry, load_sensors
from inob.io.npz import load_leadfield
from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
    divergent_cmap,
    divergent_norm,
    save_figure,
)
from inob.viz.topoplot import (
    _draw_eeg_2d_topoplot,
    _draw_skin_3d,
    _format_3d_axis,
    _radial_channel_mask,
)

logger = logging.getLogger(__name__)


def render_cross_modality(
    cfg: Config,
    *,
    source_idx: int = -1,
    out_path: Path | None = None,
    dpi: int = 300,
    noise_uV: float | None = None,
    noise_seed: int = 0,
) -> Path:
    """Build the 4-panel cross-modality coupling figure.

    ``noise_uV`` adds Gaussian noise (RMS µV) to the EEG observation before
    the lstsq inversion — this is what gives the figure pedagogical content,
    since the noise-free round trip is exact by construction. Default ``None``
    means: pull a realistic value from ``cfg.noise`` (HD-EMG amplifier +
    Johnson noise integrated over the ``cfg.noise`` recording band).
    """
    apply_nature_style()
    meg_lf = load_leadfield(cfg.outputs.forward_npz)
    eeg_lf = load_leadfield(cfg.outputs.forward_eeg_npz)
    electrodes = load_sensors(cfg.outputs.electrodes_mat)

    if source_idx < 0:
        source_idx = meg_lf.source_pos.shape[0] // 2
    src = meg_lf.source_pos[source_idx]

    # ── per-source amplitude correlation ───────────────────────────────────
    a_meg = per_source_amplitude(meg_lf.L_fT_per_nAm)
    a_eeg = per_source_amplitude(eeg_lf.L_fT_per_nAm)
    stats = amplitude_correlation(meg_lf.L_fT_per_nAm, eeg_lf.L_fT_per_nAm)

    # ── observed EEG topo + 3-moment fit + predicted MEG ───────────────────
    L_eeg_src = eeg_lf.L_fT_per_nAm[:, 3 * source_idx : 3 * source_idx + 3]  # (C_e, 3)
    L_meg_src = meg_lf.L_fT_per_nAm[:, 3 * source_idx : 3 * source_idx + 3]  # (C_m, 3)
    # The "observed" EEG: the longitudinal moment column (i.e. q = ẑ).
    V_eeg_obs_clean = L_eeg_src[:, 2].copy()

    # Add a realistic noise floor — without this the lstsq inversion is exact
    # by construction (V_obs = L_e q  ⇒  pinv(L_e) V_obs = q exactly).
    if noise_uV is None:
        # Use the same noise model as analysis/snr.py.
        from inob.analysis.snr import compute_noise_floors

        noise_uV = compute_noise_floors(cfg).eeg_per_channel_uV
    rng = np.random.default_rng(int(noise_seed))
    V_eeg_obs = V_eeg_obs_clean + rng.normal(0, float(noise_uV), V_eeg_obs_clean.shape)
    snr_db = 20.0 * np.log10(
        max(np.linalg.norm(V_eeg_obs_clean), 1e-30)
        / max(noise_uV * np.sqrt(len(V_eeg_obs_clean)), 1e-30)
    )

    B_meg_pred = predict_meg_from_eeg(L_meg_src, L_eeg_src, V_eeg_obs)
    B_meg_true = L_meg_src[:, 2]

    radial = _radial_channel_mask(list(meg_lf.channel_names))
    keep_z = (meg_lf.coil_pos[:, 2] >= src[2] - 220.0) & (meg_lf.coil_pos[:, 2] <= src[2] + 220.0)
    keep = radial & keep_z
    pos_meg = meg_lf.coil_pos[keep]
    pred_vals = B_meg_pred[keep]
    true_vals = B_meg_true[keep]

    fig = plt.figure(figsize=(13.5, 11.5))
    gs = GridSpec(
        2, 2, figure=fig, left=0.05, right=0.97, top=0.93, bottom=0.06, hspace=0.30, wspace=0.22
    )

    # ── panel a: amplitude scatter ─────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    z = meg_lf.source_pos[:, 2]
    sc = ax_a.scatter(
        a_eeg, a_meg, c=z, cmap="magma", s=22, edgecolor=NATURE_PALETTE["axis"], linewidths=0.2
    )
    ax_a.scatter(
        [a_eeg[source_idx]],
        [a_meg[source_idx]],
        s=180,
        c=NATURE_PALETTE["glow"],
        edgecolor=NATURE_PALETTE["axis"],
        linewidths=0.8,
        marker="*",
        label="Highlighted source",
    )
    ax_a.set_xlabel("EEG amplitude  ·  RMS µV  (1 nA·m source)")
    ax_a.set_ylabel("MEG amplitude  ·  RMS fT  (1 nA·m source)")
    ax_a.set_title(
        f"Per-source amplitude coupling  ·  "
        f"Pearson r = {stats.pearson_r:.2f},  "
        f"log–log slope = {stats.log_log_slope:.2f}"
    )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    cb = fig.colorbar(sc, ax=ax_a, shrink=0.85, fraction=0.04, pad=0.02)
    cb.set_label("Source z (mm)", fontsize=8)
    cb.outline.set_visible(False)
    ax_a.legend(loc="lower right", handlelength=1.0)
    add_panel_label(ax_a, "a")

    # ── panel b: observed EEG topo ─────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    vmin_b, vmax_b = divergent_norm(V_eeg_obs)
    _draw_eeg_2d_topoplot(ax_b, electrodes, V_eeg_obs, cfg, vmin=vmin_b, vmax=vmax_b)
    ax_b.set_title(f"Observed EEG  ·  source z = {src[2]:.0f} mm  (longitudinal moment)")
    add_panel_label(ax_b, "b")

    # ── panel c: predicted MEG (from EEG inversion) ────────────────────────
    ax_c = fig.add_subplot(gs[1, 0], projection="3d")
    try:
        geom = load_geometry(cfg.outputs.geometry_mat)
        skin = geom.compartments.get("mesh_skin")
        if skin is not None:
            _draw_skin_3d(ax_c, skin.vertices, skin.faces, alpha=0.05)
    except Exception:
        # The skin backdrop is decorative; the data plot must still draw.
        logger.debug("skin backdrop skipped", exc_info=True)
    vmin_c, vmax_c = divergent_norm(np.r_[pred_vals, true_vals])
    sc_c = ax_c.scatter(
        pos_meg[:, 0],
        pos_meg[:, 1],
        pos_meg[:, 2],
        c=pred_vals,
        cmap=divergent_cmap(),
        vmin=vmin_c,
        vmax=vmax_c,
        s=44,
        edgecolor=NATURE_PALETTE["axis"],
        linewidths=0.25,
        depthshade=False,
    )
    ax_c.scatter(
        [src[0]],
        [src[1]],
        [src[2]],
        s=160,
        c=NATURE_PALETTE["glow"],
        edgecolor=NATURE_PALETTE["axis"],
        linewidths=0.8,
        marker="*",
    )
    _format_3d_axis(ax_c, src=src, pos_for_lim=pos_meg)
    ax_c.set_title("Predicted MEG  ·  recovered from the EEG topo via shared source space")
    cb_c = fig.colorbar(sc_c, ax=ax_c, shrink=0.55, pad=0.06, fraction=0.04)
    cb_c.set_label("fT  (1 nA·m source)", fontsize=8)
    cb_c.outline.set_visible(False)
    add_panel_label(ax_c, "c")

    # ── panel d: actual MEG (ground truth) ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1], projection="3d")
    try:
        if skin is not None:
            _draw_skin_3d(ax_d, skin.vertices, skin.faces, alpha=0.05)
    except Exception:
        # The skin backdrop is decorative; the data plot must still draw.
        logger.debug("skin backdrop skipped", exc_info=True)
    sc_d = ax_d.scatter(
        pos_meg[:, 0],
        pos_meg[:, 1],
        pos_meg[:, 2],
        c=true_vals,
        cmap=divergent_cmap(),
        vmin=vmin_c,
        vmax=vmax_c,
        s=44,
        edgecolor=NATURE_PALETTE["axis"],
        linewidths=0.25,
        depthshade=False,
    )
    ax_d.scatter(
        [src[0]],
        [src[1]],
        [src[2]],
        s=160,
        c=NATURE_PALETTE["glow"],
        edgecolor=NATURE_PALETTE["axis"],
        linewidths=0.8,
        marker="*",
    )
    _format_3d_axis(ax_d, src=src, pos_for_lim=pos_meg)
    res = pred_vals - true_vals
    rel_err = float(np.sqrt(np.mean(res**2))) / max(float(np.sqrt(np.mean(true_vals**2))), 1e-30)
    # Cap the displayed error at 1000 % — beyond that the inversion is meaningless
    # and the exact percentage is not informative.
    err_str = f"{100 * rel_err:.1f}%" if rel_err < 10.0 else ">1000% (EEG below noise floor)"
    ax_d.set_title(
        f"Actual MEG (FEM)  ·  prediction RMS error = {err_str}  "
        f"(EEG SNR ≈ {snr_db:.0f} dB at {noise_uV:.2g} µV)"
    )
    cb_d = fig.colorbar(sc_d, ax=ax_d, shrink=0.55, pad=0.06, fraction=0.04)
    cb_d.set_label("fT  (1 nA·m source)", fontsize=8)
    cb_d.outline.set_visible(False)
    add_panel_label(ax_d, "d")

    fig.suptitle(
        "Cross-modality coupling — given a known source position, fit moment "
        "from EEG and predict MEG (forward only; not source localisation)",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )

    logger.info(
        "cross-modality Pearson r=%.3f, slope=%.3f, RMS err=%.1f%%",
        stats.pearson_r,
        stats.log_log_slope,
        100 * rel_err,
    )
    return save_figure(fig, out_path or target_output(cfg, "cross_modality.png"), dpi=dpi)
