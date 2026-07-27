"""Whole-body EEG sensitivity map: where SHOULD the patch go?

Compares the 32-channel cervical PEDOT:PSS paddle (`outputs/sensors/electrode_array.mat`
+ `outputs/forward/duneuro_eeg_leadfield_vagus.npz`) against a ~1000-electrode
whole-body array (`*_wholebody.mat` / `*_wholebody.npz`) for the same source
on the cervical vagus.

The question we answer: is the cervical paddle in the wrong place, or is
EEG fundamentally limited by bone shielding? If the whole-body peak |L|
matches the paddle peak |L|, the paddle is already optimal — moving the
electrodes won't fix the SNR. If the whole-body peak is much higher, then
the paddle is poorly sited.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from inob.analysis.snr import compute_noise_floors
from inob.config import Config, source_region_label, target_output
from inob.io.hdf5 import load_geometry, load_sensors
from inob.io.npz import load_leadfield
from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
    save_figure,
    sequential_cmap,
)
from inob.viz.surface_topoplot import gaussian_interpolate_surface

logger = logging.getLogger(__name__)


def _per_source_peak(L: np.ndarray) -> np.ndarray:
    """Peak |L| across (channels × moments) per source."""
    C, three_S = L.shape
    S = three_S // 3
    L3 = L.reshape(C, S, 3)
    return np.abs(L3).max(axis=(0, 2))


def _column_for_source_norm(L: np.ndarray, source_idx: int) -> np.ndarray:
    """L2 norm across the 3 moments for one source — orientation-agnostic
    detectability per channel."""
    block = L[:, 3 * source_idx : 3 * source_idx + 3]
    return np.linalg.norm(block, axis=1)


def render_location_optimisation(
    cfg: Config,
    *,
    paddle_npz: Path,
    wholebody_npz: Path,
    paddle_mat: Path,
    wholebody_mat: Path,
    source_idx: int = -1,
    out_path: Path | None = None,
    dpi: int = 300,
) -> Path:
    """Six-panel "is location the problem?" figure.

      a  Whole-body |L| heatmap interpolated on the full skin surface.
      b  Same map, unrolled cylinder (θ vs Z) — full-body view.
      c  Per-source best-channel SNR: paddle vs whole-body, along the vagus.
      d  Per-source peak |L|: paddle vs whole-body, along the vagus.
      e  Whole-body amplitude vs distance to source — does it follow 1/r²?
      f  Headline numbers panel (text): paddle peak, whole-body peak,
         "trials needed" comparison.
    """
    apply_nature_style()
    paddle_lf = load_leadfield(paddle_npz)
    wb_lf = load_leadfield(wholebody_npz)
    paddle_sensors = load_sensors(paddle_mat)
    wb_sensors = load_sensors(wholebody_mat)
    region = source_region_label(cfg)
    geom = load_geometry(cfg.outputs.geometry_mat)
    skin_comp = geom.compartments["mesh_skin"]
    skin = trimesh.Trimesh(skin_comp.vertices, skin_comp.faces, process=False)

    if source_idx < 0:
        source_idx = paddle_lf.source_pos.shape[0] // 2
    src = paddle_lf.source_pos[source_idx]

    # ── per-source data ───────────────────────────────────────────────────
    paddle_peak = _per_source_peak(paddle_lf.L_fT_per_nAm)      # (S,) µV/nAm
    wb_peak = _per_source_peak(wb_lf.L_fT_per_nAm)              # (S,) µV/nAm
    z = paddle_lf.source_pos[:, 2]

    # Whole-body single-source: orientation-agnostic L2 norm across moments
    wb_val = _column_for_source_norm(wb_lf.L_fT_per_nAm, source_idx)   # (1000,)
    paddle_val = _column_for_source_norm(paddle_lf.L_fT_per_nAm, source_idx)

    # Argmax channel on whole-body
    wb_best_idx = int(np.argmax(wb_val))
    wb_best_pos = wb_sensors.coilpos[wb_best_idx]
    wb_best_val = float(wb_val[wb_best_idx])
    paddle_best_idx = int(np.argmax(paddle_val))
    paddle_best_pos = paddle_sensors.coilpos[paddle_best_idx]
    paddle_best_val = float(paddle_val[paddle_best_idx])

    # ── interpolate whole-body |L| onto skin for the heatmap ──────────────
    skin_vals = gaussian_interpolate_surface(
        wb_sensors.coilpos, wb_val, skin.vertices,
        sigma_mm=60.0, k_nearest=24,
    )

    # ── figure ────────────────────────────────────────────────────────────
    cmap = sequential_cmap()
    vmax = float(wb_val.max() * 1.05)

    fig = plt.figure(figsize=(15, 11.5))
    gs = GridSpec(2, 3, figure=fig,
                  left=0.04, right=0.97, top=0.92, bottom=0.07,
                  hspace=0.34, wspace=0.24)

    # ── panel a: whole-body heatmap on skin ────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    rng = np.random.default_rng(int(cfg.reproducibility.seed))
    n_max_tris = 50_000
    faces = skin.faces
    if len(faces) > n_max_tris:
        idx = rng.choice(len(faces), n_max_tris, replace=False)
        faces = faces[idx]
    face_vals = skin_vals[faces].mean(axis=1)
    norm = np.clip(face_vals / max(vmax, 1e-30), 0.0, 1.0)
    coll = Poly3DCollection(
        skin.vertices[faces], facecolor=cmap(norm), edgecolor="none", alpha=0.95,
    )
    ax_a.add_collection3d(coll)
    ax_a.scatter([src[0]], [src[1]], [src[2]], s=200, c=NATURE_PALETTE["glow"],
                 edgecolor=NATURE_PALETTE["axis"], linewidths=0.8, marker="*")
    ax_a.scatter([wb_best_pos[0]], [wb_best_pos[1]], [wb_best_pos[2]],
                 s=120, c=NATURE_PALETTE["red"],
                 edgecolor=NATURE_PALETTE["axis"], linewidths=0.6, marker="o",
                 label="whole-body argmax")
    ax_a.scatter([paddle_best_pos[0]], [paddle_best_pos[1]], [paddle_best_pos[2]],
                 s=80, c=NATURE_PALETTE["blue"],
                 edgecolor=NATURE_PALETTE["axis"], linewidths=0.6, marker="^",
                 label="paddle argmax")
    bb = skin.vertices
    ax_a.set_xlim(bb[:, 0].min(), bb[:, 0].max())
    ax_a.set_ylim(bb[:, 1].min(), bb[:, 1].max())
    ax_a.set_zlim(bb[:, 2].min(), bb[:, 2].max())
    try:
        ax_a.set_box_aspect((bb[:, 0].ptp(), bb[:, 1].ptp(), bb[:, 2].ptp()))
    except AttributeError:
        pass
    ax_a.view_init(elev=8, azim=-70)
    ax_a.set_xlabel("X (mm)", labelpad=-3)
    ax_a.set_ylabel("Y (mm)", labelpad=-3)
    ax_a.set_zlabel("Z (mm)", labelpad=-3)
    ax_a.set_title("Whole-body |L_EEG| on skin")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb_a = fig.colorbar(sm, ax=ax_a, shrink=0.55, pad=0.05, fraction=0.04)
    cb_a.set_label("|L|  ·  µV  (1 nA·m source)", fontsize=8)
    cb_a.outline.set_visible(False)
    ax_a.legend(loc="upper left", fontsize=7, handlelength=1.0)
    add_panel_label(ax_a, "a")

    # ── panel b: unrolled cylinder ─────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    skin_centroid = skin.centroid
    dx = wb_sensors.coilpos[:, 0] - skin_centroid[0]
    dy = wb_sensors.coilpos[:, 1] - skin_centroid[1]
    theta = np.degrees(np.arctan2(dy, dx))
    z_wb = wb_sensors.coilpos[:, 2]
    sc_b = ax_b.scatter(theta, z_wb, c=wb_val, cmap=cmap, s=14, vmin=0, vmax=vmax,
                         edgecolor="none", alpha=0.9)
    src_dx = src[0] - skin_centroid[0]
    src_dy = src[1] - skin_centroid[1]
    src_theta = float(np.degrees(np.arctan2(src_dy, src_dx)))
    ax_b.scatter([src_theta], [src[2]], s=200, c=NATURE_PALETTE["glow"],
                 edgecolor=NATURE_PALETTE["axis"], linewidths=0.8, marker="*",
                 zorder=5)
    ax_b.set_xlabel("Azimuth θ around body axis (°)")
    ax_b.set_ylabel("Z position (mm)")
    ax_b.set_xlim(-180, 180)
    ax_b.set_xticks(np.arange(-180, 181, 60))
    ax_b.set_title("Whole-body |L_EEG|  ·  unrolled cylinder")
    cb_b = fig.colorbar(sc_b, ax=ax_b, shrink=0.85, fraction=0.04, pad=0.02)
    cb_b.set_label("|L|  ·  µV  (1 nA·m source)", fontsize=8)
    cb_b.outline.set_visible(False)
    add_panel_label(ax_b, "b")

    # ── panel c: per-source SNR comparison ─────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    floors = compute_noise_floors(cfg)
    sigma = floors.eeg_per_channel_uV
    Q_ref = 70.0
    snr_paddle = paddle_peak * Q_ref / sigma
    snr_wb = wb_peak * Q_ref / sigma
    ax_c.plot(z, snr_paddle, color=NATURE_PALETTE["blue"], lw=1.6,
              label=f"32-ch cervical paddle  (peak {paddle_peak.max():.2e} µV/nAm)")
    ax_c.plot(z, snr_wb, color=NATURE_PALETTE["red"], lw=1.6,
              label=f"{len(wb_sensors.coilpos)}-ch whole-body  (peak {wb_peak.max():.2e} µV/nAm)")
    ax_c.axhline(3.0, color=NATURE_PALETTE["axis"], lw=0.8, linestyle="--",
                 label="SNR = 3")
    ax_c.set_xlabel(f"Source z along {region} (mm)")
    ax_c.set_ylabel(f"Single-trial SNR  ·  Q = {Q_ref:g} nA·m, σ = {sigma:.1f} µV")
    ax_c.set_yscale("log")
    ax_c.set_title("Single-trial SNR  ·  paddle vs whole-body")
    ax_c.legend(loc="lower center", fontsize=7, handlelength=1.4)
    add_panel_label(ax_c, "c")

    # ── panel d: per-source peak |L| comparison ────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.plot(z, paddle_peak, color=NATURE_PALETTE["blue"], lw=1.6,
              label="32-ch cervical paddle")
    ax_d.plot(z, wb_peak, color=NATURE_PALETTE["red"], lw=1.6,
              label=f"{len(wb_sensors.coilpos)}-ch whole-body")
    ax_d.fill_between(z, paddle_peak, wb_peak, where=(wb_peak > paddle_peak),
                       color=NATURE_PALETTE["stone"], alpha=0.45,
                       label="Whole-body advantage")
    ax_d.set_xlabel(f"Source z along {region} (mm)")
    ax_d.set_ylabel("Best-channel |L|  ·  µV  (1 nA·m source)")
    ax_d.set_yscale("log")
    ax_d.set_title(f"Best-channel |L| along the {region}")
    ax_d.legend(loc="lower center", fontsize=7, handlelength=1.4)
    add_panel_label(ax_d, "d")

    # ── panel e: amplitude vs distance ─────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    dist = np.linalg.norm(wb_sensors.coilpos - src[None, :], axis=1)
    ax_e.scatter(dist, wb_val, c=wb_val, cmap=cmap, s=18, alpha=0.7,
                 edgecolor="none", vmin=0, vmax=vmax)
    r_grid = np.linspace(dist.min(), dist.max(), 200)
    norm_factor = float(np.percentile(wb_val, 95)) * float(np.median(dist)) ** 2
    ax_e.plot(r_grid, norm_factor / r_grid ** 2, color=NATURE_PALETTE["axis"],
              lw=0.8, linestyle="--", label="∝ 1/r²")
    ax_e.set_xlabel("Electrode–source distance (mm)")
    ax_e.set_ylabel("|L_EEG|  ·  µV  (1 nA·m source)")
    ax_e.set_yscale("log")
    ax_e.set_title("Amplitude vs distance  (whole-body)")
    ax_e.legend(loc="upper right", handlelength=1.4)
    add_panel_label(ax_e, "e")

    # ── panel f: headline numbers ──────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    ratio = wb_best_val / max(paddle_best_val, 1e-30)
    paddle_argmax_dist = float(np.linalg.norm(paddle_best_pos - src))
    wb_argmax_dist = float(np.linalg.norm(wb_best_pos - src))
    snr_paddle_now = paddle_best_val * Q_ref / sigma
    snr_wb_now = wb_best_val * Q_ref / sigma
    trials_paddle = (3.0 * sigma / max(paddle_best_val * Q_ref, 1e-30)) ** 2
    trials_wb = (3.0 * sigma / max(wb_best_val * Q_ref, 1e-30)) ** 2
    headline = (
        "Question: is the cervical paddle in the wrong place,\n"
        "or is EEG fundamentally limited by bone shielding?\n\n"
        f"Source: vagus_left #{source_idx}  z = {src[2]:.0f} mm\n"
        f"Q = {Q_ref:g} nA·m, σ_EEG = {sigma:.1f} µV (1 kHz BW)\n\n"
        f"Cervical paddle (32 ch):\n"
        f"   peak |L|  = {paddle_best_val:.3e} µV / nA·m\n"
        f"   argmax–source dist = {paddle_argmax_dist:.0f} mm\n"
        f"   single-trial SNR  = {snr_paddle_now:.2e}\n"
        f"   trials → SNR=3    = {trials_paddle:.2e}\n\n"
        f"Whole-body ({len(wb_sensors.coilpos)} ch):\n"
        f"   peak |L|  = {wb_best_val:.3e} µV / nA·m\n"
        f"   argmax–source dist = {wb_argmax_dist:.0f} mm\n"
        f"   single-trial SNR  = {snr_wb_now:.2e}\n"
        f"   trials → SNR=3    = {trials_wb:.2e}\n\n"
        f"Whole-body / paddle peak ratio: {ratio:.2f}×\n"
    )
    if ratio < 1.5:
        verdict = (
            "Verdict: paddle is at the optimal location.\n"
            "Moving electrodes does NOT solve the EEG problem —\n"
            "bone shielding is the dominant attenuator."
        )
    elif ratio < 5.0:
        verdict = (
            "Verdict: location matters somewhat — whole-body\n"
            "is "
            f"{ratio:.1f}× better. But still ≪ MEG sensitivity."
        )
    else:
        verdict = (
            "Verdict: paddle is poorly sited — whole-body sees\n"
            f"{ratio:.1f}× more signal. Worth re-siting hardware."
        )
    ax_f.text(0.02, 0.98, headline, transform=ax_f.transAxes,
              va="top", ha="left", fontsize=9,
              fontfamily="monospace", color=NATURE_PALETTE["axis"])
    ax_f.text(0.02, 0.20, verdict, transform=ax_f.transAxes,
              va="top", ha="left", fontsize=10, fontweight="bold",
              color=NATURE_PALETTE["red"] if ratio >= 1.5 else NATURE_PALETTE["axis"])
    add_panel_label(ax_f, "f")

    fig.suptitle(
        f"Is location the problem?  Whole-body EEG vs cervical paddle  "
        f"·  source z = {src[2]:.0f} mm",
        fontsize=12, fontweight="bold", y=0.985,
    )

    logger.info("location-optimisation paddle peak %.3e  whole-body peak %.3e  ratio %.2f×",
                paddle_best_val, wb_best_val, ratio)
    return save_figure(fig, out_path or target_output(cfg, "location_optimisation.png"), dpi=dpi)
