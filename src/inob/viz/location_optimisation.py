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


def _bootstrap_median_ratio_ci(
    ratio_all: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI on the median whole-body/paddle ratio across sources.

    Same non-parametric resampling-with-replacement approach as
    ``sarvas_compare._bootstrap_ratio_ci`` (resample the per-source ratios,
    take the median each draw). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(int(seed))
    n = ratio_all.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    medians = np.median(ratio_all[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(medians, 100 * alpha))
    hi = float(np.percentile(medians, 100 * (1.0 - alpha)))
    med = float(np.median(medians))
    return lo, med, hi


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
        # Nearest source to the paddle, not the source list's midpoint — on an
        # elongated target (spine) the midpoint sits far from the paddle,
        # which previously made the propagation correction below look like it
        # *amplifies* the signal (an artefact of the mismatched reference; see
        # inob.viz.detectability.default_source_idx, which this mirrors).
        from inob.viz.detectability import default_source_idx

        source_idx = default_source_idx(paddle_lf, paddle_lf)
    src = paddle_lf.source_pos[source_idx]

    # ── per-source data ───────────────────────────────────────────────────
    paddle_peak = _per_source_peak(paddle_lf.L_fT_per_nAm)  # (S,) µV/nAm
    wb_peak = _per_source_peak(wb_lf.L_fT_per_nAm)  # (S,) µV/nAm
    z = paddle_lf.source_pos[:, 2]

    # Whole-body single-source: orientation-agnostic L2 norm across moments
    wb_val = _column_for_source_norm(wb_lf.L_fT_per_nAm, source_idx)  # (1000,)
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
        wb_sensors.coilpos,
        wb_val,
        skin.vertices,
        sigma_mm=60.0,
        k_nearest=24,
    )

    # ── figure ────────────────────────────────────────────────────────────
    cmap = sequential_cmap()
    vmax = float(wb_val.max() * 1.05)

    fig = plt.figure(figsize=(15, 11.5))
    gs = GridSpec(
        2, 3, figure=fig, left=0.04, right=0.97, top=0.92, bottom=0.07, hspace=0.34, wspace=0.24
    )

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
        skin.vertices[faces],
        facecolor=cmap(norm),
        edgecolor="none",
        alpha=0.95,
    )
    ax_a.add_collection3d(coll)
    ax_a.scatter(
        [src[0]],
        [src[1]],
        [src[2]],
        s=200,
        c=NATURE_PALETTE["glow"],
        edgecolor=NATURE_PALETTE["axis"],
        linewidths=0.8,
        marker="*",
    )
    ax_a.scatter(
        [wb_best_pos[0]],
        [wb_best_pos[1]],
        [wb_best_pos[2]],
        s=120,
        c=NATURE_PALETTE["red"],
        edgecolor=NATURE_PALETTE["axis"],
        linewidths=0.6,
        marker="o",
        label="whole-body argmax",
    )
    ax_a.scatter(
        [paddle_best_pos[0]],
        [paddle_best_pos[1]],
        [paddle_best_pos[2]],
        s=80,
        c=NATURE_PALETTE["blue"],
        edgecolor=NATURE_PALETTE["axis"],
        linewidths=0.6,
        marker="^",
        label="paddle argmax",
    )
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
    sc_b = ax_b.scatter(
        theta, z_wb, c=wb_val, cmap=cmap, s=14, vmin=0, vmax=vmax, edgecolor="none", alpha=0.9
    )
    src_dx = src[0] - skin_centroid[0]
    src_dy = src[1] - skin_centroid[1]
    src_theta = float(np.degrees(np.arctan2(src_dy, src_dx)))
    ax_b.scatter(
        [src_theta],
        [src[2]],
        s=200,
        c=NATURE_PALETTE["glow"],
        edgecolor=NATURE_PALETTE["axis"],
        linewidths=0.8,
        marker="*",
        zorder=5,
    )
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
    from inob.config import source_target_tag
    from inob.physiology.profiles import profile_for_tag

    profile = profile_for_tag(source_target_tag(cfg))
    Q_ref = profile.default_strength_nAm

    # Propagating-volley correction: `peak |L| x Q` alone is the stationary
    # model (all moment lumped at one point), an upper bound the same way the
    # synchronous whole-cord model is. Where the profile says lumping is not
    # defensible (spine: stationary_ok=False) and the source list is an
    # ordered path (not a volume fill), apply the same event-level scalar
    # correction inob.viz.detectability / inob.analysis.source_models use, so
    # this figure's trial counts do not silently assume the naive model.
    from inob.analysis.propagation import (
        compute_propagation_signals,
        is_ordered_polyline,
        peak_over_channels,
        propagation_ratio,
    )

    paddle_prop_factor = 1.0
    wb_prop_factor = 1.0
    if not profile.stationary_ok and is_ordered_polyline(paddle_lf.source_pos):
        paddle_prop_factor = propagation_ratio(
            compute_propagation_signals(paddle_lf, profile, stationary_idx=source_idx),
            peak_over_channels,
        )
        wb_prop_factor = propagation_ratio(
            compute_propagation_signals(wb_lf, profile, stationary_idx=source_idx),
            peak_over_channels,
        )
        logger.info(
            "location-optimisation propagation correction (%s): paddle x%.3f, whole-body x%.3f",
            profile.name,
            paddle_prop_factor,
            wb_prop_factor,
        )

    snr_paddle = paddle_peak * Q_ref * paddle_prop_factor / sigma
    snr_wb = wb_peak * Q_ref * wb_prop_factor / sigma
    ax_c.plot(
        z,
        snr_paddle,
        color=NATURE_PALETTE["blue"],
        lw=1.6,
        label=f"32-ch cervical paddle  (peak {paddle_peak.max():.2e} µV/nAm)",
    )
    ax_c.plot(
        z,
        snr_wb,
        color=NATURE_PALETTE["red"],
        lw=1.6,
        label=f"{len(wb_sensors.coilpos)}-ch whole-body  (peak {wb_peak.max():.2e} µV/nAm)",
    )
    ax_c.axhline(3.0, color=NATURE_PALETTE["axis"], lw=0.8, linestyle="--", label="SNR = 3")
    ax_c.set_xlabel(f"Source z along {region} (mm)")
    ax_c.set_ylabel(f"Single-trial SNR  ·  Q = {Q_ref:g} nA·m, σ = {sigma:.1f} µV")
    ax_c.set_yscale("log")
    prop_note = (
        f"  ·  propagating volley applied (paddle ×{paddle_prop_factor:.2f}, "
        f"whole-body ×{wb_prop_factor:.2f})"
        if paddle_prop_factor != 1.0
        else ""
    )
    ax_c.set_title("Single-trial SNR  ·  paddle vs whole-body" + prop_note, fontsize=9)
    ax_c.legend(loc="lower center", fontsize=7, handlelength=1.4)
    add_panel_label(ax_c, "c")

    # ── panel d: per-source peak |L| comparison ────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.plot(z, paddle_peak, color=NATURE_PALETTE["blue"], lw=1.6, label="32-ch cervical paddle")
    ax_d.plot(
        z,
        wb_peak,
        color=NATURE_PALETTE["red"],
        lw=1.6,
        label=f"{len(wb_sensors.coilpos)}-ch whole-body",
    )
    ax_d.fill_between(
        z,
        paddle_peak,
        wb_peak,
        where=(wb_peak > paddle_peak),
        color=NATURE_PALETTE["stone"],
        alpha=0.45,
        label="Whole-body advantage",
    )
    ax_d.set_xlabel(f"Source z along {region} (mm)")
    ax_d.set_ylabel("Best-channel |L|  ·  µV  (1 nA·m source)")
    ax_d.set_yscale("log")
    ax_d.set_title(f"Best-channel |L| along the {region}")
    ax_d.legend(loc="lower center", fontsize=7, handlelength=1.4)
    add_panel_label(ax_d, "d")

    # ── panel e: amplitude vs distance ─────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    dist = np.linalg.norm(wb_sensors.coilpos - src[None, :], axis=1)
    ax_e.scatter(
        dist, wb_val, c=wb_val, cmap=cmap, s=18, alpha=0.7, edgecolor="none", vmin=0, vmax=vmax
    )
    r_grid = np.linspace(dist.min(), dist.max(), 200)
    norm_factor = float(np.percentile(wb_val, 95)) * float(np.median(dist)) ** 2
    ax_e.plot(
        r_grid,
        norm_factor / r_grid**2,
        color=NATURE_PALETTE["axis"],
        lw=0.8,
        linestyle="--",
        label="∝ 1/r²",
    )
    ax_e.set_xlabel("Electrode–source distance (mm)")
    ax_e.set_ylabel("|L_EEG|  ·  µV  (1 nA·m source)")
    ax_e.set_yscale("log")
    ax_e.set_title("Amplitude vs distance  (whole-body)")
    ax_e.legend(loc="upper right", handlelength=1.4)
    add_panel_label(ax_e, "e")

    # ── panel f: headline numbers ──────────────────────────────────────────
    # Two verdicts, not one: the highlighted source alone can be misleading
    # when the source region is longer than the paddle footprint (e.g. a
    # paddle centred on one vertebral level looks "optimal" only for sources
    # near that level) — so report the single-source number *and* the
    # full-region sweep (panels c/d already compute per-source peaks for
    # every source; this just summarises that array instead of discarding it).
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    ratio = wb_best_val / max(paddle_best_val, 1e-30)
    paddle_argmax_dist = float(np.linalg.norm(paddle_best_pos - src))
    wb_argmax_dist = float(np.linalg.norm(wb_best_pos - src))
    snr_paddle_now = paddle_best_val * Q_ref * paddle_prop_factor / sigma
    snr_wb_now = wb_best_val * Q_ref * wb_prop_factor / sigma
    trials_paddle = (3.0 * sigma / max(paddle_best_val * Q_ref * paddle_prop_factor, 1e-30)) ** 2
    trials_wb = (3.0 * sigma / max(wb_best_val * Q_ref * wb_prop_factor, 1e-30)) ** 2

    ratio_all = wb_peak / np.maximum(paddle_peak, 1e-30)
    ratio_median = float(np.median(ratio_all))
    ratio_min = float(ratio_all.min())
    ratio_max = float(ratio_all.max())
    frac_paddle_ok = float(np.mean(ratio_all < 1.5))
    ratio_ci_lo, _ratio_ci_med, ratio_ci_hi = _bootstrap_median_ratio_ci(
        ratio_all,
        seed=int(cfg.reproducibility.seed),
    )

    headline = (
        "Question: is the cervical paddle in the wrong place,\n"
        "or is EEG fundamentally limited by bone shielding?\n\n"
        f"At highlighted source:  {region} #{source_idx}  z = {src[2]:.0f} mm\n"
        f"Q = {Q_ref:g} nA·m, σ_EEG = {sigma:.1f} µV (1 kHz BW)\n"
        + (
            f"Propagating-volley correction: paddle ×{paddle_prop_factor:.2f}, "
            f"whole-body ×{wb_prop_factor:.2f}\n\n"
            if paddle_prop_factor != 1.0
            else "\n"
        )
        + f"Cervical paddle (32 ch):\n"
        f"   peak |L|  = {paddle_best_val:.3e} µV / nA·m\n"
        f"   argmax–source dist = {paddle_argmax_dist:.0f} mm\n"
        f"   single-trial SNR  = {snr_paddle_now:.2e}\n"
        f"   trials → SNR=3    = {trials_paddle:.2e}\n\n"
        f"Whole-body ({len(wb_sensors.coilpos)} ch):\n"
        f"   peak |L|  = {wb_best_val:.3e} µV / nA·m\n"
        f"   argmax–source dist = {wb_argmax_dist:.0f} mm\n"
        f"   single-trial SNR  = {snr_wb_now:.2e}\n"
        f"   trials → SNR=3    = {trials_wb:.2e}\n\n"
        f"Whole-body / paddle ratio here: {ratio:.2f}×\n\n"
        f"Across full {region} (N={len(z)}, z={z.min():.0f}–{z.max():.0f} mm):\n"
        f"   ratio  median={ratio_median:.1f}×  "
        f"range={ratio_min:.2f}–{ratio_max:.1f}×\n"
        f"   median ratio 95% CI = {ratio_ci_lo:.1f}–{ratio_ci_hi:.1f}×  "
        f"(B=2000 bootstrap)\n"
        f"   paddle within 1.5× of whole-body at "
        f"{100 * frac_paddle_ok:.0f}% of sources\n"
    )
    if ratio_median < 1.5:
        verdict = (
            "Verdict: paddle tracks whole-body across the region —\n"
            "moving electrodes does NOT solve the EEG problem;\n"
            "bone shielding is the dominant attenuator."
        )
    elif frac_paddle_ok > 0.5:
        verdict = (
            "Verdict: paddle is optimal near its own level, but the\n"
            "region extends beyond the paddle footprint — location\n"
            f"matters away from it (up to {ratio_max:.0f}× at the far end)."
        )
    else:
        verdict = (
            "Verdict: paddle is a single-level snapshot, not a fair\n"
            f"stand-in for the whole {region} — whole-body sees\n"
            f"{ratio_median:.0f}× more signal at a typical source.\n"
            "A fixed cervical paddle under-covers this region."
        )
    ax_f.text(
        0.02,
        0.98,
        headline,
        transform=ax_f.transAxes,
        va="top",
        ha="left",
        fontsize=8.3,
        fontfamily="monospace",
        color=NATURE_PALETTE["axis"],
    )
    ax_f.text(
        0.02,
        0.14,
        verdict,
        transform=ax_f.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=NATURE_PALETTE["red"] if ratio_median >= 1.5 else NATURE_PALETTE["axis"],
    )
    add_panel_label(ax_f, "f")

    fig.suptitle(
        f"Is location the problem?  Whole-body EEG vs cervical paddle  "
        f"·  highlighted source z = {src[2]:.0f} mm",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )

    logger.info(
        "location-optimisation @source: paddle peak %.3e  whole-body peak %.3e  ratio %.2f×  "
        "| full-region ratio median=%.2f (95%% CI %.2f-%.2f) min=%.2f max=%.2f",
        paddle_best_val,
        wb_best_val,
        ratio,
        ratio_median,
        ratio_ci_lo,
        ratio_ci_hi,
        ratio_min,
        ratio_max,
    )
    return save_figure(fig, out_path or target_output(cfg, "location_optimisation.png"), dpi=dpi)
