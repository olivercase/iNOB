"""Skin-surface interpolated topoplots (MEG + EEG).

Renders a body topo as a colour map *on the actual skin surface* — not just
sensor dots — for a single dipole source on the cervical vagus. The
interpolation is a Gaussian-weighted average over the K nearest sensors,
which is robust without picking arbitrary spline parameters.

Two views per modality:

  * 3-D body silhouette with the cervical region magnified.
  * 2-D unrolled cylinder (θ vs Z) — the canonical "field map" you'd see
    in an OPM or HD-EMG paper.

Designed to Nature Reviews standards via :mod:`inob.viz.style`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from inob.config import Config, source_region_label, target_output
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

logger = logging.getLogger(__name__)


# ── interpolation ──────────────────────────────────────────────────────────

def gaussian_interpolate_surface(
    sensor_pos_mm: np.ndarray,
    sensor_val: np.ndarray,
    target_pts_mm: np.ndarray,
    *, sigma_mm: float, k_nearest: int = 16,
) -> np.ndarray:
    """Gaussian-weighted KNN interpolation of sensor values onto target points.

    Parameters
    ----------
    sensor_pos_mm   (N, 3) sensor positions
    sensor_val      (N,)   scalar values at each sensor
    target_pts_mm   (M, 3) where to interpolate
    sigma_mm        Gaussian kernel width (use 1–2× sensor pitch)
    k_nearest       only the K nearest sensors contribute to each target

    Returns
    -------
    (M,) interpolated values.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(sensor_pos_mm)
    k = min(k_nearest, len(sensor_pos_mm))
    dists, idx = tree.query(target_pts_mm, k=k)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]
    weights = np.exp(-0.5 * (dists / sigma_mm) ** 2)
    w_sum = weights.sum(axis=1, keepdims=True)
    w_sum = np.maximum(w_sum, 1e-30)
    weighted = (sensor_val[idx] * weights).sum(axis=1, keepdims=True) / w_sum
    return weighted.ravel()


# ── rendering helpers ──────────────────────────────────────────────────────

def _draw_skin_coloured(
    ax, vertices: np.ndarray, faces: np.ndarray, vertex_vals: np.ndarray,
    *, vmin: float, vmax: float, cmap=None, alpha: float = 0.95,
    max_tris: int = 50_000, rng: np.random.Generator | None = None,
) -> None:
    """Render a coloured skin surface on a 3-D matplotlib axis."""
    if rng is None:
        rng = np.random.default_rng(0)
    if cmap is None:
        cmap = divergent_cmap()
    if len(faces) > max_tris:
        idx = rng.choice(len(faces), max_tris, replace=False)
        faces = faces[idx]
    # Per-face colour = mean of its vertex values
    face_vals = vertex_vals[faces].mean(axis=1)
    norm = (face_vals - vmin) / max(vmax - vmin, 1e-30)
    face_colors = cmap(np.clip(norm, 0.0, 1.0))
    coll = Poly3DCollection(
        vertices[faces], facecolor=face_colors, edgecolor="none", alpha=alpha,
    )
    ax.add_collection3d(coll)


def _crop_skin_to_band(
    skin: trimesh.Trimesh, z_lo: float, z_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a skin mesh to a Z slab, returning (vertices, faces) of the slab."""
    mask = (skin.vertices[:, 2] >= z_lo) & (skin.vertices[:, 2] <= z_hi)
    keep = np.where(mask)[0]
    remap = -np.ones(len(skin.vertices), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    face_keep = mask[skin.faces].all(axis=1)
    new_faces = remap[skin.faces[face_keep]]
    return np.asarray(skin.vertices[keep]), np.asarray(new_faces)


def _cylindrical_unroll(
    pos_mm: np.ndarray, *, axis_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Cylindrical unroll (θ in degrees, z in mm) about a given axis."""
    dx = pos_mm[:, 0] - axis_xy[0]
    dy = pos_mm[:, 1] - axis_xy[1]
    theta_deg = np.degrees(np.arctan2(dy, dx))
    return theta_deg, pos_mm[:, 2]


# ── main ───────────────────────────────────────────────────────────────────

def render_surface_topoplots(
    cfg: Config, *, source_idx: int = -1, out_path: Path | None = None,
    dpi: int = 300, z_band_mm: float = 220.0,
) -> Path:
    """Six-panel surface-topoplot figure (3 MEG + 3 EEG).

    For one dipole source on the cervical vagus, render the predicted field
    interpolated onto the actual skin surface (and the EEG patch) — the
    "what would the body see" view.
    """
    apply_nature_style()

    meg_lf = load_leadfield(cfg.outputs.forward_npz)
    eeg_lf = load_leadfield(cfg.outputs.forward_eeg_npz)
    geom = load_geometry(cfg.outputs.geometry_mat)
    skin_comp = geom.compartments["mesh_skin"]
    skin = trimesh.Trimesh(skin_comp.vertices, skin_comp.faces, process=False)
    electrodes = load_sensors(cfg.outputs.electrodes_mat)

    if source_idx < 0:
        source_idx = meg_lf.source_pos.shape[0] // 2
    src = meg_lf.source_pos[source_idx]

    # ── MEG: longitudinal-moment leadfield at radial OPMs ──────────────────
    n_meg = meg_lf.coil_pos.shape[0]
    radial_idx = np.arange(n_meg // 3)        # first third = R coils
    meg_pos = meg_lf.coil_pos[radial_idx]
    meg_val_full = meg_lf.L_fT_per_nAm[:, 3 * source_idx + 2]    # z-moment
    # Each radial coil has an outward-skin orientation; project the field by
    # picking the radial channel directly (already projected during forward).
    meg_val = meg_val_full[radial_idx]

    # ── EEG: longitudinal-moment leadfield at the patch ────────────────────
    eeg_val = eeg_lf.L_fT_per_nAm[:, 3 * source_idx + 2]

    # ── Crop skin to the cervical band for the headline view ───────────────
    z_lo, z_hi = src[2] - z_band_mm, src[2] + z_band_mm
    band_v, band_f = _crop_skin_to_band(skin, z_lo, z_hi)
    if len(band_v) == 0:
        raise RuntimeError(f"no skin vertices in band Z=[{z_lo:.0f}, {z_hi:.0f}]")

    # Interpolate MEG sensor values onto the cropped skin (Gaussian KNN).
    meg_skin_vals = gaussian_interpolate_surface(
        meg_pos, meg_val, band_v, sigma_mm=40.0, k_nearest=24,
    )

    # ── figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11.5))
    gs = GridSpec(
        2, 3, figure=fig,
        left=0.04, right=0.97, top=0.93, bottom=0.06,
        hspace=0.32, wspace=0.20,
    )

    cmap = divergent_cmap()
    vmin_meg, vmax_meg = divergent_norm(meg_val)

    # ── panel a: MEG body-surface view (3-D) ───────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    _draw_skin_coloured(
        ax_a, band_v, band_f, meg_skin_vals,
        vmin=vmin_meg, vmax=vmax_meg, cmap=cmap, alpha=0.95,
    )
    ax_a.scatter([src[0]], [src[1]], [src[2]],
                 s=200, c=NATURE_PALETTE["glow"],
                 edgecolor=NATURE_PALETTE["axis"], linewidths=0.8, marker="*")
    ax_a.set_xlim(band_v[:, 0].min() - 30, band_v[:, 0].max() + 30)
    ax_a.set_ylim(band_v[:, 1].min() - 30, band_v[:, 1].max() + 30)
    ax_a.set_zlim(z_lo, z_hi)
    ax_a.view_init(elev=12, azim=42)
    ax_a.set_xlabel("X (mm)", labelpad=-3)
    ax_a.set_ylabel("Y (mm)", labelpad=-3)
    ax_a.set_zlabel("Z (mm)", labelpad=-3)
    ax_a.set_title("MEG  ·  field on body surface")
    sm_meg = plt.cm.ScalarMappable(cmap=cmap,
                                    norm=plt.Normalize(vmin=vmin_meg, vmax=vmax_meg))
    cb_a = fig.colorbar(sm_meg, ax=ax_a, shrink=0.55, pad=0.05, fraction=0.04)
    cb_a.set_label("fT  (1 nA·m source)", fontsize=8)
    cb_a.outline.set_visible(False)
    add_panel_label(ax_a, "a")

    # ── panel b: MEG cylindrical unroll (θ vs Z) ───────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    axis_xy = (band_v[:, 0].mean(), band_v[:, 1].mean())
    theta_meg, z_meg = _cylindrical_unroll(meg_pos, axis_xy=axis_xy)
    in_band = (z_meg >= z_lo) & (z_meg <= z_hi)
    sc_b = ax_b.scatter(
        theta_meg[in_band], z_meg[in_band],
        c=meg_val[in_band], cmap=cmap, vmin=vmin_meg, vmax=vmax_meg,
        s=22, edgecolor=NATURE_PALETTE["axis"], linewidths=0.2,
    )
    src_theta, src_z = _cylindrical_unroll(src[None, :], axis_xy=axis_xy)
    ax_b.scatter(src_theta, src_z, s=200, c=NATURE_PALETTE["glow"],
                 edgecolor=NATURE_PALETTE["axis"], linewidths=0.8, marker="*")
    ax_b.set_xlabel("Azimuth θ around cervical axis (°)")
    ax_b.set_ylabel("Z position (mm)")
    ax_b.set_xlim(-180, 180)
    ax_b.set_xticks(np.arange(-180, 181, 60))
    ax_b.set_title("MEG  ·  unrolled cylinder")
    cb_b = fig.colorbar(sc_b, ax=ax_b, shrink=0.85, fraction=0.04, pad=0.02)
    cb_b.set_label("fT  (1 nA·m source)", fontsize=8)
    cb_b.outline.set_visible(False)
    add_panel_label(ax_b, "b")

    # ── panel c: MEG amplitude vs distance ─────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    dist_meg = np.linalg.norm(meg_pos - src[None, :], axis=1)
    ax_c.scatter(dist_meg, np.abs(meg_val),
                 c=np.abs(meg_val), cmap="magma", s=14, alpha=0.7,
                 edgecolor="none")
    # Theoretical 1/r² fall-off for visual reference (normalised to median).
    r_grid = np.linspace(50, dist_meg.max(), 200)
    norm_factor = float(np.percentile(np.abs(meg_val), 95)) * float(np.median(dist_meg)) ** 2
    ax_c.plot(r_grid, norm_factor / r_grid ** 2, color=NATURE_PALETTE["red"],
              lw=1.0, linestyle="--", label="∝ 1/r²")
    ax_c.set_xlabel("Sensor–source distance (mm)")
    ax_c.set_ylabel("|MEG L|  ·  fT  (1 nA·m source)")
    ax_c.set_yscale("log")
    ax_c.set_title("MEG amplitude vs distance")
    ax_c.legend(loc="upper right", handlelength=1.4)
    add_panel_label(ax_c, "c")

    # ── panel d: EEG patch heatmap (3-D placement) ─────────────────────────
    ax_d = fig.add_subplot(gs[1, 0], projection="3d")
    # Neck-band skin context so the patch is visibly located on the body — the
    # alpha stays low enough not to occlude the (near-side) contacts but high
    # enough to read as a body silhouette rather than the previous ghost.
    band_alpha = 0.12
    coll_skin = Poly3DCollection(
        band_v[band_f], facecolor=NATURE_PALETTE["skin"],
        edgecolor="none", alpha=band_alpha,
    )
    ax_d.add_collection3d(coll_skin)
    vmin_eeg, vmax_eeg = divergent_norm(eeg_val)
    sc_d = ax_d.scatter(
        electrodes.coilpos[:, 0], electrodes.coilpos[:, 1], electrodes.coilpos[:, 2],
        c=eeg_val, cmap=cmap, vmin=vmin_eeg, vmax=vmax_eeg,
        s=80, edgecolor=NATURE_PALETTE["axis"], linewidths=0.4, depthshade=False,
        zorder=5,
    )
    ax_d.scatter([src[0]], [src[1]], [src[2]],
                 s=200, c=NATURE_PALETTE["glow"],
                 edgecolor=NATURE_PALETTE["axis"], linewidths=0.8, marker="*")
    # Frame the whole cervical band (not just the patch bbox) so you can see
    # where on the neck the ~30 mm patch actually sits.
    ax_d.set_xlim(band_v[:, 0].min() - 15, band_v[:, 0].max() + 15)
    ax_d.set_ylim(band_v[:, 1].min() - 15, band_v[:, 1].max() + 15)
    ax_d.set_zlim(z_lo, z_hi)
    ax_d.view_init(elev=12, azim=42)
    ax_d.set_xlabel("X (mm)", labelpad=-3)
    ax_d.set_ylabel("Y (mm)", labelpad=-3)
    ax_d.set_zlabel("Z (mm)", labelpad=-3)
    ax_d.set_title("EEG  ·  HD-EMG patch on skin")
    cb_d = fig.colorbar(sc_d, ax=ax_d, shrink=0.55, pad=0.05, fraction=0.04)
    cb_d.set_label("µV  (1 nA·m source)", fontsize=8)
    cb_d.outline.set_visible(False)
    add_panel_label(ax_d, "d")

    # ── panel e: EEG patch heatmap (2-D paddle topology) ───────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    from inob.viz.topoplot import _draw_eeg_2d_topoplot
    _draw_eeg_2d_topoplot(ax_e, electrodes, eeg_val, cfg,
                          vmin=vmin_eeg, vmax=vmax_eeg)
    ax_e.set_title(f"EEG  ·  paddle topology  ·  {len(eeg_val)} contacts")
    add_panel_label(ax_e, "e")

    # ── panel f: EEG amplitude vs distance ─────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    dist_eeg = np.linalg.norm(electrodes.coilpos - src[None, :], axis=1)
    ax_f.scatter(dist_eeg, np.abs(eeg_val),
                 c=np.abs(eeg_val), cmap="magma", s=24, alpha=0.85,
                 edgecolor=NATURE_PALETTE["axis"], linewidths=0.2)
    r_grid = np.linspace(dist_eeg.min(), dist_eeg.max(), 200)
    norm_factor = float(np.percentile(np.abs(eeg_val), 95)) * float(np.median(dist_eeg)) ** 2
    ax_f.plot(r_grid, norm_factor / r_grid ** 2, color=NATURE_PALETTE["red"],
              lw=1.0, linestyle="--", label="∝ 1/r²")
    ax_f.set_xlabel("Contact–source distance (mm)")
    ax_f.set_ylabel("|EEG L|  ·  µV  (1 nA·m source)")
    ax_f.set_yscale("log")
    ax_f.set_title("EEG amplitude vs distance")
    ax_f.legend(loc="upper right", handlelength=1.4)
    add_panel_label(ax_f, "f")

    fig.suptitle(
        f"Surface topoplots — {source_region_label(cfg)} source #{source_idx} "
        f"@ z = {src[2]:.0f} mm  (longitudinal moment, 1 nA·m)",
        fontsize=12, fontweight="bold", y=0.985,
    )

    return save_figure(fig, out_path or target_output(cfg, "surface_topoplots.png"), dpi=dpi)
