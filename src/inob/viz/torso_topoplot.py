"""Four-panel field map painted on the body surface itself.

One figure per source target, laid out as modality by column so the two are
read against each other rather than in sequence:

    a  MEG  — the magnetic field on the skin of the whole upper body
    b  ESG  — the electric potential on the skin, where the patch sits
    c  MEG  — the same field with the skin unrolled to a flat θ–z map
    d  ESG  — the patch's own topography, in the patch's own frame

Left column magnetic, right column electric; top row on the body, bottom row
flattened. The coverage difference between the two modalities is the point of
the layout: a asks the whole torso, b can only ask a postage stamp of it.

The colour is a signed field with a neutral midpoint (blue negative, white
zero, red positive), so the two lobes of a dipolar pattern read immediately.
Two rules keep it honest:

  * **Nothing is painted where nothing was measured.** A vertex further than
    ``max_extrap_mm`` from the nearest sensor is drawn in neutral skin, not in
    an interpolated colour. Otherwise a Gaussian interpolation happily paints a
    confident-looking field across the whole torso from a neck array.
  * **The scale is symmetric and shared** between the front and back views, so
    the two panels can be compared. It is clipped to a high percentile rather
    than the raw maximum, because one sensor sitting a millimetre from the
    source otherwise sets the scale for the entire body.

Surfaces are Lambert-shaded so the torso reads as a body rather than a
silhouette; the shading multiplies the colour rather than replacing it, so it
never changes which side of zero a patch is on.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from inob.config import Config, source_region_label, target_output
from inob.io.hdf5 import load_geometry, load_sensors
from inob.io.npz import load_leadfield
from inob.viz.detectability import default_source_idx
from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
    divergent_cmap,
    save_figure,
)

logger = logging.getLogger(__name__)

#: Direction the scene is lit from (normalised in ``_shade``). Slightly above
#: and to the left of the camera, which is the convention that makes a convex
#: body read as convex.
_LIGHT_DIR = np.array([-0.4, -0.8, 0.45])

#: Ceiling on rendered triangles. Matplotlib's painter's algorithm is O(n log n)
#: per redraw and the whole skin mesh is 200k triangles, but the cropped band is
#: ~40k, which renders fine. Above the ceiling the mesh is simplified by quadric
#: decimation — NOT by dropping random faces, which punches holes straight
#: through the body and speckles the render.
_MAX_FACES = 120_000


# ── field on the surface ───────────────────────────────────────────────────

def _signed_radial_field(
    lf: Any, source_idx: int, moment: str = "z",
) -> tuple[np.ndarray, np.ndarray]:
    """Signed radial field per sensor *position*, with those positions.

    Two things this has to get right.

    Signed, not magnitude: a current dipole produces a field positive on one
    side and negative on the other, and |B| throws exactly that away — the
    figure would show one blob where there are two lobes.

    One component per position, not three: the OPM array is triaxial, so the
    leadfield's 4,740 channels are 1,580 positions x 3 axes and the first third
    are the radial component (the convention :mod:`inob.viz.sensor_field`
    documents). Interpolating all 4,740 puts three different values at each
    coordinate — which a distance-weighted scheme resolves as a nearest-position
    average, i.e. flat Voronoi tiles, and paints a smooth field as a mosaic.
    """
    k = {"x": 0, "y": 1, "z": 2}[moment]
    L = np.asarray(lf.L_fT_per_nAm, dtype=float)
    radial = np.arange(L.shape[0] // 3)
    return L[radial, 3 * source_idx + k], np.asarray(lf.coil_pos, float)[radial]


def _interpolate_to_surface(
    sensor_pos: np.ndarray, sensor_val: np.ndarray, verts: np.ndarray,
    *, sigma_mm: float, k_nearest: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-weighted KNN interpolation, plus the distance to the nearest sensor.

    Returns ``(values, nearest_mm)``. The second is what the caller uses to
    refuse to paint unmeasured skin.
    """
    tree = cKDTree(sensor_pos)
    k = min(k_nearest, len(sensor_pos))
    dist, idx = tree.query(verts, k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = np.exp(-0.5 * (dist / sigma_mm) ** 2)
    wsum = w.sum(axis=1)
    # A vertex with no sensor inside several sigma has no information; the
    # caller masks it, so the value it gets here is irrelevant but must be
    # finite.
    safe = np.where(wsum > 1e-12, wsum, 1.0)
    return (w * sensor_val[idx]).sum(axis=1) / safe, dist[:, 0]


def _crop_near(
    verts: np.ndarray, faces: np.ndarray, centre: np.ndarray, radius_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the cap of a mesh within ``radius_mm`` of a point, re-indexing faces.

    A Z-slab alone leaves a full ring of body around a patch that only sits on
    one side of the neck, which renders as a dome with the interesting part
    hidden on its far face.
    """
    keep = np.linalg.norm(verts - centre, axis=1) <= radius_mm
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    face_keep = keep[faces].all(axis=1)
    return verts[keep], remap[faces[face_keep]]


def _crop_to_band(
    verts: np.ndarray, faces: np.ndarray, z_lo: float, z_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the slab of a mesh between two Z planes, re-indexing its faces."""
    keep = (verts[:, 2] >= z_lo) & (verts[:, 2] <= z_hi)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    face_keep = keep[faces].all(axis=1)
    return verts[keep], remap[faces[face_keep]]


def _shade(verts: np.ndarray, faces: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    """Lambert-shade face colours so the surface reads as a solid body.

    Multiplies lightness only — a red face stays red, it just gets darker where
    it turns away from the light.
    """
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(norm > 0, norm, 1.0)
    light = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)
    lam = np.abs(n @ light)                       # abs: unknown winding
    shade = (0.62 + 0.38 * lam)[:, None]
    out = rgba.copy()
    out[:, :3] = np.clip(out[:, :3] * shade, 0.0, 1.0)
    return out


def _draw_surface(
    ax: Any, verts: np.ndarray, faces: np.ndarray, vals: np.ndarray,
    nearest_mm: np.ndarray, *, vlim: float, max_extrap_mm: float,
    cmap: Any, rng: np.random.Generator,
) -> None:
    """Paint the field on a mesh; leave unmeasured skin neutral."""
    face_val = vals[faces].mean(axis=1)
    face_near = nearest_mm[faces].min(axis=1)
    rgba = cmap(Normalize(-vlim, vlim)(face_val))
    # Neutral skin wherever the array had no sensor near enough to know.
    blind = face_near > max_extrap_mm
    rgba[blind] = plt.matplotlib.colors.to_rgba(NATURE_PALETTE["skin"])

    coll = Poly3DCollection(
        verts[faces], facecolors=_shade(verts, faces, rgba),
        edgecolors="none", linewidths=0, shade=False, zsort="average",
    )
    ax.add_collection3d(coll)


def _frame_3d(
    ax: Any, verts: np.ndarray, *, elev: float, azim: float,
    centre: np.ndarray | None = None, half_mm: float | None = None,
    zoom: float = 1.45,
) -> None:
    """A 3-D axis with nothing on it but the body: no panes, grid, or ticks.

    ``centre``/``half_mm`` zoom the view without cropping the mesh — which is
    how the close-up keeps the neck and shoulder in frame. Cropping to a short
    slab instead makes the neck read as a mound with its top sliced off.
    """
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    mid, span = (lo + hi) / 2, (hi - lo) * 0.54          # per-axis half-extent
    if centre is not None and half_mm is not None:
        mid, span = np.asarray(centre, float), np.full(3, float(half_mm))
    ax.set_xlim(mid[0] - span[0], mid[0] + span[0])
    ax.set_ylim(mid[1] - span[1], mid[1] + span[1])
    ax.set_zlim(mid[2] - span[2], mid[2] + span[2])
    # Box aspect from the real extents, not a cube: a cube around a torso that
    # is twice as tall as it is deep leaves most of the panel empty, and the
    # body ends up a small object in a large white square.
    # zoom > 1 because matplotlib's 3-D axes reserve a wide margin for the tick
    # labels and panes this figure has switched off; without it the body sits
    # small in a large white square.
    ax.set_box_aspect(tuple(span / span.max()), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    # Orthographic: a body is read by comparing sizes across the frame, and
    # perspective makes the near shoulder larger than the far one for no
    # informational gain. It also makes the scale bar exact.
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)


def _scale_bar(ax: Any, azim: float, mm: float = 50.0) -> None:
    """A bar of known length, since the axes are gone and size still matters.

    Laid along the *screen*-horizontal direction — which at azimuth θ is
    (−sin θ, cos θ, 0) — so under the orthographic projection set in
    :func:`_frame_3d` it is exactly ``mm`` long on the page. A bar along the x
    axis instead foreshortens by cos θ, and at the oblique angles these panels
    use that turned a 25 mm bar into a 6 mm tick.
    """
    (x0, x1), (y0, y1), (z0, z1) = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    th = np.radians(azim)
    dx, dy = -np.sin(th) * mm, np.cos(th) * mm
    bx = x0 + 0.10 * (x1 - x0)
    by = y0 + 0.10 * (y1 - y0)
    bz = z0 + 0.06 * (z1 - z0)
    ax.plot([bx, bx + dx], [by, by + dy], [bz, bz],
            color=NATURE_PALETTE["axis"], lw=1.6, solid_capstyle="butt")
    ax.text(bx + dx / 2, by + dy / 2, bz - 0.05 * (z1 - z0), f"{mm:.0f} mm",
            ha="center", va="top", fontsize=6.5, color=NATURE_PALETTE["axis"])


def _source_is_visible(
    pos: np.ndarray, axis_xy: np.ndarray, azim: float,
) -> bool:
    """Is the source on the side of the body facing the camera?

    Poly3DCollection has no depth buffer — it sorts whole triangles — so a
    marker inside the body is drawn *over* the skin regardless of what is in
    front of it. On the spine panels that put the source star on the model's
    face, 150 mm in front of a source in the vertebral canal. Rather than fake
    a depth test, the marker is simply not drawn when it would be behind the
    surface; the close-up panel is always oriented to show it.
    """
    view = np.array([np.cos(np.radians(azim)), np.sin(np.radians(azim))])
    outward = pos[:2] - axis_xy
    n = np.linalg.norm(outward)
    return bool(n < 1e-9 or (outward / n) @ view > 0.0)


def _mark_source(ax: Any, pos: np.ndarray) -> None:
    ax.scatter([pos[0]], [pos[1]], [pos[2]], s=52, marker="*",
               color=NATURE_PALETTE["glow"], edgecolor=NATURE_PALETTE["axis"],
               linewidth=0.5, depthshade=False, zorder=10)


# ── the flat map ───────────────────────────────────────────────────────────

def _unroll(pos: np.ndarray, axis_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cylindrical unroll about a vertical axis: (θ°, z)."""
    d = pos[:, :2] - axis_xy
    return np.degrees(np.arctan2(d[:, 1], d[:, 0])), pos[:, 2]


def _panel_unrolled(
    ax: Any, sensor_pos: np.ndarray, vals: np.ndarray, src_pos: np.ndarray,
    *, axis_xy: np.ndarray, vlim: float, cmap: Any, unit: str,
) -> Any:
    """The field with the viewpoint taken out of it: θ around the body vs height.

    Gridded rather than scattered — a scatter of 4,740 dots reads as noise,
    while the same data as a filled contour reads as the dipolar pattern it is.
    """
    th, z = _unroll(sensor_pos, axis_xy)
    ti = np.linspace(-180, 180, 240)
    zi = np.linspace(z.min(), z.max(), 260)
    TI, ZI = np.meshgrid(ti, zi)

    # Interpolate on the unrolled plane, wrapping θ so the seam at ±180° is not
    # a discontinuity in a quantity that is continuous around the body.
    # Work in millimetres of arc, not degrees: a degree near the neck is ~1 mm
    # and the kernel has to be the same physical size in both directions or the
    # map comes out tiled along whichever axis was implicitly compressed.
    radius_mm = float(np.median(np.linalg.norm(sensor_pos[:, :2] - axis_xy, axis=1)))
    deg2mm = np.pi * radius_mm / 180.0
    pts = np.column_stack([np.concatenate([th - 360, th, th + 360]) * deg2mm,
                           np.tile(z, 3)])
    vv = np.tile(vals, 3)
    tree = cKDTree(pts)
    q = np.column_stack([TI.ravel() * deg2mm, ZI.ravel()])
    dist, idx = tree.query(q, k=min(14, len(pts)))
    w = np.exp(-0.5 * (dist / 34.0) ** 2)
    grid = ((w * vv[idx]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-12))
    grid = grid.reshape(TI.shape)
    grid[dist.min(axis=1).reshape(TI.shape) > 55.0] = np.nan

    im = ax.contourf(TI, ZI, grid, levels=np.linspace(-vlim, vlim, 25),
                     cmap=cmap, extend="both")
    im.set_edgecolor("face")              # kill the hairline seams in vector output
    # No zero contour: through the near-zero majority of the map it traces
    # interpolation noise and reads as structure that is not there.

    s_th, s_z = _unroll(src_pos[None, :], axis_xy)
    ax.scatter(s_th, s_z, s=70, marker="*", color=NATURE_PALETTE["glow"],
               edgecolor=NATURE_PALETTE["axis"], linewidth=0.6, zorder=6)
    ax.set_xlabel("Angle around the body  (°)")
    ax.set_ylabel("Height  z (mm)")
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["back", "left", "front", "right", "back"])
    ax.set_title(f"Field unrolled  ·  {unit}", pad=6)
    return im


# ── figure ─────────────────────────────────────────────────────────────────

def render_torso_topoplot(
    cfg: Config, *, source_idx: int = -1, moment: str = "z",
    out_path: Path | None = None, dpi: int = 300,
) -> Path:
    """Render the four-panel body-surface field map for the configured target."""
    apply_nature_style()
    rng = np.random.default_rng(0)
    cmap = divergent_cmap()

    lf = load_leadfield(cfg.outputs.forward_npz)
    eeg_lf = None
    if cfg.outputs.forward_eeg_npz.exists():
        eeg_lf = load_leadfield(cfg.outputs.forward_eeg_npz)
    if source_idx < 0:
        source_idx = default_source_idx(lf, eeg_lf)

    src = np.asarray(lf.source_pos, dtype=float)[source_idx]
    vals, sensors = _signed_radial_field(lf, source_idx, moment)

    geom = load_geometry(cfg.outputs.geometry_mat)
    skin = geom.compartments["mesh_skin"]
    sv, sf = np.asarray(skin.vertices, float), np.asarray(skin.faces, np.int64)

    # Show the slab the array actually covers, centred on the source, rather
    # than the whole body: a neck array on a full torso is a stamp on a wall.
    half = 190.0
    sv_b, sf_b = _crop_to_band(sv, sf, src[2] - half, src[2] + half)
    pitch = float(cfg.sensors.resolution_mm)
    vals_v, near_v = _interpolate_to_surface(
        sensors, vals, sv_b, sigma_mm=1.1 * pitch, k_nearest=20)
    # Scale to what the panels actually show. Taking the percentile over all
    # 4,740 channels instead sets the range from sensors on the far side of the
    # torso, and the neck — the only part in frame — comes out uniformly pale.
    measured = np.abs(vals_v[near_v <= 1.6 * pitch])
    vlim = float(np.percentile(measured, 98.0)) if measured.size else 1.0
    axis_xy = sv_b[:, :2].mean(axis=0)

    fig = plt.figure(figsize=(11.4, 8.4))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.92],
        left=0.03, right=0.97, top=0.90, bottom=0.09, wspace=0.10, hspace=0.20)

    # a — the whole upper body for context, viewed from the front.
    # b — a close-up from the side the source is on. The back view that used to
    #     sit here is nearly blank: this is a neck source, and the far side of
    #     the torso genuinely sees almost nothing, so the panel spent its space
    #     proving that rather than showing the pattern.
    axis_xy_body = sv_b[:, :2].mean(axis=0)
    src_azim = float(np.degrees(np.arctan2(src[1] - axis_xy_body[1],
                                           src[0] - axis_xy_body[0])))
    # a — MEG on the whole upper body, from the source's own side, so the
    #     strong lobe is in frame rather than round the back.
    axa = fig.add_subplot(gs[0, 0], projection="3d", computed_zorder=False)
    _draw_surface(axa, sv_b, sf_b, vals_v, near_v, vlim=vlim,
                  max_extrap_mm=1.6 * pitch, cmap=cmap, rng=rng)
    if _source_is_visible(src, axis_xy_body, src_azim):
        _mark_source(axa, src)
    _frame_3d(axa, sv_b, elev=10.0, azim=src_azim)
    _scale_bar(axa, src_azim, mm=50.0)
    axa.set_title("MEG  ·  magnetic field on the skin", pad=-2)
    add_panel_label(axa, "a", x=0.02, y=0.97)

    # c — the same magnetic field, unrolled.
    axc = fig.add_subplot(gs[1, 0])
    band = (sensors[:, 2] >= src[2] - half) & (sensors[:, 2] <= src[2] + half)
    im = _panel_unrolled(axc, sensors[band], vals[band], src, axis_xy=axis_xy,
                         vlim=vlim, cmap=cmap, unit="fT per nA·m")
    add_panel_label(axc, "c")

    # b — the electric potential on the skin the patch sits on.
    axd = fig.add_subplot(gs[0, 1], projection="3d", computed_zorder=False)
    if eeg_lf is not None:
        # Electrodes are scalar contacts, one channel each — no radial subset.
        e_pos = np.asarray(eeg_lf.coil_pos, dtype=float)
        e_val = np.asarray(eeg_lf.L_fT_per_nAm, float)[
            :, 3 * source_idx + {"x": 0, "y": 1, "z": 2}[moment]]
        # 32 contacts, no outlier tail to guard against — use the true peak so
        # the colour bar and the quoted peak in the title are the same number.
        e_lim = float(np.abs(e_val).max()) or 1.0
        centre = e_pos.mean(axis=0)
        radius = float(np.linalg.norm(e_pos - centre, axis=1).max()) + 10.0
        sv_p, sf_p = _crop_near(sv, sf, centre, radius)
        e_vals_v, e_near_v = _interpolate_to_surface(
            e_pos, e_val, sv_p,
            sigma_mm=1.8 * float(cfg.electrodes.contact_pitch_mm), k_nearest=10)
        _draw_surface(axd, sv_p, sf_p, e_vals_v, e_near_v, vlim=e_lim,
                      max_extrap_mm=14.0, cmap=cmap, rng=rng)
        axd.scatter(e_pos[:, 0], e_pos[:, 1], e_pos[:, 2], s=24,
                    facecolor="none", edgecolor=NATURE_PALETTE["axis"],
                    linewidth=0.6, depthshade=False)
        _mark_source(axd, src)
        # Look along the patch normal — straight at the contacts, from outside
        # the body — so the panel shows the patch rather than the neck it is on.
        out_xy = centre[:2] - axis_xy_body
        azim = float(np.degrees(np.arctan2(out_xy[1], out_xy[0])))
        # No extra zoom here: the patch panel is already framed to the patch,
        # and zooming again pushes the contacts off the edge of the panel.
        _frame_3d(axd, sv_p, elev=6, azim=azim, centre=centre,
                  half_mm=radius * 0.95, zoom=1.0)
        _scale_bar(axd, azim, mm=25.0)
        axd.set_title("ESG  ·  electric potential on the skin", pad=-2)

    else:
        axd.set_axis_off()
        axd.text2D(0.5, 0.5, "no electrode leadfield for this target",
                   transform=axd.transAxes, ha="center", va="center",
                   fontsize=8, color=NATURE_PALETTE["grey"])
    add_panel_label(axd, "b", x=0.02, y=0.97)

    # d — the patch's own topography, in the patch's own frame. The paddle
    #     renderer is shared with inob.viz.topoplot so the layout (5x6 body grid
    #     plus head and foot contacts) is drawn one way everywhere.
    axdd = fig.add_subplot(gs[1, 1])
    if eeg_lf is not None:
        from inob.viz.topoplot import _draw_eeg_2d_topoplot
        electrodes = load_sensors(cfg.outputs.electrodes_mat)
        # colorbar=False: this figure supplies one horizontal bar per column,
        # and the helper's own vertical bar would make two for the same data.
        _draw_eeg_2d_topoplot(axdd, electrodes, e_val, cfg,
                              vmin=-e_lim, vmax=e_lim, colorbar=False)
        axdd.set_title(
            f"ESG  ·  {len(e_val)} contacts  ·  peak "
            f"{np.abs(e_val).max():.2f} µV per nA·m", pad=6)
    else:
        axdd.set_axis_off()
    add_panel_label(axdd, "d", x=-0.16, y=1.10)
    if eeg_lf is not None:
        cb_e = fig.colorbar(
            plt.cm.ScalarMappable(Normalize(-e_lim, e_lim), cmap),
            ax=[axdd], orientation="horizontal", fraction=0.045, pad=0.16,
            shrink=0.72)
        cb_e.set_label("Electric potential, µV per nA·m", fontsize=7)
        cb_e.ax.tick_params(labelsize=6.5)

    # One colour bar per column: the two modalities have different units and
    # differ by four orders of magnitude, so a shared bar would be meaningless.
    cb = fig.colorbar(im, ax=[axc], orientation="horizontal",
                      fraction=0.045, pad=0.16, shrink=0.72)
    cb.set_label("Magnetic field, fT per nA·m", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    region = source_region_label(cfg)
    fig.suptitle(
        f"{region} — field on the body surface   ·   source #{source_idx} "
        f"at z = {src[2]:.0f} mm, {moment}-oriented, 1 nA·m",
        y=0.975, fontsize=11)

    out = out_path or target_output(cfg, "torso_topoplot.png")
    return save_figure(fig, out, dpi=dpi)
