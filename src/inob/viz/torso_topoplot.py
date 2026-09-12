"""Two-panel field map painted on the body surface itself.

One figure per source target, the two modalities side by side:

    a  MEG  — the magnetic field on the skin
    b  ESG  — the electric potential on the skin

Same mesh, same crop, same camera, same orthographic projection, same scale
bar, same colour rule. The only thing that differs between the panels is the
physics, which is the whole point of the layout: the magnetic and electric
patterns of one current dipole are rotated roughly a quarter turn from each
other, and that is only legible if neither panel has been re-aimed or
re-scaled to flatter itself. The coverage difference falls out of the same
choice — a asks the whole torso, b can only ask the postage stamp of skin the
patch sits on, and at a shared scale you see how small that stamp is.

Three rules keep it honest:

  * **Nothing is painted where nothing was measured.** A vertex further than
    ``max_extrap_mm`` from the nearest sensor is drawn in neutral skin, not in
    an interpolated colour. Otherwise a Gaussian interpolation happily paints a
    confident-looking field across the whole torso from a neck array.
  * **The colour scale is symmetric about zero** and clipped to a high
    percentile of the skin that was actually measured, rather than the raw
    maximum, because one sensor sitting a millimetre from the source otherwise
    sets the scale for the entire body.
  * **The same rule sets both scales.** The two colour bars carry different
    units and differ by four orders of magnitude, so they cannot be shared;
    what is shared is the rule that produces them, so neither pattern is
    flattened or saturated relative to the other.

The colour is a signed field with a neutral midpoint (blue negative, white
zero, red positive), so the two lobes of a dipolar pattern read immediately.

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
from inob.io.hdf5 import load_geometry
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
    lf: Any,
    source_idx: int,
    moment: str = "z",
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
    sensor_pos: np.ndarray,
    sensor_val: np.ndarray,
    verts: np.ndarray,
    *,
    sigma_mm: float,
    k_nearest: int = 12,
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


def _crop_to_band(
    verts: np.ndarray,
    faces: np.ndarray,
    z_lo: float,
    z_hi: float,
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
    lam = np.abs(n @ light)  # abs: unknown winding
    shade = (0.62 + 0.38 * lam)[:, None]
    out = rgba.copy()
    out[:, :3] = np.clip(out[:, :3] * shade, 0.0, 1.0)
    return out


def _draw_surface(
    ax: Any,
    verts: np.ndarray,
    faces: np.ndarray,
    vals: np.ndarray,
    nearest_mm: np.ndarray,
    *,
    vlim: float,
    max_extrap_mm: float,
    cmap: Any,
    rng: np.random.Generator,
) -> None:
    """Paint the field on a mesh; leave unmeasured skin neutral."""
    face_val = vals[faces].mean(axis=1)
    face_near = nearest_mm[faces].min(axis=1)
    rgba = cmap(Normalize(-vlim, vlim)(face_val))
    # Neutral skin wherever the array had no sensor near enough to know.
    blind = face_near > max_extrap_mm
    rgba[blind] = plt.matplotlib.colors.to_rgba(NATURE_PALETTE["skin"])

    coll = Poly3DCollection(
        verts[faces],
        facecolors=_shade(verts, faces, rgba),
        edgecolors="none",
        linewidths=0,
        shade=False,
        zsort="average",
    )
    ax.add_collection3d(coll)


def _frame_3d(
    ax: Any,
    verts: np.ndarray,
    *,
    elev: float,
    azim: float,
    centre: np.ndarray | None = None,
    half_mm: float | None = None,
    zoom: float = 1.45,
) -> None:
    """A 3-D axis with nothing on it but the body: no panes, grid, or ticks.

    ``centre``/``half_mm`` zoom the view without cropping the mesh — which is
    how the close-up keeps the neck and shoulder in frame. Cropping to a short
    slab instead makes the neck read as a mound with its top sliced off.
    """
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    mid, span = (lo + hi) / 2, (hi - lo) * 0.54  # per-axis half-extent
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

    The anchor corner follows that direction: starting the bar at the same
    corner regardless of azimuth sends it out of the panel whenever the screen
    axis runs the other way, and the label is then clipped at the frame edge.
    """
    (x0, x1), (y0, y1), (z0, z1) = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    th = np.radians(azim)
    dx, dy = -np.sin(th) * mm, np.cos(th) * mm
    bx = x0 + 0.10 * (x1 - x0) if dx >= 0 else x1 - 0.10 * (x1 - x0)
    by = y0 + 0.10 * (y1 - y0) if dy >= 0 else y1 - 0.10 * (y1 - y0)
    bz = z0 + 0.06 * (z1 - z0)
    ax.plot(
        [bx, bx + dx],
        [by, by + dy],
        [bz, bz],
        color=NATURE_PALETTE["axis"],
        lw=1.6,
        solid_capstyle="butt",
    )
    ax.text(
        bx + dx / 2,
        by + dy / 2,
        bz - 0.05 * (z1 - z0),
        f"{mm:.0f} mm",
        ha="center",
        va="top",
        fontsize=6.5,
        color=NATURE_PALETTE["axis"],
    )


def _source_is_visible(
    pos: np.ndarray,
    axis_xy: np.ndarray,
    azim: float,
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
    ax.scatter(
        [pos[0]],
        [pos[1]],
        [pos[2]],
        s=52,
        marker="*",
        color=NATURE_PALETTE["glow"],
        edgecolor=NATURE_PALETTE["axis"],
        linewidth=0.5,
        depthshade=False,
        zorder=10,
    )


def _array_pitch_mm(pos: np.ndarray) -> float:
    """Median nearest-neighbour spacing of a sensor array, in mm.

    The interpolation kernel and the reach of the "measured" mask both have to
    be the array's own resolution, and only the array knows it: the ESG panel
    may be fed a 32-contact paddle at 5 mm pitch or a whole-torso array at
    ~40 mm, and ``electrodes.contact_pitch_mm`` describes only the first.
    """
    if len(pos) < 2:
        return 1.0
    d, _ = cKDTree(pos).query(pos, k=2)
    # The 90th percentile, not the median: a whole-torso array is sampled from
    # the skin rather than gridded, so its spacing has a long tail, and a
    # median-sized mask leaves the sparse patches speckled with unpainted
    # holes that read as structure in the field.
    return float(np.percentile(d[:, 1], 90.0))


def _body_panel(
    ax: Any,
    verts: np.ndarray,
    faces: np.ndarray,
    sensor_pos: np.ndarray,
    sensor_val: np.ndarray,
    src: np.ndarray,
    *,
    sigma_mm: float,
    k_nearest: int,
    max_extrap_mm: float,
    elev: float,
    azim: float,
    axis_xy: np.ndarray,
    cmap: Any,
    rng: np.random.Generator,
    title: str,
) -> float:
    """Paint one modality on the body band in the shared camera; return its vlim.

    Both panels go through here, so the framing cannot drift between them: the
    mesh, the view angles, the projection and the scale bar are arguments the
    caller passes once rather than decisions taken twice. Only the
    interpolation kernel and the reach of the array differ, because a 5 mm
    contact pitch and a 25 mm sensor pitch do not see the skin at the same
    resolution.
    """
    vals_v, near_v = _interpolate_to_surface(
        sensor_pos, sensor_val, verts, sigma_mm=sigma_mm, k_nearest=k_nearest
    )
    # Scale to the skin this panel actually measured. Taking the percentile
    # over every channel instead sets the range from sensors on the far side of
    # the torso, and the part in frame comes out uniformly pale.
    measured = np.abs(vals_v[near_v <= max_extrap_mm])
    vlim = float(np.percentile(measured, 98.0)) if measured.size else 1.0
    _draw_surface(
        ax, verts, faces, vals_v, near_v, vlim=vlim, max_extrap_mm=max_extrap_mm, cmap=cmap, rng=rng
    )
    if _source_is_visible(src, axis_xy, azim):
        _mark_source(ax, src)
    _frame_3d(ax, verts, elev=elev, azim=azim)
    _scale_bar(ax, azim, mm=50.0)
    ax.set_title(title, pad=-2)
    return vlim


# ── figure ─────────────────────────────────────────────────────────────────


def render_torso_topoplot(
    cfg: Config,
    *,
    source_idx: int = -1,
    moment: str = "z",
    eeg_npz: Path | None = None,
    out_path: Path | None = None,
    dpi: int = 300,
) -> Path:
    """Render the two-panel body-surface field map for the configured target."""
    apply_nature_style()
    rng = np.random.default_rng(0)
    cmap = divergent_cmap()

    lf = load_leadfield(cfg.outputs.forward_npz)
    # The default source is still the one the target's own HD patch was sited
    # over, whatever panel b is painted from: `--esg-npz` changes which
    # electrodes are drawn, not which source the figure is about.
    patch_lf = (
        load_leadfield(cfg.outputs.forward_eeg_npz)
        if cfg.outputs.forward_eeg_npz.exists()
        else None
    )
    if source_idx < 0:
        source_idx = default_source_idx(lf, patch_lf)
    # Panel b wants the electrode array that covers the same body panel a does.
    # The configured one is the target's HD paddle — right for detectability,
    # a postage stamp here — so the caller may point at a whole-torso array
    # instead; `inob eeg` writes one given `electrodes.shape=whole_body`.
    eeg_path = eeg_npz or cfg.outputs.forward_eeg_npz
    eeg_lf = load_leadfield(eeg_path) if eeg_path != cfg.outputs.forward_eeg_npz else patch_lf

    k = {"x": 0, "y": 1, "z": 2}[moment]
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

    # The one camera both panels use: the source's own side of the body, so the
    # strong lobe is in frame rather than round the back.
    axis_xy = sv_b[:, :2].mean(axis=0)
    elev = 10.0
    azim = float(np.degrees(np.arctan2(src[1] - axis_xy[1], src[0] - axis_xy[0])))

    fig = plt.figure(figsize=(10.6, 7.2))
    gs = fig.add_gridspec(1, 2, left=0.045, right=0.955, top=0.90, bottom=0.13, wspace=0.02)

    # a — MEG on the body band.
    axa = fig.add_subplot(gs[0, 0], projection="3d", computed_zorder=False)
    vlim = _body_panel(
        axa,
        sv_b,
        sf_b,
        sensors,
        vals,
        src,
        sigma_mm=1.1 * pitch,
        k_nearest=20,
        max_extrap_mm=1.6 * pitch,
        elev=elev,
        azim=azim,
        axis_xy=axis_xy,
        cmap=cmap,
        rng=rng,
        title="MEG  ·  magnetic field on the skin",
    )
    add_panel_label(axa, "a", x=0.02, y=0.97)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(Normalize(-vlim, vlim), cmap),
        ax=[axa],
        orientation="horizontal",
        fraction=0.04,
        pad=0.02,
        shrink=0.62,
    )
    cb.set_label("Magnetic field, fT per nA·m", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # b — ESG on the same band, same camera, same colour rule. Different units,
    #     so its own bar; everything else about the panel is identical to a.
    axb = fig.add_subplot(gs[0, 1], projection="3d", computed_zorder=False)
    if eeg_lf is not None:
        # Electrodes are scalar contacts, one channel each — no radial subset.
        e_pos = np.asarray(eeg_lf.coil_pos, dtype=float)
        e_val = np.asarray(eeg_lf.L_fT_per_nAm, float)[:, 3 * source_idx + k]
        # The same interpolation rule as panel a, at the electrode array's own
        # resolution — so what is painted, and what is left neutral, is decided
        # the same way for both modalities.
        e_pitch = _array_pitch_mm(e_pos)
        e_lim = _body_panel(
            axb,
            sv_b,
            sf_b,
            e_pos,
            e_val,
            src,
            sigma_mm=1.1 * e_pitch,
            k_nearest=20,
            max_extrap_mm=1.6 * e_pitch,
            elev=elev,
            azim=azim,
            axis_xy=axis_xy,
            cmap=cmap,
            rng=rng,
            title="ESG  ·  electric potential on the skin",
        )
        cb_e = fig.colorbar(
            plt.cm.ScalarMappable(Normalize(-e_lim, e_lim), cmap),
            ax=[axb],
            orientation="horizontal",
            fraction=0.04,
            pad=0.02,
            shrink=0.62,
        )
        cb_e.set_label("Electric potential, µV per nA·m", fontsize=7)
        cb_e.ax.tick_params(labelsize=6.5)
    else:
        axb.set_axis_off()
        axb.text2D(
            0.5,
            0.5,
            "no electrode leadfield for this target",
            transform=axb.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=NATURE_PALETTE["grey"],
        )
    add_panel_label(axb, "b", x=0.02, y=0.97)

    region = source_region_label(cfg)
    fig.suptitle(
        f"{region} — field on the body surface   ·   source #{source_idx} "
        f"at z = {src[2]:.0f} mm, {moment}-oriented, 1 nA·m",
        y=0.975,
        fontsize=11,
    )

    out = out_path or target_output(cfg, "torso_topoplot.png")
    return save_figure(fig, out, dpi=dpi)
