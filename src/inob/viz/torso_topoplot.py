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

from inob.anatomy import level_suffix
from inob.config import (
    SOURCE_TARGETS,
    Config,
    source_region_label,
    source_target_tag,
    target_output,
)
from inob.io.hdf5 import load_geometry
from inob.io.npz import load_leadfield
from inob.viz.detectability import default_source_idx, matching_eeg_source_idx
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
    The radial channels are found by label rather than as "the first third",
    because a single-axis or imported array has no thirds.
    """
    from inob.sensors.layout import radial_channels

    k = {"x": 0, "y": 1, "z": 2}[moment]
    L = np.asarray(lf.L_fT_per_nAm, dtype=float)
    radial = radial_channels(lf.channel_names)
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


#: Length of the arrow drawn for the dipole moment. Long enough to read its
#: direction at column width, short enough not to cover the lobes it explains.
_DIPOLE_ARROW_MM = 45.0

_MOMENT_VEC = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _dipole_arrow(pos: np.ndarray, moment: str, azim: float) -> list[np.ndarray]:
    """Shaft and two head strokes of the dipole arrow, as 2-point 3-D segments.

    The head is spread along the screen-horizontal direction for the panels'
    azimuth (see :func:`_scale_bar`), not along a fixed body axis: a z arrow
    whose head opens in x shows no head at all when viewed along x, which is
    how the vagus panel first came out. A moment pointing mostly along the
    line of sight has no drawable direction and gets its head opened in z.
    """
    u = np.asarray(_MOMENT_VEC[moment], float)
    tail, tip = pos - 0.5 * _DIPOLE_ARROW_MM * u, pos + 0.5 * _DIPOLE_ARROW_MM * u
    th = np.radians(azim)
    side = np.array([-np.sin(th), np.cos(th), 0.0])
    side = side - (side @ u) * u
    if np.linalg.norm(side) < 0.3:
        side = np.array([0.0, 0.0, 1.0]) - u[2] * u
    side /= np.linalg.norm(side)
    base = tip - 0.3 * _DIPOLE_ARROW_MM * u
    w = 0.14 * _DIPOLE_ARROW_MM
    return [
        np.array([tail, tip]),
        np.array([base + w * side, tip]),
        np.array([base - w * side, tip]),
    ]


def _mark_source(ax: Any, pos: np.ndarray, moment: str = "z", azim: float = -90.0) -> None:
    """The mapped dipole: a dot at its position and an arrow along its moment.

    A star alone says where but not which way, and the whole pattern on the
    skin is set by which way. The arrow is centred on the source so it points
    through it.
    """
    for seg in _dipole_arrow(pos, moment, azim):
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            seg[:, 2],
            color=NATURE_PALETTE["axis"],
            lw=1.7,
            solid_capstyle="round",
            zorder=11,
        )
    ax.scatter(
        [pos[0]],
        [pos[1]],
        [pos[2]],
        s=26,
        marker="o",
        color=NATURE_PALETTE["glow"],
        edgecolor=NATURE_PALETTE["axis"],
        linewidth=0.6,
        depthshade=False,
        zorder=12,
    )


def _mark_target(
    ax: Any,
    target_pos: np.ndarray,
    axis_xy: np.ndarray,
    azim: float,
    z_range: tuple[float, float],
) -> int:
    """Draw the target structure — every source the leadfield was solved for.

    Without it the panel reads as a dipole floating in a skin mesh: nothing
    says it sits in the cord (or the vagus, or a muscle). The sampled sources
    *are* the target compartment, so they trace it without a second mesh.
    Drawn as an x-ray overlay, and only on the camera's side of the body, for
    the depth-sorting reason in :func:`_source_is_visible`. Returns the count.
    """
    view = np.array([np.cos(np.radians(azim)), np.sin(np.radians(azim))])
    outward = target_pos[:, :2] - axis_xy
    facing = outward @ view > 0.0
    in_band = (target_pos[:, 2] >= z_range[0]) & (target_pos[:, 2] <= z_range[1])
    shown = target_pos[facing & in_band]
    if not len(shown):
        return 0
    # A nerve is a line of a hundred sources; a muscle is a cloud of several
    # hundred, which at full size buries the field it is meant to explain.
    dense = len(shown) > 150
    ax.scatter(
        shown[:, 0],
        shown[:, 1],
        shown[:, 2],
        s=2.0 if dense else 5.0,
        marker="o",
        color=NATURE_PALETTE["teal"],
        edgecolor="none",
        alpha=0.45 if dense else 0.9,
        depthshade=False,
        zorder=10,
    )
    return len(shown)


def _screen_axes(elev: float, azim: float) -> dict[str, np.ndarray]:
    """Where +x, +y and +z point on the page under the panels' camera.

    The orthographic camera at (elev, azim) has screen-right
    (-sin a, cos a, 0) and screen-up (-sin e cos a, -sin e sin a, cos e);
    each body axis is drawn as its projection onto those two.
    """
    a, e = np.radians(azim), np.radians(elev)
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    return {k: np.array([v @ right, v @ up]) for k, v in zip("xyz", np.eye(3), strict=True)}


#: Axis names as a reader needs them (docs/COORDINATES.md): +Y is posterior,
#: so anterior is -Y.
_AXIS_WORDS = {"x": "x  subject's left", "y": "y  posterior", "z": "z  superior"}

#: Below this projected length an axis is mostly along the line of sight, and
#: its arrow alone cannot say whether it points at the reader or away.
_FORESHORTENED = 0.35


def _axis_label(axis: str, elev: float, azim: float) -> str:
    """The triad label for one axis, saying so when it points in/out of the page."""
    word = _AXIS_WORDS[axis]
    if float(np.hypot(*_screen_axes(elev, azim)[axis])) >= _FORESHORTENED:
        return word
    a, e = np.radians(azim), np.radians(elev)
    to_camera = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    towards = float(np.eye(3)["xyz".index(axis)] @ to_camera) > 0.0
    return f"{word} ({'out of' if towards else 'into'} page)"


def _axis_triad(fig: Any, rect: tuple[float, float, float, float], elev: float, azim: float) -> Any:
    """A labelled x/y/z triad for the shared camera, in its own small axes.

    The figures quote heights as "z = 1393 mm" and the panels had no axes to
    say what z is. One triad serves both panels because they share a camera.
    An axis pointing mostly into the page is drawn short, as it projects.
    """
    ax = fig.add_axes(rect)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    for k, (dx, dy) in _screen_axes(elev, azim).items():
        ax.annotate(
            "",
            xy=(dx, dy),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": NATURE_PALETTE["axis"]},
        )
        n = float(np.hypot(dx, dy))
        # Labels sit past the tip; a foreshortened axis gets its label pushed
        # out to a fixed radius so it does not collide with the origin.
        r = max(n, 0.45) + 0.18
        lx, ly = (dx / n * r, dy / n * r) if n > 1e-6 else (0.25, -0.25)
        ha = "left" if lx > 0.05 else "right" if lx < -0.05 else "center"
        ax.text(lx, ly, _axis_label(k, elev, azim), fontsize=6.5, ha=ha, va="center")
    ax.set_title("body axes", fontsize=6.5, fontweight="normal", pad=1)
    return ax


def target_structure_label(cfg: Config) -> str:
    """What the sources sit in, in words: "spinal cord", "left vagus"."""
    tag = source_target_tag(cfg)
    tissues = SOURCE_TARGETS.get(tag, {}).get("tissues", "")
    if not tissues:
        return source_region_label(cfg)
    words = {"spinal_cord": "spinal cord", "vagus_left": "left vagus", "vagus_right": "right vagus"}
    return " + ".join(words.get(t, t.replace("_", " ")) for t in tissues.split(","))


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


#: How far round the body a sensor may sit and still be drawn. A plain
#: front/back test keeps everything out to the silhouette, where the markers
#: stop lying on the visible surface and trail around the outline instead.
_SENSOR_FACING_MIN = 0.35


def _mark_sensors(
    ax: Any,
    pos: np.ndarray,
    axis_xy: np.ndarray,
    azim: float,
    z_range: tuple[float, float],
) -> int:
    """Draw the front-facing sensor positions over the painted field.

    Without these the map reads as a field measured continuously on the skin.
    It is not: it is a Gaussian interpolation of discrete readings taken one
    stand-off out from the surface, and the interpolation is wide enough to
    smooth away the spacing it was built from. Marking the positions puts the
    sampling back in the figure, so the reader can see the array's resolution
    rather than take the smoothness at face value.

    Only positions facing the camera, and within the band of body the panel
    actually shows, are drawn — for the depth-sorting reason in
    :func:`_source_is_visible`. Returns how many were drawn.
    """
    view = np.array([np.cos(np.radians(azim)), np.sin(np.radians(azim))])
    outward = pos[:, :2] - axis_xy
    norm = np.linalg.norm(outward, axis=1, keepdims=True)
    facing = (outward / np.where(norm > 1e-9, norm, 1.0)) @ view > _SENSOR_FACING_MIN
    # The mesh is cropped to a band; the array is not. Sensors outside it have
    # no surface under them and would be drawn against the background.
    in_band = (pos[:, 2] >= z_range[0]) & (pos[:, 2] <= z_range[1])
    front = pos[facing & in_band]
    if not len(front):
        return 0
    ax.scatter(
        front[:, 0],
        front[:, 1],
        front[:, 2],
        s=3.5,
        marker="o",
        facecolor="none",
        edgecolor=NATURE_PALETTE["axis"],
        linewidth=0.35,
        alpha=0.55,
        depthshade=False,
        zorder=9,
    )
    return len(front)


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
    show_sensors: bool = True,
    moment: str = "z",
    target_pos: np.ndarray | None = None,
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
    z_range = (float(verts[:, 2].min()), float(verts[:, 2].max()))
    if show_sensors:
        _mark_sensors(ax, sensor_pos, axis_xy, azim, z_range)
    if target_pos is not None:
        _mark_target(ax, target_pos, axis_xy, azim, z_range)
    if _source_is_visible(src, axis_xy, azim):
        _mark_source(ax, src, moment, azim)
    _frame_3d(ax, verts, elev=elev, azim=azim)
    _scale_bar(ax, azim, mm=50.0)
    ax.set_title(title, pad=-2)
    return vlim


# ── figure ─────────────────────────────────────────────────────────────────


def torso_title(region: str, source_idx: int, z_mm: float, moment: str, level: str = "") -> str:
    """The figure title: which source, where, in words a reader can place.

    ``level`` is the vertebral landmark suffix, e.g. " (C6/C7)", or "" when
    the bone meshes are not available to name one.
    """
    return (
        f"{region} — field on the body surface   ·   source #{source_idx}, "
        f"z = {z_mm:.0f} mm{level}, {moment}-oriented, 1 nA·m"
    )


def render_torso_topoplot(
    cfg: Config,
    *,
    source_idx: int = -1,
    moment: str = "z",
    eeg_npz: Path | None = None,
    out_path: Path | None = None,
    dpi: int = 300,
    show_sensors: bool = True,
) -> Path:
    """Render the two-panel body-surface field map for the configured target.

    ``show_sensors`` overlays the array positions, so the painted field is
    read as the interpolation of a discrete array that it is rather than as a
    continuous skin measurement. Turn it off for a presentation slide where
    the markers only add clutter.
    """
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

    target_pos = np.asarray(lf.source_pos, dtype=float)

    fig = plt.figure(figsize=(10.6, 7.4))
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.86, bottom=0.13, wspace=0.02)

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
        show_sensors=show_sensors,
        elev=elev,
        azim=azim,
        axis_xy=axis_xy,
        cmap=cmap,
        rng=rng,
        title="MEG  ·  magnetic field on the skin",
        moment=moment,
        target_pos=target_pos,
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
        e_idx = matching_eeg_source_idx(lf, eeg_lf, source_idx)
        e_val = np.asarray(eeg_lf.L_fT_per_nAm, float)[:, 3 * e_idx + k]
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
            show_sensors=show_sensors,
            elev=elev,
            azim=azim,
            axis_xy=axis_xy,
            cmap=cmap,
            rng=rng,
            title="ESG  ·  electric potential on the skin",
            moment=moment,
            target_pos=target_pos,
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

    _axis_triad(fig, (0.0, 0.40, 0.085, 0.16), elev, azim)

    region = source_region_label(cfg)
    structure = target_structure_label(cfg)
    fig.suptitle(
        torso_title(
            region, source_idx, float(src[2]), moment, level_suffix(cfg.data.bone_dir, src[2])
        ),
        y=0.985,
        fontsize=11,
    )
    # Which forward model drew this: the text compares FEM against the sphere
    # and Biot-Savart models, and a reader of this figure alone cannot tell.
    fig.text(
        0.5,
        0.945,
        "Both panels: full-body finite-element forward model (DUNEuro), "
        "not the sphere or Biot–Savart approximation",
        ha="center",
        fontsize=8,
        color=NATURE_PALETTE["grey"],
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker=r"$\uparrow$",
            ls="none",
            ms=9,
            color=NATURE_PALETTE["axis"],
            label=f"dipole #{source_idx} and its moment ({moment})",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=3,
            color=NATURE_PALETTE["teal"],
            label=f"target: {structure} (all modelled sources)",
        ),
    ]
    if show_sensors:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=3,
                mfc="none",
                mew=0.5,
                color=NATURE_PALETTE["axis"],
                label="sensor / electrode positions",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(handles),
        fontsize=7.5,
        handletextpad=0.4,
        columnspacing=1.6,
    )

    out = out_path or target_output(cfg, "torso_topoplot.png")
    return save_figure(fig, out, dpi=dpi)
