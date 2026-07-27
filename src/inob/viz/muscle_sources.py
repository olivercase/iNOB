"""Pre-solve muscle figures: anatomical pairing and source fibre orientation.

Muscle-specific and self-contained: nothing outside :mod:`inob.cli.muscle_sources`
imports this module, so a vagus or spine run never pays for it (no other
command touches :mod:`inob.sources.muscle`).

  render_muscle_source_pairs         full muscle meshes coloured by L/R pair
  render_muscle_source_orientations  selected source dipoles + fibre-axis arrows

The pairing figure needs only the raw muscle STLs (``cfg.data.muscle_dir``)
and, for context, the skin compartment of a built geometry — it does not need
a FEM or a forward solve. The orientation figure plots exactly the source set
:func:`inob.sources.muscle.muscle_sources` would hand to a real forward
solve, so it does need the FEM (muscle compartment included, regardless of
which ``--source-target`` a run is currently pointed at — the FEM carries
every tissue).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from inob.config import Config
from inob.io.hdf5 import load_fem
from inob.sources.muscle import muscle_source_orientations, muscle_sources
from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
    save_figure,
)

logger = logging.getLogger(__name__)

_SIDE_RE = re.compile(r"\b(left|right)\b", re.IGNORECASE)
_MIRROR_RE = re.compile(r"\(mirrored\)", re.IGNORECASE)

MAX_TRIS_MUSCLE_PROJ = 3_000
MAX_TRIS_MUSCLE_3D = 1_000
MAX_TRIS_SKIN_PROJ = 40_000
MAX_TRIS_SKIN_3D = 8_000


def _muscle_pair_name(stl_path: Path) -> tuple[str, str]:
    """(canonical pair name, side) parsed from a BodyParts3D muscle filename.

    Filenames are ``<code>_<code>_<code>_<descriptive tail>.stl``, e.g.
    ``FJ1573_BP23424_FMA13409_Left sternocleidomastoid.stl``. Side is not
    always a leading word — the longus colli parts read "...part of right
    longus colli (mirrored)" — so search for the first "left"/"right" token
    anywhere in the tail rather than anchoring on position.
    """
    tail = stl_path.stem.split("_")[-1]
    m = _SIDE_RE.search(tail)
    side = m.group(1).capitalize() if m else "?"
    name = _SIDE_RE.sub("", tail, count=1)
    name = _MIRROR_RE.sub("", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name, side


def _require_muscle(fem) -> None:
    if "muscle" not in fem.tissue_labels:
        raise ValueError(
            "FEM has no 'muscle' tissue compartment "
            f"(have {list(fem.tissue_labels)}). Build geometry/FEM with the "
            "muscle compartment included before rendering muscle source figures."
        )


def _muscle_stl_paths(muscle_dir) -> list[Path]:
    paths = sorted(Path(muscle_dir).glob("*.stl"))
    if not paths:
        raise ValueError(f"no muscle STLs in {muscle_dir}")
    return paths


def _load_muscle_meshes(muscle_dir) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Every muscle STL as ``(pair name, side, vertices, faces)``."""
    import trimesh
    meshes = []
    for p in _muscle_stl_paths(muscle_dir):
        name, side = _muscle_pair_name(p)
        m = trimesh.load(p, process=False)
        meshes.append((
            name, side,
            np.asarray(m.vertices, dtype=np.float64),
            np.asarray(m.faces, dtype=np.int64),
        ))
    return meshes


def _load_muscle_sources(cfg: Config, *, spacing_mm: float):
    """FEM + the actual source set/orientations a muscle forward solve uses.

    Independent of ``cfg.forward.source_tissue`` / ``--source-target`` — the
    FEM mesh (unlike a per-target leadfield) carries every tissue, muscle
    included, no matter which target a run currently points at.
    """
    fem = load_fem(cfg.outputs.fem_mat)
    _require_muscle(fem)
    positions = muscle_sources(fem, spacing_mm=spacing_mm)
    orientations = muscle_source_orientations(
        fem, positions, muscle_dir=cfg.data.muscle_dir,
    )
    return fem, positions, orientations


def _muscle_backdrop(cfg: Config) -> tuple[np.ndarray, np.ndarray] | None:
    """Merged muscle-compartment (vertices, faces) for a faint context silhouette.

    ``None`` if no geometry HDF5 has been built yet — figures still render
    without it, just without the anatomical backdrop.
    """
    return _compartment_backdrop(cfg, "mesh_muscle")


def _skin_backdrop(cfg: Config) -> tuple[np.ndarray, np.ndarray] | None:
    """Skin-compartment (vertices, faces) for a translucent body context."""
    return _compartment_backdrop(cfg, "mesh_skin")


def _compartment_backdrop(cfg: Config, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not cfg.outputs.geometry_mat.exists():
        return None
    from inob.io.hdf5 import load_geometry
    geom = load_geometry(cfg.outputs.geometry_mat)
    comp = geom.compartments.get(name)
    if comp is None:
        return None
    return comp.vertices, comp.faces


def _subsample_faces(faces: np.ndarray, max_tris: int, rng: np.random.Generator) -> np.ndarray:
    if len(faces) <= max_tris:
        return faces
    idx = rng.choice(len(faces), max_tris, replace=False)
    return faces[idx]


def _crop_to_bbox(
    verts: np.ndarray, faces: np.ndarray, lo: np.ndarray, hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only faces whose vertices all fall inside ``[lo, hi]``.

    Used to crop a whole-body skin mesh down to the region around the
    muscles of interest *before* subsampling for a fill render — otherwise
    the fixed triangle budget is mostly spent on far-away skin (torso,
    limbs) that gets discarded anyway, and the region that actually matters
    ends up sparse or invisible.
    """
    vmask = np.all((verts >= lo) & (verts <= hi), axis=1)
    fmask = vmask[faces].all(axis=1)
    return verts, faces[fmask]


class _Panel:
    """One 2-D projection panel: which world axes it plots and with what sign.

    ``sign`` lets a panel show a rotated view of its plane rather than a raw
    axis-aligned one — e.g. the lateral panel plots (−Y, Z) rather than
    (Z, Y), a 90°-CCW rotation of the naive Z/Y projection, so the body's
    long axis (Z) runs vertically instead of the view looking "on its side".
    """
    __slots__ = ("title", "xi", "sx", "xlabel", "yi", "sy", "ylabel")

    def __init__(self, title, xi, sx, xlabel, yi, sy, ylabel):
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.xi, self.sx, self.yi, self.sy = xi, sx, yi, sy

    def project(self, pts: np.ndarray) -> np.ndarray:
        """``pts`` (..., 3) world coordinates -> (..., 2) signed panel coordinates."""
        return np.stack([self.sx * pts[..., self.xi], self.sy * pts[..., self.yi]], axis=-1)

    def lims(self, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float, float, float]:
        xa, xb = self.sx * lo[self.xi], self.sx * hi[self.xi]
        ya, yb = self.sy * lo[self.yi], self.sy * hi[self.yi]
        return min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb)


# BodyParts3D world axes: X left-right, Y anterior-posterior, Z inferior-
# superior (mm). Axial = looking along Z (X-Y plane); coronal/posterior =
# looking along Y, from behind (X-Z plane); sagittal/lateral = looking along
# X, from the side (Z-Y plane) — plotted as (−Y, Z), a 90° CCW rotation of
# the raw (Z, Y) projection, so the body's long axis (Z) reads vertically
# rather than the figure looking "on its side".
_PANELS = {
    "axial":   _Panel("Axial (X – Y)",     0, 1.0, "X (mm)",  1, 1.0, "Y (mm)"),
    "lateral": _Panel("Lateral (−Y – Z)", 1, -1.0, "−Y (mm)", 2, 1.0, "Z (mm)"),
    "coronal": _Panel("Posterior (X – Z)", 0, 1.0, "X (mm)",  2, 1.0, "Z (mm)"),
}


def _draw_backdrop_projection(ax, backdrop, panel: _Panel, rng) -> None:
    """Faint scattered-point context silhouette (used by the orientation figure)."""
    if backdrop is None:
        return
    verts, faces = backdrop
    sub = _subsample_faces(faces, MAX_TRIS_SKIN_PROJ, rng)
    pts = panel.project(verts[sub].reshape(-1, 3))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.2, c=NATURE_PALETTE["stone"],
               alpha=0.35, linewidths=0, rasterized=True, zorder=1)


def _draw_backdrop_3d(ax, backdrop, rng) -> None:
    if backdrop is None:
        return
    verts, faces = backdrop
    sub = _subsample_faces(faces, MAX_TRIS_SKIN_3D, rng)
    coll = Poly3DCollection(verts[sub], alpha=0.10,
                             facecolor=NATURE_PALETTE["stone"], edgecolor="none")
    ax.add_collection3d(coll)


def _draw_mesh_fill_projection(
    ax, verts, faces, panel: _Panel, colour, max_tris: int, rng, *,
    alpha: float = 1.0, zorder: int = 2,
) -> None:
    """Solid-filled triangle projection of a mesh (not a point scatter)."""
    sub = _subsample_faces(faces, max_tris, rng)
    tri2d = panel.project(verts[sub])
    ax.add_collection(PolyCollection(
        tri2d, facecolor=colour, edgecolor="none", alpha=alpha,
        zorder=zorder, rasterized=True,
    ))


def _draw_mesh_fill_3d(ax, verts, faces, colour, max_tris: int, rng, *, alpha: float = 1.0) -> None:
    sub = _subsample_faces(faces, max_tris, rng)
    ax.add_collection3d(Poly3DCollection(
        verts[sub], alpha=alpha, facecolor=colour, edgecolor="none",
    ))


def _panel_axes(fig):
    """The 4-panel (axial / lateral / posterior / 3-D) layout shared by both figures."""
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25,
                          left=0.06, right=0.97, top=0.92, bottom=0.10)
    ax_axial = fig.add_subplot(gs[0, 0])
    ax_lateral = fig.add_subplot(gs[0, 1])
    ax_coronal = fig.add_subplot(gs[1, 0])
    ax_3d = fig.add_subplot(gs[1, 1], projection="3d")
    for ax, key in ((ax_axial, "axial"), (ax_lateral, "lateral"), (ax_coronal, "coronal")):
        panel = _PANELS[key]
        ax.set_title(panel.title)
        ax.set_xlabel(panel.xlabel)
        ax.set_ylabel(panel.ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.4, alpha=0.3)
    ax_3d.set_title("3-D perspective")
    ax_3d.set_xlabel("X (mm)")
    ax_3d.set_ylabel("Y (mm)")
    ax_3d.set_zlabel("Z (mm)")
    ax_3d.view_init(elev=15, azim=35)
    return ax_axial, ax_lateral, ax_coronal, ax_3d


def _set_limits(axes, all_pts: np.ndarray, *, margin: float = 0.05) -> None:
    ax_axial, ax_lateral, ax_coronal, ax_3d = axes
    lo, hi = all_pts.min(0), all_pts.max(0)
    span = hi - lo
    lo, hi = lo - margin * span, hi + margin * span
    for ax, key in ((ax_axial, "axial"), (ax_lateral, "lateral"), (ax_coronal, "coronal")):
        x0, x1, y0, y1 = _PANELS[key].lims(lo, hi)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    ax_3d.set_xlim(lo[0], hi[0]); ax_3d.set_ylim(lo[1], hi[1]); ax_3d.set_zlim(lo[2], hi[2])


# Fixed hue order over canonical pair names (assigned once below, sorted by
# name so the same muscle always gets the same colour across regenerations —
# never resampled/cycled per run). tab20 gives 20 mutually-distinguishable
# categorical steps, comfortably more than the ~18 neck-muscle pairs here.
_PAIR_CMAP = plt.get_cmap("tab20")


def render_muscle_source_pairs(
    cfg: Config, *, out_path: Path | None = None, dpi: int = 150,
) -> Path:
    """Full muscle meshes, each coloured by its anatomical left/right pair.

    Corresponding muscles on both sides of the body (``Left``/``Right``
    sternocleidomastoid, ...) get the same colour, parsed straight from the
    BodyParts3D STL filenames. A translucent skin mesh is drawn underneath
    for anatomical context — where in the body this all sits.
    """
    apply_nature_style()
    rng = np.random.default_rng(cfg.reproducibility.seed)

    meshes = _load_muscle_meshes(cfg.data.muscle_dir)
    pair_names = sorted({name for name, _side, _v, _f in meshes})
    pair_colour = {name: _PAIR_CMAP(i % 20) for i, name in enumerate(pair_names)}

    skin = _skin_backdrop(cfg)
    # View scoped tightly to the muscle region, not the whole body — the skin
    # mesh is drawn only for immediate context (where on the body this sits)
    # and must not pull the view out to a half-body shot.
    all_pts = np.vstack([v for _n, _s, v, _f in meshes])
    view_margin = 0.08
    if skin is not None:
        muscle_lo, muscle_hi = all_pts.min(0), all_pts.max(0)
        crop_pad = 0.35 * (muscle_hi - muscle_lo)
        skin_v, skin_f = _crop_to_bbox(
            skin[0], skin[1], muscle_lo - crop_pad, muscle_hi + crop_pad,
        )
        skin = (skin_v, skin_f) if len(skin_f) else None

    fig = plt.figure(figsize=(13, 11))
    fig.suptitle(
        f"Muscle anatomy by pair — {len(meshes)} muscles "
        f"({len(pair_names)} left/right pairs)",
        fontsize=11, fontweight="bold", y=0.985,
    )
    axes = _panel_axes(fig)
    ax_axial, ax_lateral, ax_coronal, ax_3d = axes
    for ax, key in ((ax_axial, "axial"), (ax_lateral, "lateral"), (ax_coronal, "coronal")):
        panel = _PANELS[key]
        if skin is not None:
            _draw_mesh_fill_projection(ax, skin[0], skin[1], panel,
                                       NATURE_PALETTE["skin"], MAX_TRIS_SKIN_PROJ,
                                       rng, alpha=0.12, zorder=1)
        for name, _side, verts, faces in meshes:
            _draw_mesh_fill_projection(ax, verts, faces, panel, pair_colour[name],
                                       MAX_TRIS_MUSCLE_PROJ, rng, alpha=1.0, zorder=2)
    if skin is not None:
        _draw_mesh_fill_3d(ax_3d, skin[0], skin[1], NATURE_PALETTE["skin"],
                           MAX_TRIS_SKIN_3D, rng, alpha=0.16)
    for name, _side, verts, faces in meshes:
        _draw_mesh_fill_3d(ax_3d, verts, faces, pair_colour[name],
                           MAX_TRIS_MUSCLE_3D, rng, alpha=1.0)
    _set_limits(axes, all_pts, margin=view_margin)
    for ax, lbl in zip(axes, "abcd", strict=True):
        add_panel_label(ax, lbl)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="none", markersize=8,
                   markerfacecolor=pair_colour[n], markeredgecolor="none", label=n)
        for n in pair_names
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5,
              framealpha=0.9, bbox_to_anchor=(0.5, 0.005))

    out = out_path or (cfg.outputs.base / "muscle_sources_pairs.png")
    return save_figure(fig, out, dpi=dpi)


def render_muscle_source_orientations(
    cfg: Config, *, spacing_mm: float = 15.0, out_path: Path | None = None,
    dpi: int = 150,
) -> Path:
    """Selected muscle source dipoles with their fibre-axis orientation.

    Plots exactly the source set a muscle forward solve samples (not raw tet
    centroids), each with a quiver arrow along its per-muscle fibre-direction
    proxy (:func:`inob.sources.muscle.muscle_source_orientations`).
    """
    apply_nature_style()
    rng = np.random.default_rng(cfg.reproducibility.seed)

    fem, positions, orientations = _load_muscle_sources(cfg, spacing_mm=spacing_mm)

    # Arrow length: a visible fraction of the sampling spacing so neighbouring
    # arrows don't overlap into an unreadable mass.
    arrow_len = 0.6 * spacing_mm
    arrow_colour = NATURE_PALETTE["glow"]   # "source / focal element" accent

    backdrop = _muscle_backdrop(cfg)
    fig = plt.figure(figsize=(13, 11))
    fig.suptitle(
        f"Muscle source dipoles — fibre orientation — {len(positions)} sources, "
        f"{spacing_mm:.0f} mm spacing",
        fontsize=11, fontweight="bold", y=0.985,
    )
    axes = _panel_axes(fig)
    ax_axial, ax_lateral, ax_coronal, ax_3d = axes
    for ax, key in ((ax_axial, "axial"), (ax_lateral, "lateral"), (ax_coronal, "coronal")):
        panel = _PANELS[key]
        _draw_backdrop_projection(ax, backdrop, panel, rng)
        p2 = panel.project(positions)
        o2 = panel.project(orientations)   # a direction transforms the same linear way
        ax.quiver(p2[:, 0], p2[:, 1], o2[:, 0], o2[:, 1],
                  color=arrow_colour, scale=1.0 / arrow_len, scale_units="xy",
                  angles="xy", width=0.003, alpha=0.9, zorder=2)
    _draw_backdrop_3d(ax_3d, backdrop, rng)
    N = orientations * arrow_len
    ax_3d.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
                N[:, 0], N[:, 1], N[:, 2],
                color=arrow_colour, linewidth=0.7, arrow_length_ratio=0.3)
    all_pts = positions if backdrop is None else np.vstack([positions, backdrop[0]])
    _set_limits(axes, all_pts)
    for ax, lbl in zip(axes, "abcd", strict=True):
        add_panel_label(ax, lbl)

    handles = [plt.Line2D([0], [0], color=arrow_colour, lw=2, label="Fibre axis")]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8,
              framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    out = out_path or (cfg.outputs.base / "muscle_sources_orientations.png")
    return save_figure(fig, out, dpi=dpi)
