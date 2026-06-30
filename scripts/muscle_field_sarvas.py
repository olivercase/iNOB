#!/usr/bin/env python3
"""Analytic (Sarvas) magnetic-field map for muscle-group dipoles,
painted onto the torso skin surface — a 3-D topoplot.

Stand-in for the full DUNEuro FEM solve while DUNEuro is (re)built on the
cluster. Implements Gareth Barnes' suggestion: one current dipole at the centre
of each scalene muscle, pointing along the muscle's long axis. All orthogonal components of the
magnetic field (B·n̂, what an OPM on the skin senses) is evaluated directly at
each skin-surface vertex via iNOB's own
``inob.analysis.analytic_sphere.sarvas_meg_field`` — no sensor interpolation,
the analytic field is exact at every vertex. Per-source sphere centre sits on
the local neck axis at the source's own z (moving-sphere convention,
inob.analysis.sarvas_compare).

Outputs:
  outputs/muscle_skin_topoplot.png        radial component
  outputs/muscle_skin_topoplot_tang1.png  superior-inferior tangential
  outputs/muscle_skin_topoplot_tang2.png  azimuthal tangential
  outputs/muscle_skin_topoplot_vectors.png  measurement-vector directions
  outputs/muscle_skin_topoplot.json       peak values for all three
Units: fT per 1 nA·m of dipole moment (iNOB leadfield convention).
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from inob.analysis.analytic_sphere import sarvas_meg_field
from inob.io.hdf5 import load_geometry
from inob.viz.style import apply_nature_style, divergent_cmap, NATURE_PALETTE
from inob.viz.surface_topoplot import (
    _crop_skin_to_band,
    _cylindrical_unroll,
    _draw_skin_coloured,
)

ROOT = Path(__file__).resolve().parents[1]
GEOM = ROOT / "outputs/geometry/vagus_geometry.mat"
MUSCLE_DIR = ROOT / "data/muscle"
OUT_PNG         = ROOT / "outputs/muscle_skin_topoplot.png"
OUT_PNG_TANG1   = ROOT / "outputs/muscle_skin_topoplot_tang1.png"
OUT_PNG_TANG2   = ROOT / "outputs/muscle_skin_topoplot_tang2.png"
OUT_PNG_VECTORS = ROOT / "outputs/muscle_skin_topoplot_vectors.png"
OUT_JSON        = ROOT / "outputs/muscle_skin_topoplot.json"

DEFAULT_PATTERNS = [
    "scalenus anter",
    "scalenus medi",
    "omohyoid",
    "sternocleidomastoid",
    "sternothyroid",
    "longus capitis",
]

Q_NAM = 1.0           # report as leadfield: fT per nA·m
SKIN_MARGIN_MM = 80.0 # skin band extends this far beyond the outermost source in z


def muscle_sources(patterns: list[str]):
    """One dipole per matching muscle STL: centroid + PCA long axis (z-up)."""
    srcs, axes, names = [], [], []
    for pat in patterns:
        for f in sorted(glob.glob(str(MUSCLE_DIR / f"*{pat}*.stl"))):
            v = np.asarray(trimesh.load(f, process=False).vertices)
            c = v.mean(0)
            ax = np.linalg.svd(v - c, full_matrices=False)[2][0]
            if ax[2] < 0:
                ax = -ax  # consistent superior orientation
            srcs.append(c)
            axes.append(ax)
            names.append(Path(f).stem.split("FMA")[-1].lstrip("0123456789_ ").strip())
    if not srcs:
        sys.exit(f"no muscle STLs matched {patterns} in {MUSCLE_DIR}")
    return np.array(srcs), np.array(axes), names


def body_axis_xy(verts: np.ndarray, z: float, band: float = 60.0) -> np.ndarray:
    """Mean XY of skin vertices within ±band mm of z — local body-axis centre."""
    sel = np.abs(verts[:, 2] - z) <= band
    if sel.sum() < 8:
        sel = np.ones(len(verts), dtype=bool)
    return verts[sel, :2].mean(axis=0)


def sarvas_field_on_surface(src_mm, axis_mm, pts_mm, centre_mm) -> np.ndarray:
    """Full B vector (M, 3) in Tesla at each surface point for one axial dipole."""
    r0 = (np.asarray(src_mm) - centre_mm) * 1e-3
    q = np.asarray(axis_mm, float)
    q = q / max(np.linalg.norm(q), 1e-12) * (Q_NAM * 1e-9)
    pts = (pts_mm - centre_mm) * 1e-3
    return sarvas_meg_field(r0, q, pts)  # (M, 3) T


def tangential_frame(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute two orthonormal tangential vectors per vertex.

    Returns t1 (superior–inferior) and t2 (azimuthal) such that
    {t1, t2, n̂} form a right-handed orthonormal frame at each vertex.
    """
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    bad = (norms < 1e-12).ravel()
    norms = np.where(norms < 1e-12, 1.0, norms)
    n = normals / norms
    n[bad] = [0., 0., 1.]           # fallback for degenerate vertices

    ref = np.zeros_like(n)
    use_z = np.abs(n[:, 2]) < 0.9   # use [0,0,1] unless nearly parallel
    ref[use_z] = [0., 0., 1.]
    ref[~use_z] = [1., 0., 0.]

    t1 = ref - (ref * n).sum(axis=1, keepdims=True) * n
    t1_norms = np.linalg.norm(t1, axis=1, keepdims=True)
    t1 /= np.where(t1_norms < 1e-12, 1.0, t1_norms)

    t2 = np.cross(n, t1)
    return t1, t2


def _band_plot_params(band_v: np.ndarray, z_lo: float, z_hi: float) -> dict:
    """Shared 3-D axis limits and box-aspect ratio derived from the band geometry.

    box_aspect uses the actual data ranges in mm so the rendered geometry has
    correct proportions regardless of which body region is being plotted.
    """
    x_lo, x_hi = band_v[:, 0].min(), band_v[:, 0].max()
    y_lo, y_hi = band_v[:, 1].min(), band_v[:, 1].max()
    return {
        "xlim": (x_lo, x_hi),
        "ylim": (y_lo, y_hi),
        "zlim": (z_lo, z_hi),
        "box_aspect": (x_hi - x_lo, y_hi - y_lo, z_hi - z_lo),
    }


def _make_figure(
    path: Path,
    combined: np.ndarray,
    per: list[np.ndarray],
    component_label: str,
    names: list[str],
    srcs: np.ndarray,
    band_v: np.ndarray,
    band_f: np.ndarray,
    z_lo: float,
    z_hi: float,
    axis_xy: tuple[float, float],
) -> None:
    """Render and save one topoplot figure for a single field component."""
    cmap = divergent_cmap()
    lim = float(np.percentile(np.abs(combined), 99)) or 1.0
    pp = _band_plot_params(band_v, z_lo, z_hi)

    fig = plt.figure(figsize=(20, 14))
    outer = GridSpec(2, 1, figure=fig,
                     left=0.03, right=0.91, top=0.88, bottom=0.08,
                     hspace=0.25, height_ratios=[1.35, 1.0])
    gs_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.05)
    gs_bot = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.50)

    n_dip = len(srcs)
    fig.suptitle(
        f"Muscle field on the skin surface ({n_dip} dipole(s), Sarvas analytic,"
        f" {component_label})\n"
        "one dipole at each muscle centre along its long axis · fT per nA·m",
        fontsize=13, fontweight="bold", y=0.95,
    )

    views = [("anterior", 12, -90), ("left lateral", 8, 0), ("posterior", 12, 90)]
    ax3d = []
    for i, (vname, elev, azim) in enumerate(views):
        ax = fig.add_subplot(gs_top[0, i], projection="3d")
        _draw_skin_coloured(ax, band_v, band_f, combined, vmin=-lim, vmax=lim,
                            cmap=cmap, alpha=0.97)
        ax.scatter(srcs[:, 0], srcs[:, 1], srcs[:, 2], s=70,
                   c=NATURE_PALETTE["glow"], edgecolor=NATURE_PALETTE["axis"],
                   linewidths=0.6, marker="*")
        ax.set_xlim(*pp["xlim"])
        ax.set_ylim(*pp["ylim"])
        ax.set_zlim(*pp["zlim"])
        ax.set_box_aspect(pp["box_aspect"])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{n_dip} dipoles · {vname}", fontsize=10, pad=10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zlabel("z (mm)", fontsize=9, labelpad=6)
        ax.tick_params(axis="z", labelsize=8)
        ax3d.append(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-lim, lim))
    cb = fig.colorbar(sm, ax=ax3d, shrink=0.55, pad=0.06, fraction=0.02, aspect=25)
    cb.set_label(f"{component_label}  fT / nA·m", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    ax_u = fig.add_subplot(gs_bot[0, :2])
    th, zz = _cylindrical_unroll(band_v, axis_xy=axis_xy)
    sc = ax_u.scatter(th, zz, c=combined, cmap=cmap, vmin=-lim, vmax=lim, s=8)
    sth, sz = _cylindrical_unroll(srcs, axis_xy=axis_xy)
    ax_u.scatter(sth, sz, s=140, c=NATURE_PALETTE["glow"],
                 edgecolor=NATURE_PALETTE["axis"], linewidths=0.8, marker="*")
    ax_u.set_xlabel("azimuth θ around body axis (°)", fontsize=10)
    ax_u.set_ylabel("z (mm)", fontsize=10)
    ax_u.set_xlim(-180, 180)
    ax_u.set_xticks(np.arange(-180, 181, 60))
    ax_u.tick_params(labelsize=9)
    ax_u.set_title(f"combined {component_label} · unrolled skin cylinder",
                   fontsize=10, pad=8)
    cb2 = fig.colorbar(sc, ax=ax_u, shrink=0.90, pad=0.02, fraction=0.03, aspect=25)
    cb2.set_label("fT / nA·m", fontsize=9)
    cb2.ax.tick_params(labelsize=9)

    ax_b = fig.add_subplot(gs_bot[0, 2])
    short = [n.replace("scalenus", "scal.") for n in names]
    peaks = [float(np.max(np.abs(v))) for v in per]
    ax_b.barh(range(len(names)), peaks, color=NATURE_PALETTE.get("blue", "#3b6fb0"))
    ax_b.set_yticks(range(len(names)))
    ax_b.set_yticklabels(short, fontsize=9)
    ax_b.invert_yaxis()
    ax_b.set_xlabel(f"peak |{component_label}| on skin  fT / nA·m", fontsize=9)
    ax_b.tick_params(labelsize=9)
    ax_b.set_title("per-muscle peak", fontsize=10, pad=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _make_vector_figure(
    path: Path,
    band_v: np.ndarray,
    band_f: np.ndarray,
    band_n: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    srcs: np.ndarray,
    z_lo: float,
    z_hi: float,
) -> None:
    """3×3 figure: one row per measurement component, one column per view.

    Each panel shows the skin surface in translucent grey with quiver arrows
    indicating the OPM sensitive-axis direction, sampled uniformly from the
    source band (which already covers the region of interest).
    """
    components = [
        ("radial  B·n̂\n(outward normal)", band_n, "#3b6fb0"),
        ("tangential  B·t₁\n(superior–inferior)", t1, "#c0392b"),
        ("tangential  B·t₂\n(azimuthal)", t2, "#27ae60"),
    ]
    views = [("anterior", 12, -90), ("left lateral", 8, 0), ("posterior", 12, 90)]

    # Uniform sub-sample across the entire band — the band is already the ROI.
    rng = np.random.default_rng(42)
    n_q = min(500, len(band_v))
    idx = rng.choice(len(band_v), n_q, replace=False)
    qv = band_v[idx]

    pp = _band_plot_params(band_v, z_lo, z_hi)
    arrow_len = (z_hi - z_lo) * 0.045  # ~4.5% of band height

    fig = plt.figure(figsize=(18, 18))
    fig.suptitle(
        "OPM measurement-vector directions on the skin surface\n"
        "rows: radial n̂  |  superior–inferior t₁  |  azimuthal t₂"
        "        (arrows sub-sampled uniformly over source band, star = dipole centre)",
        fontsize=13, fontweight="bold", y=1.00,
    )

    tris = band_v[band_f]

    for row, (label, vec, color) in enumerate(components):
        qu = vec[idx]
        for col, (vname, elev, azim) in enumerate(views):
            ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection="3d")

            poly = Poly3DCollection(tris, alpha=0.18, linewidth=0)
            poly.set_facecolor("#b0b0b0")
            poly.set_edgecolor("none")
            ax.add_collection3d(poly)

            ax.quiver(
                qv[:, 0], qv[:, 1], qv[:, 2],
                qu[:, 0], qu[:, 1], qu[:, 2],
                length=arrow_len, normalize=True,
                color=color, alpha=0.75, linewidth=0.7,
                arrow_length_ratio=0.25,
            )

            ax.scatter(
                srcs[:, 0], srcs[:, 1], srcs[:, 2], s=60,
                c=NATURE_PALETTE["glow"], edgecolor=NATURE_PALETTE["axis"],
                linewidths=0.6, marker="*", zorder=5,
            )

            ax.set_xlim(*pp["xlim"])
            ax.set_ylim(*pp["ylim"])
            ax.set_zlim(*pp["zlim"])
            ax.set_box_aspect(pp["box_aspect"])
            ax.view_init(elev=elev, azim=azim)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zlabel("z (mm)", fontsize=8, labelpad=4)
            ax.tick_params(axis="z", labelsize=7)

            if row == 0:
                ax.set_title(vname, fontsize=11, pad=10)
            if col == 0:
                ax.text2D(
                    -0.18, 0.5, label,
                    transform=ax.transAxes,
                    fontsize=10, color=color, fontweight="bold",
                    va="center", ha="left", rotation=90,
                )

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    apply_nature_style()
    geom = load_geometry(GEOM)
    skin_comp = geom.compartments["mesh_skin"]
    skin = trimesh.Trimesh(skin_comp.vertices, skin_comp.faces, process=False)

    patterns = sys.argv[1:] or DEFAULT_PATTERNS
    srcs, axes, names = muscle_sources(patterns)
    print(f"{len(srcs)} muscle dipole(s): {', '.join(names)}")

    # Atomated for other muscle groups(ish)
    # Skin band: spans the source z-extent plus SKIN_MARGIN_MM on each side.
    # All figure limits are derived from this, so the ROI follows the sources.
    z_band_lo = float(srcs[:, 2].min()) - SKIN_MARGIN_MM
    z_band_hi = float(srcs[:, 2].max()) + SKIN_MARGIN_MM
    band_v, band_f = _crop_skin_to_band(skin, z_band_lo, z_band_hi)
    if len(band_v) == 0:
        raise RuntimeError("no skin vertices in source band — check muscle STL paths")

    band_mesh = trimesh.Trimesh(band_v, band_f, process=False)
    band_n = np.asarray(band_mesh.vertex_normals)
    t1, t2 = tangential_frame(band_n)
    axis_xy = (float(band_v[:, 0].mean()), float(band_v[:, 1].mean()))

    # Full B field per source (n_srcs, M, 3), converted to fT/nA·m.
    B_per = []
    for src, ax in zip(srcs, axes):
        centre = np.array([*body_axis_xy(skin.vertices, src[2]), src[2]])
        B = sarvas_field_on_surface(src, ax, band_v, centre) * 1e15 / Q_NAM
        B_per.append(B)
    B_per = np.array(B_per)          # (n_srcs, M, 3)
    B_combined = B_per.sum(axis=0)   # (M, 3)

    def proj(B_arr, basis):
        return np.einsum("ij,ij->i", B_arr, basis)

    per_radial = [proj(B, band_n) for B in B_per]
    per_tang1  = [proj(B, t1)     for B in B_per]
    per_tang2  = [proj(B, t2)     for B in B_per]

    combined_radial = proj(B_combined, band_n)
    combined_tang1  = proj(B_combined, t1)
    combined_tang2  = proj(B_combined, t2)

    fig_kwargs = dict(names=names, srcs=srcs, band_v=band_v, band_f=band_f,
                      z_lo=z_band_lo, z_hi=z_band_hi, axis_xy=axis_xy)
    _make_figure(OUT_PNG,       combined_radial, per_radial, "radial B·n̂",       **fig_kwargs)
    _make_figure(OUT_PNG_TANG1, combined_tang1,  per_tang1,  "tangential B·t₁ (superior–inferior)", **fig_kwargs)
    _make_figure(OUT_PNG_TANG2, combined_tang2,  per_tang2,  "tangential B·t₂ (azimuthal)",         **fig_kwargs)
    _make_vector_figure(OUT_PNG_VECTORS, band_v, band_f, band_n, t1, t2,
                        srcs, z_band_lo, z_band_hi)

    stats = []
    for nm, src, r, g1, g2 in zip(names, srcs, per_radial, per_tang1, per_tang2):
        stats.append({
            "muscle": nm, "source_mm": src.tolist(),
            "peak_abs_fT_per_nAm": {
                "radial": float(np.max(np.abs(r))),
                "tang1":  float(np.max(np.abs(g1))),
                "tang2":  float(np.max(np.abs(g2))),
            },
        })

    OUT_JSON.write_text(json.dumps({
        "Q_nAm": Q_NAM,
        "z_band_mm": {"lo": z_band_lo, "hi": z_band_hi,
                      "skin_margin": SKIN_MARGIN_MM},
        "n_skin_vertices": int(len(band_v)),
        "per_muscle": stats,
        "combined_peak_fT_per_nAm": {
            "radial": float(np.max(np.abs(combined_radial))),
            "tang1":  float(np.max(np.abs(combined_tang1))),
            "tang2":  float(np.max(np.abs(combined_tang2))),
        },
    }, indent=2))

    print(f"figures: {OUT_PNG.name}  {OUT_PNG_TANG1.name}  "
          f"{OUT_PNG_TANG2.name}  {OUT_PNG_VECTORS.name}")
    print(f"  band z: {z_band_lo:.0f} → {z_band_hi:.0f} mm  "
          f"({int(len(band_v))} skin vertices)")
    header = f"  {'muscle':<28} {'radial':>10} {'tang1':>10} {'tang2':>10}  fT/nA·m"
    print(header)
    for s in stats:
        p = s["peak_abs_fT_per_nAm"]
        print(f"  {s['muscle']:<28} {p['radial']:10.2f} {p['tang1']:10.2f} {p['tang2']:10.2f}")
    cr, ct1, ct2 = (np.max(np.abs(combined_radial)), np.max(np.abs(combined_tang1)),
                    np.max(np.abs(combined_tang2)))
    print(f"  {'combined':<28} {cr:10.2f} {ct1:10.2f} {ct2:10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
