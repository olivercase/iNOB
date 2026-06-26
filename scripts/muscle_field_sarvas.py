#!/usr/bin/env python3
"""Analytic (Sarvas) magnetic-field map for neck-muscle dipoles around C7,
painted onto the torso skin surface — a 3-D topoplot.

Stand-in for the full DUNEuro FEM solve while DUNEuro is (re)built on the
cluster. Implements Gareth Barnes' suggestion: one current dipole at the centre
of each scalene muscle, pointing along the muscle's long axis. The *radial*
magnetic field (B·n̂, what an OPM on the skin senses) is evaluated directly at
each skin-surface vertex via iNOB's own
``inob.analysis.analytic_sphere.sarvas_meg_field`` — no sensor interpolation,
the analytic field is exact at every vertex. Per-source sphere centre sits on
the local neck axis at the source's own z (moving-sphere convention,
inob.analysis.sarvas_compare).

Output: outputs/muscle_skin_topoplot.png
        outputs/muscle_skin_topoplot.json
Units : fT per 1 nA·m of dipole moment (iNOB leadfield convention).
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
from matplotlib.gridspec import GridSpec

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
OUT_PNG = ROOT / "outputs/muscle_skin_topoplot.png"
OUT_JSON = ROOT / "outputs/muscle_skin_topoplot.json"

# Default target muscles: the scalene group around C7. Override on the command
# line with substrings, e.g. `python scripts/muscle_field_sarvas.py sternocleido`.
DEFAULT_PATTERNS = ["scalenus anter", "scalenus medi"]

Q_NAM = 1.0          # report as leadfield: fT per nA·m
Z_BAND_MM = 170.0    # cervical slab half-height around the source z


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


def neck_axis_xy(verts: np.ndarray, z: float, band: float = 60.0) -> np.ndarray:
    sel = np.abs(verts[:, 2] - z) <= band
    if sel.sum() < 8:
        sel = np.ones(len(verts), dtype=bool)
    return verts[sel, :2].mean(axis=0)


def sarvas_radial_on_surface(src_mm, axis_mm, pts_mm, normals, centre_mm) -> np.ndarray:
    """B·n̂ at each surface point (Tesla) for one axial dipole."""
    r0 = (np.asarray(src_mm) - centre_mm) * 1e-3
    q = np.asarray(axis_mm, float)
    q = q / max(np.linalg.norm(q), 1e-12) * (Q_NAM * 1e-9)
    pts = (pts_mm - centre_mm) * 1e-3
    B = sarvas_meg_field(r0, q, pts)              # (M,3) T
    return np.einsum("ij,ij->i", B, normals)      # project on outward normal


def main() -> int:
    apply_nature_style()
    geom = load_geometry(GEOM)
    skin_comp = geom.compartments["mesh_skin"]
    skin = trimesh.Trimesh(skin_comp.vertices, skin_comp.faces, process=False)

    patterns = sys.argv[1:] or DEFAULT_PATTERNS
    srcs, axes, names = muscle_sources(patterns)
    print(f"{len(srcs)} muscle dipole(s): {', '.join(names)}")

    # Crop skin to the cervical band around the sources for the headline view.
    z0 = float(srcs[:, 2].mean())
    band_v, band_f = _crop_skin_to_band(skin, z0 - Z_BAND_MM, z0 + Z_BAND_MM)
    if len(band_v) == 0:
        raise RuntimeError("no skin vertices in cervical band")
    # Outward vertex normals on the cropped patch.
    band_mesh = trimesh.Trimesh(band_v, band_f, process=False)
    band_n = np.asarray(band_mesh.vertex_normals)

    # Radial field at each skin vertex, per source (fT per nA·m), and combined.
    per = []
    stats = []
    for src, ax, nm in zip(srcs, axes, names):
        centre = np.array([*neck_axis_xy(skin.vertices, src[2]), src[2]])
        v = sarvas_radial_on_surface(src, ax, band_v, band_n, centre) * 1e15 / Q_NAM
        per.append(v)
        stats.append({"muscle": nm, "source_mm": src.tolist(),
                      "peak_abs_fT_per_nAm": float(np.max(np.abs(v)))})
    per = np.array(per)
    combined = per.sum(axis=0)

    cmap = divergent_cmap()
    axis_xy = (float(band_v[:, 0].mean()), float(band_v[:, 1].mean()))

    fig = plt.figure(figsize=(16, 10.5))
    gs = GridSpec(2, 3, figure=fig, left=0.02, right=0.98, top=0.9, bottom=0.06,
                  hspace=0.18, wspace=0.12)
    fig.suptitle(
        f"Neck-muscle field on the torso skin ({len(srcs)} dipole(s), Sarvas analytic, radial B·n̂)\n"
        "one dipole at each muscle centre along its long axis · fT per nA·m",
        fontsize=12, fontweight="bold", y=0.975)

    # Row 1: combined field, three views (anterior, left-lateral, posterior).
    views = [("anterior", 12, -90), ("left lateral", 8, 0), ("posterior", 12, 90)]
    lim = float(np.percentile(np.abs(combined), 99)) or 1.0
    for i, (vname, elev, azim) in enumerate(views):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        _draw_skin_coloured(ax, band_v, band_f, combined, vmin=-lim, vmax=lim,
                            cmap=cmap, alpha=0.97)
        ax.scatter(srcs[:, 0], srcs[:, 1], srcs[:, 2], s=70, c=NATURE_PALETTE["glow"],
                   edgecolor=NATURE_PALETTE["axis"], linewidths=0.6, marker="*")
        ax.set_xlim(band_v[:, 0].min(), band_v[:, 0].max())
        ax.set_ylim(band_v[:, 1].min(), band_v[:, 1].max())
        ax.set_zlim(z0 - Z_BAND_MM, z0 + Z_BAND_MM)
        ax.set_box_aspect((1, 1, 1.6))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{len(srcs)} dipoles · {vname}", fontsize=9)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_zlabel("z (mm)", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-lim, lim))
    cb = fig.colorbar(sm, ax=fig.axes[:3], shrink=0.5, pad=0.01, fraction=0.025)
    cb.set_label("radial field  fT / nA·m", fontsize=9)

    # Row 2 left+mid: unrolled cylinder (θ vs z) of the combined skin field.
    ax_u = fig.add_subplot(gs[1, :2])
    th, zz = _cylindrical_unroll(band_v, axis_xy=axis_xy)
    sc = ax_u.scatter(th, zz, c=combined, cmap=cmap, vmin=-lim, vmax=lim, s=8)
    sth, sz = _cylindrical_unroll(srcs, axis_xy=axis_xy)
    ax_u.scatter(sth, sz, s=140, c=NATURE_PALETTE["glow"],
                 edgecolor=NATURE_PALETTE["axis"], linewidths=0.8, marker="*")
    ax_u.set_xlabel("azimuth θ around neck axis (°)")
    ax_u.set_ylabel("z (mm)")
    ax_u.set_xlim(-180, 180); ax_u.set_xticks(np.arange(-180, 181, 60))
    ax_u.set_title("combined field · unrolled skin cylinder", fontsize=9)
    cb2 = fig.colorbar(sc, ax=ax_u, shrink=0.85, pad=0.02, fraction=0.04)
    cb2.set_label("fT / nA·m", fontsize=8)

    # Row 2 right: per-muscle peak bar.
    ax_b = fig.add_subplot(gs[1, 2])
    short = [n.replace("scalenus", "scal.") for n in names]
    ax_b.barh(range(len(names)), [s["peak_abs_fT_per_nAm"] for s in stats],
              color=NATURE_PALETTE.get("blue", "#3b6fb0"))
    ax_b.set_yticks(range(len(names))); ax_b.set_yticklabels(short, fontsize=8)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("peak |B·n̂| on skin  fT / nA·m", fontsize=8)
    ax_b.set_title("per-muscle peak", fontsize=9)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    OUT_JSON.write_text(json.dumps(
        {"Q_nAm": Q_NAM, "z_band_mm": Z_BAND_MM, "n_skin_vertices": int(len(band_v)),
         "per_muscle": stats,
         "combined_peak_fT_per_nAm": float(np.max(np.abs(combined)))}, indent=2))
    print(f"figure: {OUT_PNG}")
    for s in stats:
        print(f"  {s['muscle']:24s} peak |B·n̂| {s['peak_abs_fT_per_nAm']:6.2f} fT/nA·m")
    print(f"  {'combined':24s} peak |B·n̂| {np.max(np.abs(combined)):6.2f} fT/nA·m  "
          f"({int(len(band_v))} skin vertices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
