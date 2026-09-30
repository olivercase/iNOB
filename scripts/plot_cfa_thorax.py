#!/usr/bin/env python3
"""Figures for the head+thorax conductor: slices, 3D renders, and the old/new comparison.

    python scripts/plot_cfa_thorax.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# compartment -> (colour, z-order, label)
STYLE = {
    "skin":        ("#9aa3ad", 1, "skin"),
    "bone":        ("#e8e2d4", 2, "bone"),
    "lung_left":   ("#7fb2d9", 3, "lung (L)"),
    "lung_right":  ("#7fb2d9", 3, "lung (R)"),
    "spinal_cord": ("#b58fd6", 4, "spinal cord"),
    "heart":       ("#c8323c", 5, "heart"),
}


def outline(m: trimesh.Trimesh, axis: int, coord: float) -> list[np.ndarray]:
    """Cross-section polylines of ``m`` at ``x[axis] == coord``, in the other two axes."""
    n = np.zeros(3)
    n[axis] = 1.0
    sec = m.section(plane_origin=n * coord, plane_normal=n)
    if sec is None:
        return []
    keep = [i for i in range(3) if i != axis]
    V = np.asarray(sec.vertices)[:, keep]
    return [V[e.points] for e in sec.entities if len(e.points) > 1]


def slice_panel(ax, meshes, axis, coord, xlab, ylab, *, sensors=None, marks=()):
    for name, m in meshes.items():
        if name not in STYLE or len(m.faces) == 0:
            continue
        col, z, _ = STYLE[name]
        lw = 1.4 if name in ("heart", "skin") else 0.7
        for pl in outline(m, axis, coord):
            ax.plot(pl[:, 0], pl[:, 1], color=col, lw=lw, zorder=z,
                    solid_capstyle="round")
    if sensors is not None:
        keep = [i for i in range(3) if i != axis]
        ax.scatter(sensors[:, keep[0]], sensors[:, keep[1]], s=6, c="#1b7f3b",
                   zorder=20, label="OPM (85)")
    for pos, col, lab, mk in marks:
        keep = [i for i in range(3) if i != axis]
        ax.scatter([pos[keep[0]]], [pos[keep[1]]], s=90, marker=mk,
                   facecolor="none", edgecolor=col, lw=2.0, zorder=30, label=lab)
    ax.set_aspect("equal")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/cfa_thorax")
    p.add_argument("--ref", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/_cfa_ref.npz")
    p.add_argument("--dpi", type=int, default=170)
    a = p.parse_args(argv)

    meshes = {}
    for f in sorted(a.dir.glob("*.stl")):
        meshes[f.stem] = trimesh.load(f, process=False)
    if not meshes:
        raise SystemExit(f"no STLs in {a.dir} — run build_cfa_thorax.py first")

    ref = np.load(a.ref)
    sens = ref["chanpos"] * 1000.0
    pack_heart = ref["bnd0_pos"] * 1000.0
    pack_body = ref["bnd2_pos"] * 1000.0
    src = ref["src"] * 1000.0
    inob_heart_c = np.asarray(meshes["heart"].centroid)
    pack_heart_c = pack_heart.mean(0)

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    slice_panel(ax, meshes, 0, 0.0, "y  anterior +  (mm)", "z  superior +  (mm)",
                sensors=sens,
                marks=[(inob_heart_c, "#c8323c", "iNOB heart centroid", "o")])
    ax.set_title("sagittal  x = 0", loc="left", fontsize=11)

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    slice_panel(ax2, meshes, 1, float(inob_heart_c[1]), "x  right +  (mm)",
                "z  superior +  (mm)", sensors=sens)
    ax2.set_title(f"coronal  y = {inob_heart_c[1]:.0f}", loc="left", fontsize=11)

    ax3 = fig.add_subplot(gs[0, 2], sharey=ax)
    for pl in outline(meshes["skin"], 0, 0.0):
        ax3.plot(pl[:, 0], pl[:, 1], color="#9aa3ad", lw=1.3, zorder=1)
    for pl in outline(meshes["heart"], 0, 0.0):
        ax3.plot(pl[:, 0], pl[:, 1], color="#c8323c", lw=1.8, zorder=5)
    ph = trimesh.points.PointCloud(pack_heart)
    ax3.scatter(pack_heart[:, 1], pack_heart[:, 2], s=1.5, c="#d98f00", zorder=4)
    ax3.scatter(pack_body[:, 1], pack_body[:, 2], s=1.0, c="#c9b27a", zorder=2, alpha=0.5)
    ax3.scatter(sens[:, 1], sens[:, 2], s=6, c="#1b7f3b", zorder=20)
    for c, col, lab in [(inob_heart_c, "#c8323c", "iNOB"), (pack_heart_c, "#d98f00", "pack")]:
        ax3.scatter([c[1]], [c[2]], s=110, marker="o", facecolor="none",
                    edgecolor=col, lw=2.2, zorder=30)
    ax3.annotate("", xy=(pack_heart_c[1], pack_heart_c[2]),
                 xytext=(inob_heart_c[1], inob_heart_c[2]),
                 arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    d = pack_heart_c - inob_heart_c
    ax3.text(inob_heart_c[1] + 8, (inob_heart_c[2] + pack_heart_c[2]) / 2,
             f"Δ = {np.linalg.norm(d):.0f} mm\n(Δz {d[2]:+.0f}, Δy {d[1]:+.0f})",
             fontsize=9.5, va="center")
    ax3.set_aspect("equal"); ax3.set_xlabel("y  anterior +  (mm)")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_title("iNOB vs CFA pack (torso2mni)", loc="left", fontsize=11)

    handles = [Line2D([], [], color=c, lw=2.5, label=l)
               for c, _, l in dict.fromkeys(
                   [(v[0], v[1], v[2]) for k, v in STYLE.items() if k in meshes])]
    handles += [Line2D([], [], marker="o", ls="", mfc="#1b7f3b", mec="#1b7f3b",
                       ms=5, label="OPM sensors (85)"),
                Line2D([], [], marker="o", ls="", mfc="none", mec="#c8323c",
                       mew=2, ms=9, label="heart centroid (iNOB)"),
                Line2D([], [], marker="o", ls="", mfc="none", mec="#d98f00",
                       mew=2, ms=9, label="heart centroid (CFA pack)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9.5)
    fig.suptitle("Head + thorax volume conductor from BodyParts3D 4.3, rigidly "
                 "registered to the MNI OPM array", fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    out = a.dir / "thorax_slices.png"
    fig.savefig(out, dpi=a.dpi)
    print("figure:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
