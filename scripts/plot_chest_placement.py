#!/usr/bin/env python3
"""Where the real 8-site chest array sits on the iNOB torso, and what is assumed.

    python scripts/plot_chest_placement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

INK, MUTED = "#0b0b0b", "#52514e"
COL = {"skin": "#9aa3ad", "heart": "#c8323c", "lung_left": "#7fb2d9",
       "lung_right": "#7fb2d9", "bone": "#ded8c8"}


def outline(m, axis, coord):
    n = np.zeros(3); n[axis] = 1.0
    sec = m.section(plane_origin=n * coord, plane_normal=n)
    if sec is None:
        return []
    keep = [i for i in range(3) if i != axis]
    V = np.asarray(sec.vertices)[:, keep]
    return [V[e.points] for e in sec.entities if len(e.points) > 1]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    d = root / "outputs/cfa_thorax"
    meshes = {p.stem: trimesh.load(p, process=True) for p in sorted(d.glob("*.stl"))}
    z = np.load(root / "outputs/chest/chest_array_placed.npz")
    chest = z["pos"]
    head = np.load(root / "outputs/_cfa_ref.npz")["chanpos"] * 1000.0
    hc = np.asarray(meshes["heart"].centroid)

    views = [(0, float(chest[:, 0].mean()), "sagittal  x = %.0f" % chest[:, 0].mean(),
              "y  anterior +", "z  superior +"),
             (2, float(hc[2]), "axial  z = %.0f  (heart level)" % hc[2],
              "x  right +", "y  anterior +"),
             (1, float(hc[1]), "coronal  y = %.0f" % hc[1], "x  right +", "z  superior +")]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.4))
    for ax, (axis, coord, title, xl, yl) in zip(axes, views):
        for name in ("skin", "bone", "lung_left", "lung_right", "heart"):
            m = meshes.get(name)
            if m is None:
                continue
            lw = 1.6 if name in ("skin", "heart") else 0.6
            for pl in outline(m, axis, coord):
                ax.plot(pl[:, 0], pl[:, 1], color=COL[name], lw=lw, zorder=3)
        keep = [i for i in range(3) if i != axis]
        ax.scatter(head[:, keep[0]], head[:, keep[1]], s=7, c="#1b7f3b", zorder=6)
        ax.scatter(chest[:, keep[0]], chest[:, keep[1]], s=92, marker="s",
                   facecolor="#eb6834", edgecolor="white", linewidth=1.2, zorder=8)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=11.5, loc="left", color=INK)
        ax.set_xlabel(xl, fontsize=9.5, color=MUTED)
        ax.set_ylabel(yl, fontsize=9.5, color=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.5, colors=MUTED)
    if len(chest):
        axes[0].set_ylim(-500, 150)
    h = [Line2D([], [], marker="s", ls="", mfc="#eb6834", mec="white", ms=10,
                label="real chest array — 8 sites × 3 axes"),
         Line2D([], [], marker="o", ls="", mfc="#1b7f3b", mec="#1b7f3b", ms=6,
                label="head array (85, for context)"),
         Line2D([], [], color=COL["heart"], lw=2.2, label="heart"),
         Line2D([], [], color=COL["lung_left"], lw=2, label="lungs"),
         Line2D([], [], color=COL["skin"], lw=2, label="body")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.suptitle(
        "The measured 8-site triaxial chest array on the BodyParts3D torso\n"
        "plate geometry, spacing and per-site axes are MEASURED (OptiTrack, "
        "rigid to 0.00 mm); its placement over the precordium is ASSUMED",
        fontsize=13.5, x=0.010, ha="left", y=0.985, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    out = root / "outputs/chest/fig_chest_placement.png"
    fig.savefig(out, dpi=185, facecolor="white")
    print("figure:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
