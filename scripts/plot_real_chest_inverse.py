#!/usr/bin/env python3
"""Figure: real chest MCG inverted through the iNOB FEM torso."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
OUT = root / "outputs/heart_fig"; OUT.mkdir(parents=True, exist_ok=True)
z = np.load(root / "outputs/chest_real/real_inverse.npz")
d = root / "outputs/cfa_thorax"
heart = trimesh.load(d / "heart.stl", process=True)
skin = trimesh.load(d / "skin.stl", process=True)
P, t, fld = z["pos"], z["t_ms"], z["field"]
grid, cog, dip, gof = z["grid"], z["cog"], z["dip_pos"], z["dip_gof"]
hc = z["heart_centroid"]
k = int(np.linalg.norm(fld, axis=0).argmax())
old = np.load(root / "outputs/chest/chest_array_placed.npz")["pos"]


def sect(ax, mesh, i, j, kx, lv, **kw):
    n = np.zeros(3); n[kx] = 1.0
    try:
        s = mesh.section(plane_origin=n * lv, plane_normal=n)
        if s is None: return
        for w, dsc in enumerate(s.discrete):
            kw2 = dict(kw)
            if w: kw2.pop("label", None)
            ax.plot(dsc[:, i], dsc[:, j], **kw2)
    except Exception: pass


fig = plt.figure(figsize=(16, 9))

ax = fig.add_subplot(2, 3, 1)
ax.plot(t, fld.T, lw=.6, alpha=.8)
ax.axvline(t[k], color="crimson", ls="--", lw=1)
ax.set_xlim(-300, 500); ax.set_xlabel("time from R-peak (ms)"); ax.set_ylabel("pT")
ax.set_title("(a) REAL chest MCG, R-locked average\n"
             "24 channels, 58 beats, peak 171 pT, SNR 165:1", fontsize=9)
ax.grid(alpha=.3)

ax = fig.add_subplot(2, 3, 2)
ax.plot(t, np.linalg.norm(fld, axis=0), color="0.3", lw=1.2, label="array norm")
ax.axvline(t[k], color="crimson", ls="--", lw=1, label="R-peak")
ok = np.isfinite(dip[:, 0])
ax2 = ax.twinx(); ax2.plot(t[ok], gof[ok], "o", ms=3, color="crimson", alpha=.7)
ax2.set_ylabel("dipole GoF", color="crimson"); ax2.set_ylim(0, 1)
ax.set_xlim(-300, 500); ax.set_xlabel("time from R-peak (ms)"); ax.set_ylabel("pT")
ax.set_title("(b) fit quality: single-dipole GoF peaks ~0.4\n"
             "(simulation reached 0.83 — real mismatch is genuine)", fontsize=9)
ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.3)

for n, (nm, i, j, kx, lv, xl, yl) in enumerate([
        ("(c) coronal y = +4 mm", 0, 2, 1, 3.9, "x R->L (mm)", "z I->S (mm)"),
        ("(d) axial z = -321 mm", 0, 1, 2, -320.9, "x R->L (mm)", "y P->A (mm)")]):
    ax = fig.add_subplot(2, 3, n + 3)
    sect(ax, skin, i, j, kx, lv, color="0.6", lw=1.2, label="skin")
    sect(ax, heart, i, j, kx, lv, color="crimson", lw=1.8, label="heart")
    ax.scatter(grid[:, i], grid[:, j], s=1, c="0.85", zorder=0)
    ax.scatter(P[:, i], P[:, j], s=55, c="darkorange", edgecolor="k",
               zorder=6, label="chest OPM (CFA pose)")
    ax.scatter(old[:, i], old[:, j], s=32, facecolor="none", edgecolor="tab:blue",
               zorder=5, label="earlier assumption")
    ax.scatter(*[[hc[i]], [hc[j]]], marker="*", s=200, c="crimson",
               edgecolor="k", zorder=8, label="heart centroid")
    ax.scatter(*[[cog[k, i]], [cog[k, j]]], marker="X", s=130, c="lime",
               edgecolor="k", zorder=9, label="MNE CoG @R (5.9 mm)")
    ax.scatter(*[[dip[k, i]], [dip[k, j]]], marker="P", s=110, c="magenta",
               edgecolor="k", zorder=9, label="moving dipole @R")
    ax.set_title(nm, fontsize=9); ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_aspect("equal"); ax.grid(alpha=.25); ax.legend(fontsize=6)

ax = fig.add_subplot(2, 3, 5)
w = np.abs(z["mne"]).reshape(len(grid), 3, -1).sum(1)[:, k]
sect(ax, heart, 0, 2, 1, 3.9, color="k", lw=1.4)
s = ax.scatter(grid[:, 0], grid[:, 2], c=w, s=26, cmap="inferno")
ax.scatter(hc[0], hc[2], marker="*", s=200, c="cyan", edgecolor="k", zorder=8)
fig.colorbar(s, ax=ax, fraction=.04, label="|J| (a.u.)")
ax.set_aspect("equal"); ax.set_title("(e) MNE current density at the R-peak\n"
                                     "(coronal, grid inside the heart)", fontsize=9)
ax.set_xlabel("x R->L (mm)"); ax.set_ylabel("z I->S (mm)")

ax = fig.add_subplot(2, 3, 6); ax.axis("off")
ax.text(0, 1, "REAL chest MCG -> iNOB FEM inversion\n\n"
        f"plate pose      reproduced from the CFA pilot's own\n"
        f"                torso fiducials (hand-picked, ~16 mm),\n"
        f"                slid {float(z['slide_mm']):.0f} mm out to clear this torso\n\n"
        "placement       CFA pose lands 60-91 mm from epicardium;\n"
        "                earlier precordium assumption gave 55-88 mm\n"
        "                => agree in depth, differ ~20 mm laterally\n\n"
        "forward         DUNEuro FEM, 279,947 tets, 7 tissues\n"
        "                974-point grid filling the heart\n\n"
        "RESULT\n"
        "  MNE centre of gravity  5.9 mm from heart centroid  [OK]\n"
        "  single-dipole GoF      0.41 at the R-peak  [weak]\n"
        "  dipole sits ~50 mm out, pinned to the anterior wall\n\n"
        "Reading: the real data localises to the heart, but a\n"
        "template torso + ~16 mm plate pose will not support a\n"
        "well-fitting single equivalent dipole.",
        va="top", ha="left", fontsize=8.5, family="monospace")

fig.suptitle("Real measured chest MCG inverted through the iNOB BodyParts3D FEM torso, "
             "at the CFA pilot's plate pose", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fig7_real_chest_inverse.png", dpi=140, bbox_inches="tight")
print("fig7 ->", OUT / "fig7_real_chest_inverse.png")
