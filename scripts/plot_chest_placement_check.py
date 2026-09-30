#!/usr/bin/env python3
"""Chest-array placement check: measured geometry, assumed position, and the
real head<->chest relationship recovered from the OptiTrack take.

    python scripts/plot_chest_placement_check.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, trimesh, scipy.io as sio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

root = Path(__file__).resolve().parents[1]
OUT = root / "outputs/heart_fig"; OUT.mkdir(parents=True, exist_ok=True)
d = root / "outputs/cfa_thorax"
PACK = Path("/Users/olivercase/Desktop/CFA/replication pack")

skin = trimesh.load(d / "skin.stl", process=True)
heart = trimesh.load(d / "heart.stl", process=True)
lungs = [trimesh.load(d / f"lung_{s}.stl", process=True) for s in ("left", "right")]
P = np.load(root / "outputs/chest/chest_array_placed.npz")["pos"]


def sect(ax, mesh, i, j, kx, lv, **kw):
    n = np.zeros(3); n[kx] = 1.0
    try:
        s = mesh.section(plane_origin=n * lv, plane_normal=n)
        if s is None: return
        for w, dsc in enumerate(s.discrete):
            kw2 = dict(kw)
            if w: kw2.pop("label", None)
            ax.plot(dsc[:, i], dsc[:, j], **kw2)
    except Exception:
        pass


fig = plt.figure(figsize=(17, 10.5))

# (a) plate face-on: the square lattice
ax = fig.add_subplot(2, 3, 1)
c = P.mean(0); X = P - c
Vt = np.linalg.svd(X, full_matrices=False)[2]
loc = X @ Vt.T
pitch = 48.3
for g in np.arange(-2, 2.01, 1) * pitch:
    ax.axvline(g, color="0.9", lw=.8, zorder=0); ax.axhline(g, color="0.9", lw=.8, zorder=0)
ax.scatter(loc[:, 0], loc[:, 1], s=210, c="darkorange", edgecolor="k", zorder=3)
for n, p in enumerate(loc):
    ax.text(p[0], p[1], str(n + 1), ha="center", va="center", fontsize=8, zorder=4)
ax.set_aspect("equal")
ax.set_title("(a) MEASURED plate, face-on in its own plane\n"
             "48.3 mm square lattice (4 mid row, 2 above, 2 below)\n"
             "147.4 x 98.9 mm in-plane, flat to 0.007 mm", fontsize=9)
ax.set_xlabel("plate u (mm)"); ax.set_ylabel("plate v (mm)")

# (b,c) cross-sections, only sensors near the slice
for n, (nm, i, j, kx, lv, xl, yl) in enumerate([
        ("(b) axial z = -321 mm (heart centroid)", 0, 1, 2, -320.9, "x  R->L (mm)", "y  P->A (mm)"),
        ("(c) sagittal x = -23 mm (plate midline)", 1, 2, 0, -23.0, "y  P->A (mm)", "z  I->S (mm)")]):
    ax = fig.add_subplot(2, 3, n + 2)
    for mesh, col, lw, lab in [(skin, "0.45", 1.5, "skin"), (heart, "crimson", 2.0, "heart"),
                               (lungs[0], "steelblue", 1.2, "lung"), (lungs[1], "steelblue", 1.2, None)]:
        sect(ax, mesh, i, j, kx, lv, color=col, lw=lw, label=lab)
    m = np.abs(P[:, kx] - lv) < 55
    ax.scatter(P[m, i], P[m, j], s=95, c="darkorange", edgecolor="k", zorder=6,
               label=f"chest OPM ({m.sum()} near slice)")
    ax.set_title(nm, fontsize=9); ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_aspect("equal"); ax.grid(alpha=.25); ax.legend(fontsize=7)

# (d) 3D torso + heart + plate
ax = fig.add_subplot(2, 3, 4, projection="3d")
sk = skin.copy()
try: sk = sk.simplify_quadric_decimation(face_count=3000)
except Exception: pass
keep = sk.triangles_center[:, 2] > -450
ax.add_collection3d(Poly3DCollection(sk.triangles[keep], facecolor="0.8", alpha=.12, linewidths=0))
hh = heart.copy()
try: hh = hh.simplify_quadric_decimation(face_count=1800)
except Exception: pass
ax.add_collection3d(Poly3DCollection(hh.triangles, facecolor="crimson", alpha=.55, linewidths=0))
ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=60, c="darkorange", edgecolor="k", depthshade=False)
ctr = np.vstack([P, heart.vertices]).mean(0); rad = 170
ax.set_xlim(ctr[0]-rad, ctr[0]+rad); ax.set_ylim(ctr[1]-rad, ctr[1]+rad); ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
ax.view_init(14, -62); ax.set_axis_off()
ax.set_title("(d) plate over the precordium (ASSUMED position)", fontsize=9)

# (e) distances
ax = fig.add_subplot(2, 3, 5)
clear = -trimesh.proximity.ProximityQuery(skin).signed_distance(P)
dh = -trimesh.proximity.ProximityQuery(heart).signed_distance(P)
w = .38; x = np.arange(8)
ax.bar(x - w/2, clear, w, label="clearance outside skin", color="0.6")
ax.bar(x + w/2, dh, w, label="distance to epicardium", color="crimson")
ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels(x + 1)
ax.set_xlabel("chest site"); ax.set_ylabel("mm"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
ax.set_title("(e) 0 sites inside body; 5.8-34.5 mm clearance;\n55-88 mm to epicardium", fontsize=9)

# (f) the REAL measured head<->chest geometry, OptiTrack frame
ax = fig.add_subplot(2, 3, 6)
m = sio.loadmat(root / "outputs/chest/arrays_frames.mat", squeeze_me=True, struct_as_record=False)
cp, hp, fr = m["chest_pos"], m["head_pos"], m["frames"]
cols = plt.cm.viridis(np.linspace(0, .9, cp.shape[2]))
for k in range(cp.shape[2]):
    C = np.unique(np.round(cp[:, :, k], 4), axis=0) * 1000
    H = np.unique(np.round(hp[:, :, k], 4), axis=0) * 1000
    ax.scatter(H[:, 0], H[:, 1], s=7, color=cols[k], label=f"head, frame {fr[k]}")
    if k == 0:
        ax.scatter(C[:, 0], C[:, 1], s=70, c="darkorange", edgecolor="k",
                   zorder=5, label="chest plate (rigid)")
ax.set_aspect("equal"); ax.set_xlabel("OptiTrack x (mm)"); ax.set_ylabel("OptiTrack y, up (mm)")
ax.legend(fontsize=6.5, loc="center left"); ax.grid(alpha=.25)
ax.set_title("(f) REAL measured head<->chest geometry\n"
             "same OptiTrack frame: head sits ~430 mm above the plate\n"
             "and rotates while the plate stays rigid", fontsize=9)

fig.suptitle("Chest array: geometry and head-relationship are MEASURED; "
             "position on the anatomy is ASSUMED (no subject MRI/scannercast on disk)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fig6_array_placement.png", dpi=140, bbox_inches="tight")
print("fig6 ->", OUT / "fig6_array_placement.png")
