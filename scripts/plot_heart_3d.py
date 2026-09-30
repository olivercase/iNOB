#!/usr/bin/env python3
"""3D-rendered heart activation map + chest-array placement check.

    python scripts/plot_heart_3d.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

root = Path(__file__).resolve().parents[1]
SP = "/private/tmp/claude-501/-Users-olivercase-Desktop-CFA/8e26989c-c5b0-4b6c-a490-4ecb2d71b975/scratchpad"
OUT = root / "outputs/heart_fig"; OUT.mkdir(parents=True, exist_ok=True)
d = root / "outputs/cfa_thorax"

z = np.load(f"{SP}/myo_src.npz")
pos, amps, t = z["pos"], np.abs(z["amps"]), z["time"] * 1000.0
myo = trimesh.Trimesh(z["mv"], z["mf"], process=False)
QRS = (t >= 390) & (t <= 490)
P = np.load(root / "outputs/chest/chest_array_placed.npz")["pos"]


def face_values(vals, k=6, p=2.0):
    """Inverse-distance map from the 400 dipole sites onto mesh faces."""
    fc = myo.triangles_center
    tree = cKDTree(pos)
    dist, idx = tree.query(fc, k=k)
    w = 1.0 / np.maximum(dist, 1e-6) ** p
    return (w * vals[idx]).sum(1) / w.sum(1)


def render(ax, vals, vmax, elev, azim, cmap="turbo", shade=True):
    fc = face_values(vals)
    norm = np.clip(fc / max(vmax, 1e-30), 0, 1)
    cols = plt.get_cmap(cmap)(norm)
    if shade:                                  # simple lambert shading for depth
        n = myo.face_normals
        light = np.array([0.4, -0.8, 0.45]); light /= np.linalg.norm(light)
        lam = 0.55 + 0.45 * np.clip(n @ light, 0, 1)
        cols[:, :3] *= lam[:, None]
    pc = Poly3DCollection(myo.triangles, facecolors=cols, linewidths=0, alpha=1.0)
    ax.add_collection3d(pc)
    b = myo.bounds
    ctr, rad = b.mean(0), 0.55 * (b[1] - b[0]).max()
    for setlim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), ctr):
        setlim(c - rad, c + rad)
    ax.view_init(elev, azim); ax.set_axis_off()
    try: ax.set_box_aspect((1, 1, 1))
    except Exception: pass


def fig_heartmap():
    """The 3D heart map: activation rendered on the myocardium across the QRS."""
    ks = np.linspace(np.where(QRS)[0][0], np.where(QRS)[0][-1], 6).astype(int)
    mne = np.abs(np.load(root / "outputs/chest/inv_real_8_(flat_plate).npz")["mne"])
    mne64 = np.abs(np.load(root / "outputs/chest/inv_conformal_64.npz")["mne"])
    rows = [("TRUTH\npropagating source", amps),
            ("recovered\nreal 8-site plate", mne),
            ("recovered\nconformal 64-site", mne64)]
    fig = plt.figure(figsize=(3.0 * len(ks), 3.1 * len(rows)))
    for r, (lab, M) in enumerate(rows):
        vmax = np.percentile(M[:, QRS], 99.5)
        for c, k in enumerate(ks):
            ax = fig.add_subplot(len(rows), len(ks), r * len(ks) + c + 1, projection="3d")
            render(ax, M[:, k], vmax, elev=12, azim=-72)
            if r == 0: ax.set_title(f"{t[k]:.0f} ms", fontsize=12)
            if c == 0: ax.text2D(-0.04, 0.5, lab, transform=ax.transAxes, rotation=90,
                                 va="center", ha="center", fontsize=10)
    fig.suptitle("3D activation map on the myocardium, across the QRS  (anterior-left view)\n"
                 "true source is a sparse propagating wavefront; chest reconstructions "
                 "recover a smooth, spatially blurred blob", fontsize=13)
    fig.tight_layout(rect=[0.01, 0, 1, 0.95])
    fig.savefig(OUT / "fig4_heart_3d_map.png", dpi=135, bbox_inches="tight")
    print("fig4 ->", OUT / "fig4_heart_3d_map.png")


def fig_rotate():
    """Peak-QRS activation seen from four angles, with the chest array."""
    k = int(np.argmax(amps.sum(0) * QRS))
    views = [(12, -72, "anterior-left"), (12, -160, "left lateral"),
             (12, 18, "anterior-right"), (78, -90, "superior")]
    fig = plt.figure(figsize=(16, 4.6))
    for i, (el, az, nm) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        render(ax, amps[:, k], np.percentile(amps[:, QRS], 99.5), el, az)
        ax.set_title(nm, fontsize=11)
    fig.suptitle(f"Myocardial activation at peak QRS ({t[k]:.0f} ms), four views", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_heart_3d_views.png", dpi=135, bbox_inches="tight")
    print("fig5 ->", OUT / "fig5_heart_3d_views.png")


def fig_placement():
    """Is the measured chest plate where it would really sit on the torso?"""
    skin = trimesh.load(d / "skin.stl", process=True)
    heart = trimesh.load(d / "heart.stl", process=True)
    lungs = [trimesh.load(d / f"lung_{s}.stl", process=True) for s in ("left", "right")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    PL = [("sagittal  (x = -23 mm, through the plate)", 1, 2, 0, -23.0,
           "y  P→A (mm)", "z  I→S (mm)"),
          ("axial  (z = -320 mm, heart centroid)", 0, 1, 2, -320.9,
           "x  R→L (mm)", "y  P→A (mm)"),
          ("coronal  (y = +60 mm)", 0, 2, 1, 60.0, "x  R→L (mm)", "z  I→S (mm)")]
    for ax, (nm, i, j, kx, lv, xl, yl) in zip(axes, PL):
        n = np.zeros(3); n[kx] = 1.0
        for mesh, col, lw, lab in [(skin, "0.45", 1.4, "skin"), (heart, "crimson", 1.8, "heart"),
                                   (lungs[0], "steelblue", 1.1, "lung"), (lungs[1], "steelblue", 1.1, None)]:
            try:
                sec = mesh.section(plane_origin=n * lv, plane_normal=n)
                if sec is None: continue
                for w, dsc in enumerate(sec.discrete):
                    ax.plot(dsc[:, i], dsc[:, j], color=col, lw=lw,
                            label=lab if w == 0 else None)
            except Exception: pass
        sel = np.abs(P[:, kx] - lv) < 60
        ax.scatter(P[:, i], P[:, j], s=55, c="darkorange", edgecolor="k",
                   zorder=6, label="chest OPM sites")
        ax.set_title(nm, fontsize=10); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal"); ax.grid(alpha=.25); ax.legend(fontsize=8, loc="best")
    fig.suptitle("Measured 8-site chest plate on the BodyParts3D torso — placement check\n"
                 "0 sites inside the body | 5.8–34.5 mm skin clearance | 55–88 mm to the "
                 "epicardial surface | plate centroid 111 mm anterior to the heart, x/z aligned to 2 mm\n"
                 "Plate GEOMETRY is measured (flat to 0.00 mm, 48.3 mm spacing); its POSITION on the "
                 "torso is an assumption — OptiTrack→MNI has no registration bridge", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig6_array_placement.png", dpi=140, bbox_inches="tight")
    print("fig6 ->", OUT / "fig6_array_placement.png")


if __name__ == "__main__":
    fig_heartmap(); fig_rotate(); fig_placement()
