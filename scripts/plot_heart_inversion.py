#!/usr/bin/env python3
"""Figures: cardiac dipole movement through the heart across the QRS.

Reconstructions come from ``run_chest_inverse.py``'s cached inversions; the truth
is the propagating myocardial source they were generated from. All coordinates
are MNI mm from the BodyParts3D thorax build.

    python scripts/plot_heart_inversion.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
SP = "/private/tmp/claude-501/-Users-olivercase-Desktop-CFA/8e26989c-c5b0-4b6c-a490-4ecb2d71b975/scratchpad"
OUT = root / "outputs/heart_fig"; OUT.mkdir(parents=True, exist_ok=True)

z = np.load(f"{SP}/myo_src.npz")
pos, amps, t = z["pos"], np.abs(z["amps"]), z["time"] * 1000.0
mv, mf = z["mv"], z["mf"]
myo = trimesh.Trimesh(mv, mf, process=False)

# the two physiological events, found from the source itself
power = amps.sum(0)
QRS = (t >= 390) & (t <= 490)
TW = (t >= 600) & (t <= 800)

PLANES = [("coronal", 0, 2, "x (R-L)", "z (I-S)"),
          ("sagittal", 1, 2, "y (P-A)", "z (I-S)"),
          ("axial", 0, 1, "x (R-L)", "y (P-A)")]


def silhouette(ax, mesh, i, j, **kw):
    """Outline of a mesh projected on the (i, j) plane."""
    v = mesh.vertices
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(v[:, [i, j]])
        p = np.append(h.vertices, h.vertices[0])
        ax.plot(v[p, i], v[p, j], **kw)
    except Exception:
        ax.plot(v[:, i], v[:, j], ".", ms=0.3, **kw)


def fig1():
    """Moving equivalent dipole through the QRS, three projections."""
    inv = np.load(root / "outputs/chest/inv_real_8_(flat_plate).npz")
    dp, gof = inv["dip_pos"], inv["dip_gof"]
    ok = np.isfinite(dp[:, 0])
    # truth centre of gravity, masked to where there is actually a source
    w = amps
    tot = w.sum(0)
    good = tot > 0.02 * tot.max()
    cog = np.full_like(dp, np.nan)
    cog[good] = (w[:, good].T @ pos) / tot[good, None]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (nm, i, j, xl, yl) in zip(axes, PLANES):
        silhouette(ax, myo, i, j, color="0.55", lw=1.2, zorder=1)
        ax.scatter(pos[:, i], pos[:, j], s=3, c="0.85", zorder=0)
        # truth CoG path across QRS+T, coloured by time
        m = good & (t >= 390) & (t <= 800)
        seg = np.stack([cog[m][:-1][:, [i, j]], cog[m][1:][:, [i, j]]], 1)
        lc = LineCollection(seg, cmap="viridis", linewidths=2.6,
                            array=t[m][:-1], zorder=3)
        ax.add_collection(lc)
        # reconstructed moving dipole, sized by goodness of fit
        s = ax.scatter(dp[ok, i], dp[ok, j], c=t[ok], cmap="viridis",
                       s=18 + 90 * np.clip(gof[ok], 0, 1) ** 2,
                       edgecolor="crimson", linewidth=0.7, zorder=4)
        ax.set_title(nm); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
    cb = fig.colorbar(s, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("time (ms)")
    fig.suptitle("Cardiac source movement through the heart\n"
                 "grey = myocardial wall & outline | line = true centre of gravity | "
                 "red-edged = moving dipole recovered from the real 8-site chest plate "
                 "(size = GoF)", fontsize=11)
    fig.savefig(OUT / "fig1_dipole_movement.png", dpi=140, bbox_inches="tight")
    print("fig1 ->", OUT / "fig1_dipole_movement.png")


def fig2():
    """Activation sequence on the myocardial wall: truth vs reconstruction."""
    inv = np.load(root / "outputs/chest/inv_real_8_(flat_plate).npz")
    mne = np.abs(inv["mne"])
    inv64 = np.load(root / "outputs/chest/inv_conformal_64.npz")
    mne64 = np.abs(inv64["mne"])
    ks = np.linspace(np.where(QRS)[0][0], np.where(QRS)[0][-1], 6).astype(int)
    rows = [("truth (propagating source)", amps), ("real 8-site plate", mne),
            ("conformal 64-site", mne64)]
    fig, axes = plt.subplots(len(rows), len(ks), figsize=(2.5 * len(ks), 2.7 * len(rows)))
    i, j = 0, 2                                   # coronal
    for r, (lab, M) in enumerate(rows):
        vmax = M[:, QRS].max()
        for c, k in enumerate(ks):
            ax = axes[r, c]
            silhouette(ax, myo, i, j, color="0.6", lw=0.9)
            ax.scatter(pos[:, i], pos[:, j], c=M[:, k], s=13, cmap="inferno",
                       vmin=0, vmax=vmax)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(f"{t[k]:.0f} ms", fontsize=10)
            if c == 0: ax.set_ylabel(lab, fontsize=9)
    fig.suptitle("Activation sequence across the QRS, on the myocardial wall (coronal)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_activation_sequence.png", dpi=140, bbox_inches="tight")
    print("fig2 ->", OUT / "fig2_activation_sequence.png")


def fig3():
    """Time courses: chest field, fit quality, trajectory speed."""
    inv = np.load(root / "outputs/chest/inv_real_8_(flat_plate).npz")
    fld, gof, dp = inv["field"], inv["dip_gof"], inv["dip_pos"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    ax[0].plot(t, fld.T / 1e3, lw=0.7, alpha=0.75)
    ax[0].set_ylabel("chest field (pT)")
    ax[0].set_title("Real 8-site triaxial chest plate: 24 channels")
    ax[1].plot(t, power / power.max(), color="0.3")
    ax[1].fill_between(t, 0, power / power.max(), where=QRS, color="crimson", alpha=.3, label="QRS")
    ax[1].fill_between(t, 0, power / power.max(), where=TW, color="steelblue", alpha=.3, label="T")
    ax[1].set_ylabel("source power (norm.)"); ax[1].legend(loc="upper right")
    ok = np.isfinite(dp[:, 0])
    ax[2].plot(t[ok], gof[ok], "o", ms=4, color="crimson")
    ax[2].set_ylabel("moving-dipole GoF"); ax[2].set_xlabel("time (ms)")
    ax[2].set_ylim(0, 1); ax[2].grid(alpha=.3)
    for a in ax: a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_timecourses.png", dpi=140, bbox_inches="tight")
    print("fig3 ->", OUT / "fig3_timecourses.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
