#!/usr/bin/env python3
"""The one figure the CFA argument needs: signal turns, artefact does not.

Three rows, five head orientations, one column per orientation, ordered by yaw so
the head sweeps left to right:

    row 1   what the subject is doing -- head seen from above, nose turning,
            heart fixed in the room
    row 2   HEP, the brain source: rotates WITH the head
    row 3   CFA, the heart source: stays with the ROOM

Everything is drawn in the room's frame, which is what makes that readable. In a
head-fixed layout both maps are pinned to the same sensor positions and the
dissociation is invisible -- it has to be read off correlation numbers instead of
seen. Here it is the difference between a pattern that tracks the grey nose and
one that ignores it.

Colour is diverging (two hues, neutral midpoint) because the data are signed
field; one shared scale per row, because comparing panels within a row is the
whole point, and the two rows differ by ~80x so they cannot share one.

    python scripts/plot_cfa_dissociation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.path import Path as MplPath
from scipy.interpolate import Rbf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inob.cfa.array import CFAArray
from inob.cfa.posture import rotation_matrix

def _heart_marker() -> MplPath:
    """A heart glyph, as a marker path.

    The classic parametric heart, centred and scaled to roughly unit area so it
    sits at the same visual weight as matplotlib's stock markers.
    """
    t = np.linspace(0.0, 2.0 * np.pi, 160)
    x = 16.0 * np.sin(t) ** 3
    y = (13.0 * np.cos(t) - 5.0 * np.cos(2 * t)
         - 2.0 * np.cos(3 * t) - np.cos(4 * t))
    v = np.column_stack([x, y])
    v -= v.mean(0)
    v /= np.abs(v).max()
    return MplPath(v, closed=True)


HEART_MARKER = _heart_marker()

INK = "#1c1f23"
MUTED = "#6b7280"
HEART = "#c8323c"
HEAD_FILL = "#e9ecef"
HEAD_EDGE = "#495057"


def flatten(pos: np.ndarray) -> np.ndarray:
    """Azimuthal-equidistant projection of sensor positions onto a disc."""
    v = pos - pos.mean(0)
    u = v / np.linalg.norm(v, axis=1)[:, None]
    theta = np.arccos(np.clip(u[:, 2], -1, 1))
    phi = np.arctan2(u[:, 1], u[:, 0])
    return np.stack([theta * np.cos(phi), theta * np.sin(phi)], 1)


def rot2(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def draw_ears(ax, r: float, yaw: float, *, color=None, lw: float = 1.5,
              size: float = 0.14, zorder: int = 4):
    """Semicircular ears at +/-90 deg from the nose, turning with the head.

    Standard on a MEG/EEG topography, and they earn their place here: with a nose
    alone a turned head is ambiguous between yaw and its reflection, because a
    disc with one marker has a mirror symmetry the ears break.
    """
    color = color or HEAD_EDGE
    for side in (+1.0, -1.0):
        # The nose is drawn along +y, i.e. at polar angle 90 deg, so an ear at
        # "90 deg from the nose" is at polar 90 + yaw +/- 90 -- not at yaw +/- 90,
        # which puts one ear on top of the nose.
        a = np.deg2rad(90.0 + yaw + side * 90.0)
        c = np.array([np.cos(a), np.sin(a)]) * r
        t = np.linspace(a - np.pi / 2, a + np.pi / 2, 48)
        ax.plot(c[0] + size * r * np.cos(t), c[1] + size * r * np.sin(t),
                color=color, lw=lw, solid_capstyle="round", zorder=zorder,
                clip_on=False)


def draw_head(ax, yaw: float, heart_dir: np.ndarray, *, r: float = 1.0):
    """Top-down head with a turning nose, and the heart fixed in the room."""
    ax.add_artist(Circle((0, 0), r, facecolor=HEAD_FILL, edgecolor=HEAD_EDGE, lw=1.6))
    draw_ears(ax, r, yaw, lw=1.6)
    nose = np.array([[0.0, 1.34], [-0.26, 0.94], [0.26, 0.94]]) * r @ rot2(yaw).T
    ax.fill(nose[:, 0], nose[:, 1], color=HEAD_EDGE, zorder=5)
    h = heart_dir / np.linalg.norm(heart_dir) * r * 1.72
    ax.scatter([h[0]], [h[1]], s=210, marker=HEART_MARKER, color=HEART,
               zorder=5, clip_on=False)
    ax.set_xlim(-2.05, 2.05); ax.set_ylim(-2.05, 2.05)
    ax.set_aspect("equal"); ax.axis("off")


def draw_topo(ax, xy: np.ndarray, val: np.ndarray, vlim: float, yaw: float,
              heart_dir: np.ndarray, *, n: int = 170):
    xyr = xy @ rot2(yaw).T
    r_max = np.linalg.norm(xyr, axis=1).max() * 1.05
    g = np.linspace(-r_max, r_max, n)
    gx, gy = np.meshgrid(g, g)
    z = Rbf(xyr[:, 0], xyr[:, 1], val, function="multiquadric", smooth=0.02)(gx, gy)
    z[gx ** 2 + gy ** 2 > r_max ** 2] = np.nan
    im = ax.contourf(gx, gy, z, levels=np.linspace(-vlim, vlim, 25),
                     cmap="RdBu_r", extend="both", zorder=1)
    ax.contour(gx, gy, z, levels=np.linspace(-vlim, vlim, 11), colors=INK,
               linewidths=0.3, alpha=0.28, zorder=2)
    ax.add_artist(Circle((0, 0), r_max, fill=False, lw=1.5, color=HEAD_EDGE, zorder=3))
    draw_ears(ax, r_max, yaw, lw=1.5, zorder=4)
    nose = np.array([[0.0, 1.20], [-0.16, 1.0], [0.16, 1.0]]) * r_max @ rot2(yaw).T
    ax.fill(nose[:, 0], nose[:, 1], color=HEAD_EDGE, zorder=5, clip_on=False)
    ax.scatter(xyr[:, 0], xyr[:, 1], s=2.6, c=INK, alpha=0.5, zorder=5)
    h = heart_dir / np.linalg.norm(heart_dir) * r_max * 1.42
    ax.scatter([h[0]], [h[1]], s=170, marker=HEART_MARKER, color=HEART,
               zorder=6, clip_on=False)
    ax.set_xlim(-1.6 * r_max, 1.6 * r_max); ax.set_ylim(-1.6 * r_max, 1.6 * r_max)
    ax.set_aspect("equal"); ax.axis("off")
    return im


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--dir", type=Path, default=root / "outputs/cfa_forward")
    p.add_argument("--source-nam", type=float, default=10.0)
    p.add_argument("--dpi", type=int, default=190)
    a = p.parse_args(argv)

    s = np.load(a.dir / "sources.npz")
    myo_ori, q = s["myo_ori"], s["q"]
    erp = int(s["erp_index"])
    array = CFAArray.from_reference(root / "outputs/_cfa_ref.npz")
    S = array.n_site
    xy = flatten(array.site_pos_mm)

    post = json.loads((root / "outputs/cfa_thorax/posture.json").read_text())
    pivot = np.asarray(post["neck"]["pivot_mm"], float)
    heart_c = np.array([-21.2, 3.9, -320.9])
    heart_dir = (heart_c - pivot)[:2]

    w = s["myo_area_mm2"] / s["myo_area_mm2"].mean()
    nw = (myo_ori * w[:, None]).T
    amps = w[:, None] * (nw.T @ np.linalg.solve(nw @ nw.T, q))

    rows = []
    for f in sorted(a.dir.glob("orient_*.npz")):
        d = np.load(f)
        ypr = d["ypr_deg"]
        m = rotation_matrix(ypr) @ (np.ones(3) / np.sqrt(3.0))
        hep = array.project(np.einsum("sck,k->sc", d["B_ins"], m)[:, :, None],
                            d["chanori"])[:S, 0] * a.source_nam * 1e-9 * 1e15
        per = np.einsum("scnk,nk->scn", d["B_myo"].reshape(S, 3, -1, 3), myo_ori)
        cfa = (array.project(per, d["chanori"]) @ amps[:, erp])[:S] * 1e15
        rows.append({"yaw": float(ypr[0]), "ypr": ypr, "hep": hep, "cfa": cfa})
    rows.sort(key=lambda r: r["yaw"])
    keep = [r for r in rows if abs(r["yaw"]) in (0.0, 45.0, 100.0)]

    v_hep = max(np.abs(r["hep"]).max() for r in keep)
    v_cfa = max(np.abs(r["cfa"]).max() for r in keep)

    n = len(keep)
    fig = plt.figure(figsize=(2.35 * n + 4.4, 9.1))
    gs = fig.add_gridspec(3, n + 1, height_ratios=[0.72, 1, 1],
                          width_ratios=[*([1] * n), 0.1],
                          hspace=0.06, wspace=0.04, left=0.175, right=0.925,
                          top=0.845, bottom=0.075)

    for j, r in enumerate(keep):
        ax = fig.add_subplot(gs[0, j])
        draw_head(ax, r["yaw"], heart_dir)
        lab = "head still" if r["yaw"] == 0 else f"{r['yaw']:+.0f}° yaw"
        ax.set_title(lab, fontsize=12.5, color=INK, pad=8)
        if j == 0:
            h = heart_dir / np.linalg.norm(heart_dir) * 1.72
            ax.annotate("heart\nfixed in the room", xy=(h[0], h[1]),
                        xytext=(h[0] - 1.05, h[1] - 1.05), fontsize=9.5,
                        color=HEART, ha="right", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.8, color=HEART,
                                        shrinkA=2, shrinkB=6),
                        annotation_clip=False)

    for i, (key, vlim, cmap_lab) in enumerate(
            [("hep", v_hep, "fT"), ("cfa", v_cfa, "fT")], start=1):
        for j, r in enumerate(keep):
            ax = fig.add_subplot(gs[i, j])
            im = draw_topo(ax, xy, r[key], vlim, r["yaw"], heart_dir)
            ax.text(0.5, -0.02, f"{np.abs(r[key]).max():,.0f} {cmap_lab}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, color=MUTED)
        cax = fig.add_subplot(gs[i, n])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(cmap_lab, fontsize=10, color=MUTED)
        cb.ax.tick_params(labelsize=8.5, colors=MUTED)
        cb.outline.set_visible(False)

    def rowlabel(y, line1, line2, colour):
        fig.text(0.010, y, line1, fontsize=14, color=INK, va="center",
                 fontweight="bold")
        fig.text(0.010, y - 0.036, line2, fontsize=11, color=colour, va="center")

    rowlabel(0.785, "the subject", "seen from above", MUTED)
    rowlabel(0.560, "HEP  ·  brain source", "turns WITH the head", "#1b6ec2")
    rowlabel(0.240, "CFA  ·  heart source", "stays with the ROOM", HEART)

    ratio = v_cfa / v_hep
    fig.suptitle(
        "The heart-evoked field turns with the head; the cardiac artefact does not\n"
        f"OPM array rigidly fixed to the skull · both maps at the HEP latency "
        f"(650 ms) · artefact is {ratio:,.0f}× the signal",
        fontsize=15, x=0.012, ha="left", y=0.975, color=INK)
    out = a.dir / "fig_dissociation.png"
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    print("figure:", out, f"| CFA/HEP = {ratio:.0f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
