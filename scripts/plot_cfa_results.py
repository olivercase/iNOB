#!/usr/bin/env python3
"""Results figures for the precision x movement 2x2.

``fig_2x2_results.png``       the factorial itself -- one panel per cell, laid out
                              as the 2x2 actually is (precision down, movement
                              across), each panel a dot plot of every arm's
                              coherent bias against the detection gate.
``fig_2x2_dissociation.png``  why the 2x2 has an answer: the crossover between the
                              model-based and data-driven arms, the variance each
                              factor explains, and what the triaxial readout buys.

Bias is on a LOG axis because the arms span 1.4 to 3600 fT; on a linear axis every
corrected arm collapses onto zero and the figure says nothing. The gate at
signal/3 is drawn on every panel because it, not the bar length, decides whether
the experiment is possible: the bias is R-locked and never averages down, so an
arm to the right of that line cannot be rescued by recording for longer.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# validated categorical slots (dataviz references/palette.md; all checks PASS)
C_NONE = "#e34948"      # slot 8, red
C_GLM = "#2a78d6"       # slot 1, blue
C_HFC = "#eb6834"       # slot 2, orange
C_CDM = "#4a3aa7"       # slot 7, violet
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# HFC order is ordinal, so its four arms are one hue stepped by lightness rather
# than four categorical slots -- and each is directly labelled, so the ramp never
# has to be read as identity.
HFC_SHADES = ["#f6b799", "#f19468", "#eb6834", "#b8481f"]


def arm_style(arm, hfc_orders):
    if arm == "none":
        return C_NONE, "no correction"
    if arm == "glm":
        return C_GLM, "joint GLM"
    if arm == "cdm":
        return C_CDM, "CDM (Woelk)"
    o = int(arm[3:arm.index("_")])
    return HFC_SHADES[list(hfc_orders).index(o)], f"HFC {o} + LCMV"


def panel(ax, res, cell, xlim, gate, signal, hfc_orders):
    order = ["none", "cdm", *[f"hfc{o}_lcmv" for o in hfc_orders], "glm"]
    labels = []
    for i, arm in enumerate(order):
        a = res.arms.index(arm)
        st = res.stats(cell, a)
        col, lab = arm_style(arm, hfc_orders)
        labels.append(lab)
        if st is None:
            ax.text(xlim[0] * 1.8, i, "undefined — needs ≥2 orientations",
                    va="center", fontsize=8.4, color=MUTED, style="italic")
            continue
        b = st.bias
        ax.plot([xlim[0], b], [i, i], color=col, lw=2.2, alpha=0.30,
                solid_capstyle="butt", zorder=2)
        ax.scatter([b], [i], s=80, color=col, zorder=4,
                   edgecolor="white", linewidth=1.1)
        ok = b < gate
        ax.annotate(f"{b:,.1f}", (b, i), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9.2, color=INK if ok else C_NONE,
                    fontweight="bold" if ok else "normal")
        if not ok:
            ax.annotate("✕", (b, i), xytext=(-17, 0), textcoords="offset points",
                        va="center", ha="center", fontsize=11, color=C_NONE)
    ax.axvline(gate, color=INK, lw=1.3, ls="--", zorder=1)
    ax.axvline(signal, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def fig_grid(res, out, dpi):
    d = res.design
    gate = res.sig_rms / d.snr_target
    xlim = (0.8, 9000.0)
    pos = {("high", "low"): (0, 0), ("high", "high"): (0, 1),
           ("low", "low"): (1, 0), ("low", "high"): (1, 1)}
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.8), sharex=True)
    for c, (p, m) in enumerate(d.cells):
        r, col = pos[(p, m)]
        ax = axes[r][col]
        panel(ax, res, c, xlim, gate, res.sig_rms, d.hfc_orders)
        ax.set_title(d.cell_labels[c], fontsize=13.5, color=INK, loc="left",
                     fontweight="bold", pad=9)
        if col == 1:
            ax.tick_params(labelleft=False)
    for col, lab in enumerate(["head STILL",
                               "head MOVING   (±100° yaw, 8 orientations)"]):
        axes[0][col].annotate(lab, xy=(0.5, 1.17), xycoords="axes fraction",
                              ha="center", fontsize=12.5, color=INK,
                              fontweight="bold")
    for row, lab in enumerate(["precision HIGH\n0.1% gain · 0.05° tilt",
                               "precision LOW\n10% gain · 3° tilt"]):
        axes[row][0].annotate(lab, xy=(-0.33, 0.5), xycoords="axes fraction",
                              rotation=90, va="center", ha="center",
                              fontsize=12, color=INK, fontweight="bold")
    for ax in axes[1]:
        ax.set_xlabel("coherent bias at the HEP latency   (fT, log scale)",
                      fontsize=10.5, color=MUTED)
    handles = [
        Line2D([], [], color=INK, ls="--", lw=1.3,
               label=f"detection gate  (signal/3 = {gate:.0f} fT)"),
        Line2D([], [], color=MUTED, ls=":", lw=1.0,
               label=f"the signal itself  ({res.sig_rms:.0f} fT)"),
        Line2D([], [], marker="o", ls="", color=C_NONE, ms=8, label="no correction"),
        Line2D([], [], marker="o", ls="", color=C_HFC, ms=8, label="HFC + LCMV"),
        Line2D([], [], marker="o", ls="", color=C_GLM, ms=8, label="joint GLM"),
        Line2D([], [], marker="o", ls="", color=C_CDM, ms=8, label="CDM"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=10)
    fig.suptitle(
        "Recovering a 10 nA·m heart-evoked field through a cardiac artefact 35× larger\n"
        "255-channel triaxial OPM array · head+thorax FEM · "
        "✕ = bias exceeds the gate, so no recording length succeeds",
        fontsize=14.5, x=0.010, ha="left", y=0.985, color=INK)
    fig.tight_layout(rect=(0.05, 0.06, 1, 0.895))
    fig.savefig(out, dpi=dpi, facecolor="white")
    print("figure:", out)


def fig_dissoc(res, an, axis, out, dpi):
    d = res.design
    gate = res.sig_rms / d.snr_target
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.4))
    cells = {(p, m): i for i, (p, m) in enumerate(d.cells)}

    ax = axes[0]
    for arm, col, lab in (("glm", C_GLM, "joint GLM"),
                          ("hfc5_lcmv", HFC_SHADES[-1], "HFC 5 + LCMV")):
        a = res.arms.index(arm)
        for m, ls, mk in (("low", "--", "o"), ("high", "-", "s")):
            y = [res.stats(cells[("high", m)], a).bias,
                 res.stats(cells[("low", m)], a).bias]
            ax.plot([0, 1], y, ls=ls, color=col, lw=2.1, marker=mk, ms=8.5,
                    markeredgecolor="white", markeredgewidth=1.2,
                    label=f"{lab} · {'moving' if m == 'high' else 'still'}")
    ax.axhline(gate, color=INK, lw=1.3, ls="--")
    ax.annotate(f"detection gate ({gate:.0f} fT)", (-0.15, gate * 1.14),
                fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["precision HIGH", "precision LOW"], fontsize=10.5, color=INK)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylabel("coherent bias (fT)", fontsize=10.5, color=MUTED)
    ax.set_title("A   the arms cross over", fontsize=12.5, loc="left", color=INK,
                 fontweight="bold")
    ax.legend(fontsize=8.6, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    arms = [a for a in ["glm", "hfc1_lcmv", "hfc2_lcmv", "hfc3_lcmv",
                        "hfc5_lcmv", "none"] if a in an]
    yy = np.arange(len(arms))
    w = 0.38
    for k, (key, col, lab) in enumerate((("eta2_precision", C_GLM, "precision"),
                                         ("eta2_movement", C_HFC, "movement"))):
        v = [an[a][key] for a in arms]
        ax.barh(yy + (k - 0.5) * w, v, height=w, color=col, label=lab,
                edgecolor="white", linewidth=1.0)
        for i, val in enumerate(v):
            ax.annotate(f"{val:.2f}", (val, yy[i] + (k - 0.5) * w), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8.6,
                        color=MUTED)
    ax.set_yticks(yy)
    ax.set_yticklabels([arm_style(a, d.hfc_orders)[1] for a in arms], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("partial η²  of log₁₀(signal / bias)", fontsize=10.5, color=MUTED)
    ax.set_title("B   which factor each arm answers to", fontsize=12.5, loc="left",
                 color=INK, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    ax = axes[2]
    ratios = [axis[k]["ratio"] for k in ("radial", "tangential_1", "tangential_2")]
    ratios.append(res.cfa_rms / res.sig_rms)
    cols = [C_NONE, C_HFC, C_HFC, C_GLM]
    ax.barh(np.arange(4), ratios, color=cols, height=0.62,
            edgecolor="white", linewidth=1.0)
    for i, v in enumerate(ratios):
        ax.annotate(f"{v:.0f}×", (v, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["radial only", "tangential 1", "tangential 2",
                        "all 255 (triaxial)"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(ratios) * 1.24)
    ax.set_xlabel("artefact : signal, before any correction", fontsize=10.5,
                  color=MUTED)
    ax.set_title("C   why the array is triaxial", fontsize=12.5, loc="left",
                 color=INK, fontweight="bold")
    ax.annotate("a radial-only array starts\n5× further behind",
                xy=(ratios[0] * 0.99, 0.18), xytext=(ratios[0] * 0.30, 1.45),
                fontsize=9.2, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle("The 2×2 has an answer: the model-based arm is precision-limited, "
                 "the beamformer is precision-robust",
                 fontsize=14, x=0.008, ha="left", y=0.98, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=dpi, facecolor="white")
    print("figure:", out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--dir", type=Path, default=root / "outputs/cfa_forward")
    p.add_argument("--dpi", type=int, default=190)
    a = p.parse_args(argv)
    res = pickle.load(open(a.dir / "sim_2x2.pkl", "rb"))
    js = json.loads((a.dir / "analysis.json").read_text())
    fig_grid(res, a.dir / "fig_2x2_results.png", a.dpi)
    fig_dissoc(res, js["anova"], js["axis_split"],
               a.dir / "fig_2x2_dissociation.png", a.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
