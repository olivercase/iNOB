#!/usr/bin/env python3
"""Visual check of the head-on-neck rotations used by the CFA 2x2.

Builds the neck model from cervical landmarks (C1/C2/T1), deforms every
compartment for each head orientation in the design, and draws them so the
posture can be inspected before any FEM is built on it.

    python scripts/plot_cfa_posture.py [--orientation-figs]
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inob.cfa.design import DESIGN
from inob.cfa.posture import NeckModel, physiological_fraction

logger = logging.getLogger("plot_cfa_posture")

#: BodyParts3D component ids for the cervical landmarks the pivot is built from.
LANDMARKS = {"atlas": "FJ3176", "axis": "FJ3177", "t1": "FJ3158"}

# label -> (colour, deformation rule)
#   "soft"  per-vertex twist      "body" per-connected-component rigid twist
#   "head"  rigid, head-fixed     "fixed" not deformed at all
RULE = {
    "skin":        ("#9aa3ad", "soft"),
    "bone":        ("#d9d2c0", "body"),
    "brain":       ("#c08fd6", "head"),
    "spinal_cord": ("#b58fd6", "soft"),
    "heart":       ("#c8323c", "fixed"),
    "lung_left":   ("#7fb2d9", "fixed"),
    "lung_right":  ("#7fb2d9", "fixed"),
}


def load_landmarks(root: Path, T: np.ndarray) -> dict[str, trimesh.Trimesh]:
    idx: dict[str, str] = {}
    for p in glob.glob(str(root / "internal_meshes/*/*.obj")):
        idx.setdefault(Path(p).name.split("_", 1)[0], p)
    out = {}
    for name, fj in LANDMARKS.items():
        m = trimesh.load(idx[fj], process=True)
        m.apply_transform(T)
        out[name] = m
    return out


def head_block(root: Path, T: np.ndarray, brain: trimesh.Trimesh) -> trimesh.Trimesh:
    """Skull (FMA46565, 64 parts incl. the mandible) + brain: what rides on C1."""
    from inob.geometry.composites import default_manifest_path, load_composite
    skull = load_composite("FMA46565", mesh_root=root / "internal_meshes",
                           manifest_path=default_manifest_path(root))
    skull.apply_transform(T)
    return trimesh.util.concatenate([skull, brain])


def build_neck(root: Path, T: np.ndarray, cord: trimesh.Trimesh,
               brain: trimesh.Trimesh) -> NeckModel:
    lm = load_landmarks(root, T)
    # Full head rotation at and above the top of C1: the skull sits on the atlas.
    z_top = float(lm["atlas"].bounds[1, 2])
    # Nothing below T1 turns -- that is the thorax, and the heart is in it.
    z_bot = float(lm["t1"].centroid[2])
    # Pivot at the atlanto-axial joint: the dens of C2 inside the ring of C1.
    pivot_z = float(lm["axis"].bounds[1, 2])
    logger.info("[landmarks] C1 top %.1f, C2 dens %.1f, T1 %.1f mm", z_top, pivot_z, z_bot)
    return NeckModel.from_anatomy(cord=cord, skull_base_z=z_top, t1_z=z_bot,
                                  pivot_z=pivot_z,
                                  head_block=head_block(root, T, brain))


def deform_all(meshes, sensors, neck: NeckModel, ypr):
    out = {}
    for name, m in meshes.items():
        rule = RULE.get(name, ("#888", "fixed"))[1]
        if rule == "fixed":
            out[name] = m
        elif rule == "head":
            out[name] = neck.deform_mesh(m, ypr, rigid=True)
        elif rule == "body":
            out[name] = neck.deform_mesh(m, ypr, per_body=True)
        else:
            out[name] = neck.deform_mesh(m, ypr)
    sp, so = sensors
    return out, (neck.deform_points(sp, ypr, weights=1.0),
                 neck.deform_directions(so, sp, ypr, weights=1.0))


def outline(m: trimesh.Trimesh, axis: int, coord: float):
    n = np.zeros(3); n[axis] = 1.0
    sec = m.section(plane_origin=n * coord, plane_normal=n)
    if sec is None:
        return []
    keep = [i for i in range(3) if i != axis]
    V = np.asarray(sec.vertices)[:, keep]
    return [V[e.points] for e in sec.entities if len(e.points) > 1]


def panel(ax, meshes, sens, *, axis, coord, show=("skin", "bone", "heart")):
    for name in show:
        m = meshes.get(name)
        if m is None:
            continue
        col = RULE[name][0]
        lw = 1.5 if name in ("skin", "heart") else 0.6
        for pl in outline(m, axis, coord):
            ax.plot(pl[:, 0], pl[:, 1], color=col, lw=lw, zorder=3, solid_capstyle="round")
    keep = [i for i in range(3) if i != axis]
    ax.scatter(sens[:, keep[0]], sens[:, keep[1]], s=3, c="#1b7f3b", zorder=20)
    ax.set_aspect("equal")
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--dir", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=165)
    a = p.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(message)s")

    d = a.dir or a.root / "outputs/cfa_thorax"
    meshes = {q.stem: trimesh.load(q, process=True) for q in sorted(d.glob("*.stl"))}
    T = np.loadtxt(d / "bp3d_to_mni.txt", delimiter=",")
    ref = np.load(a.root / "outputs/_cfa_ref.npz")
    sensors = (ref["chanpos"] * 1000.0, ref["chanori"])
    neck = build_neck(a.root, T, meshes["spinal_cord"], meshes["brain"])

    angles = DESIGN.orientations()
    names = ["still"] + [f"{y:+.0f}/{p_:+.0f}/{r:+.0f}" for y, p_, r in angles[1:]]
    heart_c = np.asarray(meshes["heart"].centroid)

    hb_mesh = head_block(a.root, T, meshes["brain"])
    head_pts = np.asarray(hb_mesh.vertices, float)
    states, rows = [], []
    for ypr, nm in zip(angles, names):
        dm, (sp, so) = deform_all(meshes, sensors, neck, ypr)
        states.append((nm, ypr, dm, sp))
        # sensors must stay outside the deformed body
        sd = trimesh.proximity.signed_distance(dm["skin"], sp)
        # The head must move as ONE body: check that every point of the head
        # block gets the same share of the rotation as the array does.
        hb = head_pts
        w_head = neck.weight(hb)
        rows.append({
            "name": nm, "ypr": np.asarray(ypr).tolist(),
            "phys_fraction": round(physiological_fraction(ypr), 3),
            "sensors_inside_skin": int((sd > 0).sum()),
            "min_clearance_mm": round(float(-sd.max()), 1),
            "heart_to_nearest_sensor_mm": round(
                float(np.linalg.norm(sp - heart_c, axis=1).min()), 1),
            "array_centroid_shift_mm": round(
                float(np.linalg.norm(sp.mean(0) - sensors[0].mean(0))), 1),
            "head_block_min_share": round(float(w_head.min()), 3),
        })

    (d / "posture.json").write_text(json.dumps(
        {"neck": {"pivot_mm": neck.pivot.round(1).tolist(),
                  "blend_mm": [neck.z_bot, neck.z_top]}, "orientations": rows}, indent=2))

    print(f"\n{'orientation':>16s} {'phys':>5s} {'sens in skin':>12s} {'clearance':>10s} "
          f"{'heart->sens':>12s} {'array shift':>12s} {'head rigid':>11s}")
    for r in rows:
        print(f"{r['name']:>16s} {r['phys_fraction']:5.2f} {r['sensors_inside_skin']:12d} "
              f"{r['min_clearance_mm']:9.1f} {r['heart_to_nearest_sensor_mm']:12.1f} "
              f"{r['array_centroid_shift_mm']:12.1f} {r['head_block_min_share']:11.3f}")

    for tag, axis, coord, sub in (("axial", 2, -120.0, "neck, z = -120 mm  (view from above)"),
                                  ("sagittal", 0, 0.0, "sagittal, x = 0 mm")):
        fig, axes = plt.subplots(3, 3, figsize=(11.5, 11.8))
        for ax, (nm, ypr, dm, sp) in zip(axes.ravel(), states):
            if tag == "axial":
                panel(ax, dm, sp, axis=2, coord=coord, show=("skin", "bone"))
                for z in (40.0, -300.0):
                    for pl in outline(dm["skin"], 2, z):
                        ax.plot(pl[:, 0], pl[:, 1], color="#c3c8ce", lw=0.9, ls="--", zorder=1)
            else:
                panel(ax, dm, sp, axis=0, coord=coord,
                      show=("skin", "bone", "spinal_cord", "heart"))
            ax.set_title(nm, fontsize=10, loc="left")
        for ax in axes.ravel()[len(states):]:
            ax.axis("off")
        h = [Line2D([], [], color=RULE[k][0], lw=2, label=k) for k in ("skin", "bone")]
        h += [Line2D([], [], marker="o", ls="", mfc="#1b7f3b", mec="#1b7f3b", ms=4,
                     label="OPM array (head-fixed)")]
        if tag == "axial":
            h += [Line2D([], [], color="#c3c8ce", lw=1, ls="--", label="skin at z=+40 / -300")]
        else:
            h += [Line2D([], [], color=RULE["heart"][0], lw=2, label="heart (torso-fixed)")]
        fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=10)
        fig.suptitle(f"Head-on-neck rotation, {sub}\npivot {neck.pivot.round(0).tolist()} mm "
                     f"(atlanto-axial), blend {neck.z_bot:.0f} to {neck.z_top:.0f} mm",
                     fontsize=12.5, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0.05, 1, 0.93))
        out = d / f"posture_{tag}.png"
        fig.savefig(out, dpi=a.dpi); plt.close(fig)
        print("figure:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
