#!/usr/bin/env python3
"""Build the head+thorax volume conductor for the CFA work, aligned to the OPM array.

Two stages, both auditable:

1. **Register.** Fit the BodyParts3D head skin rigidly onto the SPM canonical MNI
   scalp and report the residual. Rigid, not similarity: the extra 2.8% scale buys
   0.4 mm of head fit and would distort every organ size in the torso.
2. **Build.** Transform, clip to a head+thorax band, cap every cut, and
   watertighten each compartment.

Outputs land in ``outputs/cfa_thorax/``: per-compartment STLs, ``summary.json``
(topology, placement, conductivity), ``bp3d_to_mni.txt`` (the 4x4) and
``registration.json`` (its residual).

    python scripts/build_cfa_thorax.py [--z-lo -600] [--keep-arms] [--no-bone]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inob.geometry.cfa_thorax import ThoraxSpec, build_thorax, summarise, write_summary
from inob.geometry.register import residual_mm
from inob.mesh.clip import clip_band

logger = logging.getLogger("build_cfa_thorax")

SPM_SCALP = Path("/Users/olivercase/Documents/Matlab_repos/spm-main/canonical/scalp_2562.surf.gii")


def load_gii(path: Path) -> trimesh.Trimesh:
    g = nib.load(str(path))
    v = [a for a in g.darrays if a.intent == 1008][0].data
    f = [a for a in g.darrays if a.intent == 1009][0].data
    return trimesh.Trimesh(np.asarray(v, float), np.asarray(f, np.int64), process=True)


def fit_head_to_mni(skin: trimesh.Trimesh, scalp: trimesh.Trimesh,
                    *, scale: bool = False, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Rigid BodyParts3D -> MNI, from head skin onto the canonical scalp.

    The initial guess is deterministic, not a principal-axis search: BodyParts3D
    puts right-sided structures at negative x and anterior structures at negative
    y (checked against the right hip and frontal bones), so the atlas is MNI
    rotated 180 deg about z. Centroid alignment then puts it within ~8 mm, and
    ICP refines. A principal-axis search on a head is front/back ambiguous and
    was landing on a 25 mm / 0.75-scale solution.

    The source is clipped to the scalp's own z-span so the two surfaces cover the
    same anatomy — including the neck (which the scalp mesh has none of) is what
    made ICP shrink the head to compensate.
    """
    span = float(scalp.bounds[1, 2] - scalp.bounds[0, 2])
    head = clip_band(skin, lo=float(skin.bounds[1, 2]) - span, label="bp3d_head")

    init = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
    init[:3, 3] = scalp.centroid - trimesh.transform_points(head.centroid[None], init)[0]
    rms0, max0 = residual_mm(head, scalp, init, seed=seed)
    logger.info("[register] init (180deg + centroid): rms=%.2f max=%.2f mm", rms0, max0)

    pts, _ = trimesh.sample.sample_surface(head, 6000, seed=seed)
    T, _, _ = trimesh.registration.icp(np.asarray(pts), scalp, initial=init,
                                       scale=scale, max_iterations=200, threshold=1e-6)
    det = float(np.linalg.det(T[:3, :3]))
    if det <= 0:
        raise ValueError(f"fitted transform is a reflection (det={det:+.4f})")
    rms, mx = residual_mm(head, scalp, T, seed=seed)
    info = {
        "rms_mm": rms, "max_mm": mx, "scale": float(np.cbrt(abs(det))),
        "det": det, "allow_scale": bool(scale),
        "init_rms_mm": rms0, "source_faces": int(len(head.faces)),
        "target": str(SPM_SCALP), "n_icp_points": int(len(pts)),
    }
    logger.info("[register] icp: rms=%.2f max=%.2f scale=%.4f", rms, mx, info["scale"])
    return T, info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--z-lo", type=float, default=-600.0,
                   help="inferior clip of the body in MNI mm (default -600)")
    p.add_argument("--keep-arms", action="store_true", help="do not truncate the T-pose arms")
    p.add_argument("--arm-halfwidth", type=float, default=200.0)
    p.add_argument("--allow-scale", action="store_true",
                   help="fit a similarity instead of a rigid transform")
    p.add_argument("--no-bone", action="store_true")
    p.add_argument("--no-cord", action="store_true")
    p.add_argument("--scalp", type=Path, default=SPM_SCALP)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)

    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    root = a.project_root.resolve()
    out = (a.out or root / "outputs/cfa_thorax").resolve()
    out.mkdir(parents=True, exist_ok=True)

    skin = trimesh.load(root / "data/torso/FJ2810_BP22617_FMA7163_Skin.stl", process=True)
    scalp = load_gii(a.scalp)
    T, reg = fit_head_to_mni(skin, scalp, scale=a.allow_scale)
    np.savetxt(out / "bp3d_to_mni.txt", T, fmt="%.10g", delimiter=",")
    (out / "registration.json").write_text(json.dumps(reg, indent=2))

    spec = ThoraxSpec(
        project_root=root, transform=T, z_lo=a.z_lo,
        x_halfwidth=None if a.keep_arms else a.arm_halfwidth,
        include_bone=not a.no_bone, include_cord=not a.no_cord,
    )
    meshes = build_thorax(spec)
    for name, m in meshes.items():
        m.export(out / f"{name}.stl")
    write_summary(meshes, out / "summary.json")

    print(f"\n{'compartment':14s} {'V':>8s} {'F':>8s} {'bodies':>7s} {'wtight':>7s} "
          f"{'euler':>6s} {'vol_cm3':>9s}  centroid (MNI mm)")
    for name, s in summarise(meshes).items():
        vol = s["volume_mm3"] / 1000.0
        print(f"{name:14s} {s['vertices']:8d} {s['faces']:8d} {s['bodies']:7d} "
              f"{str(s['watertight']):>7s} {s['euler']:6d} {vol:9.1f}  {s['centroid_mm']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
