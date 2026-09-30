#!/usr/bin/env python3
"""Build the head+thorax conductor at every head orientation and solve leadfields.

    python scripts/build_cfa_forward.py                 # all 9 orientations
    python scripts/build_cfa_forward.py --only 0        # just the still head
    python scripts/build_cfa_forward.py --maxvol 25 --radbound 3
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inob.cfa.array import CFAArray
from inob.cfa.design import DESIGN
from inob.cfa.fem import FemSpec
from inob.cfa.forward import SolverOptions
from inob.cfa.pipeline import save_orientation, solve_orientation
from inob.cfa.posture import NeckModel
from inob.cfa.sources import (
    CorticalSource, MyocardialSource, equivalent_dipoles, load_cdm,
)
from inob.geometry.composites import default_manifest_path, load_composite

logger = logging.getLogger("build_cfa_forward")

LANDMARKS = {"atlas": "FJ3176", "axis": "FJ3177", "t1": "FJ3158"}
CDM_MAT = Path("/Users/olivercase/Desktop/CFA/replication pack/data/cache/"
               "cdm_correction.mat")


def _fj_index(root: Path) -> dict[str, str]:
    idx: dict[str, str] = {}
    for p in glob.glob(str(root / "internal_meshes/*/*.obj")):
        idx.setdefault(Path(p).name.split("_", 1)[0], p)
    return idx


def build_neck(root: Path, T: np.ndarray, cord, brain) -> NeckModel:
    idx = _fj_index(root)
    lm = {}
    for name, fj in LANDMARKS.items():
        m = trimesh.load(idx[fj], process=True)
        m.apply_transform(T)
        lm[name] = m
    skull = load_composite("FMA46565", mesh_root=root / "internal_meshes",
                           manifest_path=default_manifest_path(root))
    skull.apply_transform(T)
    return NeckModel.from_anatomy(
        cord=cord,
        skull_base_z=float(lm["atlas"].bounds[1, 2]),
        t1_z=float(lm["t1"].centroid[2]),
        pivot_z=float(lm["axis"].bounds[1, 2]),
        head_block=trimesh.util.concatenate([skull, brain]),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--pitch", type=float, default=3.0)
    p.add_argument("--radbound", type=float, default=3.0, help="mm")
    p.add_argument("--maxvol", type=float, default=25.0, help="mm^3")
    p.add_argument("--n-myo", type=int, default=DESIGN.n_myocardial_dipoles)
    p.add_argument("--only", type=int, default=None, help="orientation index")
    p.add_argument("--cdm", type=Path, default=CDM_MAT)
    a = p.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    root = a.root.resolve()
    src_dir = root / "outputs/cfa_thorax"
    out = (a.out or root / "outputs/cfa_forward").resolve()
    out.mkdir(parents=True, exist_ok=True)

    meshes = {q.stem: trimesh.load(q, process=True) for q in sorted(src_dir.glob("*.stl"))}
    T = np.loadtxt(src_dir / "bp3d_to_mni.txt", delimiter=",")
    mr, mp = root / "internal_meshes", default_manifest_path(root)

    myo_mesh = load_composite("myocardium", mesh_root=mr, manifest_path=mp)
    myo_mesh.apply_transform(T)
    ins_mesh = load_composite("insula_right", mesh_root=mr, manifest_path=mp)
    ins_mesh.apply_transform(T)

    array = CFAArray.from_reference(root / "outputs/_cfa_ref.npz")
    neck = build_neck(root, T, meshes["spinal_cord"], meshes["brain"])
    myo = MyocardialSource.from_wall(myo_mesh, n_sites=a.n_myo)
    cdm = load_cdm(a.cdm)
    eq = equivalent_dipoles(myo, cdm["q"],
                            offset_mm=1e3 * np.asarray(DESIGN.heart_offset_m))
    insula = CorticalSource.from_mesh(ins_mesh)

    spec = FemSpec(pitch_mm=a.pitch, radbound_mm=a.radbound, maxvol_mm3=a.maxvol)
    opts = SolverOptions()

    angles = DESIGN.orientations()
    todo = [a.only] if a.only is not None else list(range(len(angles)))
    meta = []
    for i in todo:
        r = solve_orientation(
            meshes=meshes, neck=neck, array=array, myo=myo, eq_pos_mm=eq,
            insula=insula, ypr=angles[i], fem_spec=spec, solver_opts=opts)
        save_orientation(out / f"orient_{i:02d}.npz", r)
        meta.append({"index": i, **r.meta()})
        logger.info("--- orientation %d done in %.0f s", i, r.seconds)

    # source geometry is orientation-independent; store once
    np.savez_compressed(out / "sources.npz",
                        myo_pos_mm=myo.pos_mm, myo_ori=myo.ori,
                        myo_area_mm2=myo.area_mm2, eq_pos_mm=eq,
                        insula_pos_mm=insula.pos_mm, q=cdm["q"], q6=cdm["q6"],
                        time=cdm["time"], erp_index=cdm["erp_index"],
                        erp_latency=cdm["erp_latency"],
                        chanori_nominal=array.chanori,
                        site_pos_mm=array.site_pos_mm)
    mp_out = out / "forward_meta.json"
    prev = json.loads(mp_out.read_text()) if mp_out.exists() else []
    keep = [m for m in prev if m["index"] not in {m2["index"] for m2 in meta}]
    mp_out.write_text(json.dumps(sorted(keep + meta, key=lambda m: m["index"]),
                                 indent=2))
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
