#!/usr/bin/env python3
"""Pick the FEM resolution by measuring convergence, not by guessing.

Builds the still-head conductor at several CGAL settings and solves the two
leadfields the study actually uses -- the insula (HEF) source and the distributed
myocardial source -- against the triaxial array. Each level is compared with the
finest one built, so the reported error is a discretisation error, not a
guess.

    python scripts/cfa_mesh_convergence.py
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inob.cfa.array import CFAArray
from inob.cfa.design import CONDUCTIVITY_SM, DESIGN
from inob.cfa.fem import FemSpec, build_conductor_fem
from inob.cfa.forward import ConductorSolver
from inob.cfa.sources import (
    CorticalSource, MyocardialSource, equivalent_dipoles, load_cdm,
)
from inob.geometry.composites import default_manifest_path, load_composite

logger = logging.getLogger("cfa_mesh_convergence")

#: (radbound mm, maxvol mm^3). Coarse to fine.
LEVELS = [(8.0, 300.0), (7.0, 200.0), (6.0, 120.0), (5.0, 70.0)]

CDM_MAT = Path("/Users/olivercase/Desktop/CFA/replication pack/data/cache/"
               "cdm_correction.mat")


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--pitch", type=float, default=3.0)
    p.add_argument("--n-myo", type=int, default=100,
                   help="fewer sites than production: this measures the mesh, "
                        "not the source model")
    a = p.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    root = a.root.resolve()
    d = root / "outputs/cfa_thorax"
    meshes = {q.stem: trimesh.load(q, process=True) for q in sorted(d.glob("*.stl"))}
    T = np.loadtxt(d / "bp3d_to_mni.txt", delimiter=",")
    mr, mp = root / "internal_meshes", default_manifest_path(root)
    myo_mesh = load_composite("myocardium", mesh_root=mr, manifest_path=mp)
    myo_mesh.apply_transform(T)
    ins_mesh = load_composite("insula_right", mesh_root=mr, manifest_path=mp)
    ins_mesh.apply_transform(T)

    array = CFAArray.from_reference(root / "outputs/_cfa_ref.npz")
    myo = MyocardialSource.from_wall(myo_mesh, n_sites=a.n_myo)
    cdm = load_cdm(CDM_MAT)
    eq = equivalent_dipoles(myo, cdm["q"],
                            offset_mm=1e3 * np.asarray(DESIGN.heart_offset_m))
    insula = CorticalSource.from_mesh(ins_mesh)
    amps = myo.amplitudes(cdm["q"])
    k = int(np.linalg.norm(cdm["q"], axis=0).argmax())

    rows = []
    for radbound, maxvol in LEVELS:
        t0 = time.time()
        spec = FemSpec(pitch_mm=a.pitch, radbound_mm=radbound, maxvol_mm3=maxvol)
        fem = build_conductor_fem(meshes, spec)
        t_mesh = time.time() - t0
        t1 = time.time()
        sol = ConductorSolver.for_sites(fem, CONDUCTIVITY_SM, array.site_pos_mm)
        b_ins = sol.site_field(insula.pos_mm[None])
        b_myo = sol.site_field(myo.pos_mm)
        b_eq = sol.site_field(eq)
        t_solve = time.time() - t1
        # CFA at the R peak, as the array actually reads it. b_myo is
        # (site, field component, source-major x/y/z moment): contract each
        # site's three moment columns with its wall normal, then project the
        # per-site field vector onto the 255 channel axes and drive it with the
        # site amplitudes at the R peak.
        per_site = np.einsum("scnk,nk->scn",
                             b_myo.reshape(len(b_myo), 3, -1, 3), myo.ori)
        cfa = array.project(per_site) @ amps[:, k]
        del sol
        gc.collect()
        rows.append({
            "radbound_mm": radbound, "maxvol_mm3": maxvol,
            "n_nodes": len(fem.nodes), "n_tets": len(fem.tets),
            "mesh_s": round(t_mesh, 1), "solve_s": round(t_solve, 1),
            "_b_ins": b_ins, "_b_myo": b_myo, "_b_eq": b_eq, "_cfa": cfa,
        })
        logger.info("level r=%.1f v=%.0f -> %d tets, mesh %.0f s, solve %.0f s, "
                    "CFA rms %.0f fT", radbound, maxvol, len(fem.tets),
                    t_mesh, t_solve, 1e15 * np.sqrt((cfa ** 2).mean()))

    ref = rows[-1]
    print(f"\n{'radbound':>9s} {'maxvol':>7s} {'tets':>9s} {'mesh s':>7s} "
          f"{'solve s':>8s} {'err ins':>9s} {'err myo':>9s} {'err eq':>8s} "
          f"{'err CFA':>8s}")
    out = []
    for r in rows:
        e = {k: rel(r[f"_{k}"], ref[f"_{k}"]) for k in ("b_ins", "b_myo", "b_eq", "cfa")}
        print(f"{r['radbound_mm']:9.1f} {r['maxvol_mm3']:7.0f} {r['n_tets']:9d} "
              f"{r['mesh_s']:7.1f} {r['solve_s']:8.1f} {e['b_ins']:9.2%} "
              f"{e['b_myo']:9.2%} {e['b_eq']:8.2%} {e['cfa']:8.2%}")
        out.append({k: v for k, v in r.items() if not k.startswith("_")} |
                   {f"rel_err_{k}": round(v, 5) for k, v in e.items()})
    (root / "outputs/cfa_forward").mkdir(parents=True, exist_ok=True)
    (root / "outputs/cfa_forward/mesh_convergence.json").write_text(
        json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
