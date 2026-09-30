"""Build one volume conductor per head orientation and solve its leadfields.

This is the expensive stage and the one the whole 2x2 rests on. For each head
orientation in the design it deforms the anatomy (:mod:`inob.cfa.posture`),
meshes it (:mod:`inob.cfa.fem`), and solves three leadfields against the
triaxial array (:mod:`inob.cfa.forward`):

``B_myo``   the **distributed truth**: field vector at every site for every
            myocardial site's x/y/z moment. Torso-fixed, so the source positions
            do not move -- what changes with orientation is the conductor around
            them and where the array is.
``B_eq``    the **estimator's model**: the two equivalent dipoles, at their
            template positions plus the registration offset. Solved in the same
            conductor, because a real analyst has one head model and knows the
            tracked orientation.
``B_ins``   the insula (HEF) source. Head-fixed: the source rotates rigidly with
            the brain, the skull and the array.

**The insula leadfield is solved once and reused.** Array and brain move as one
rigid body, so the head-local geometry -- source position, skull, scalp, sensor
standoff -- is identical at every orientation, and the HEF topography must not
drift: that invariance is exactly what makes the joint GLM identifiable. Holding
it fixed by construction is the model. The per-orientation solve is still done,
but only to *measure* the residual (the neck and torso below the head do not
rotate, so return-current paths differ slightly) and report it as a validation
number the replication pack could not produce -- its head and torso were
separate conductors, so it had to assume this rather than check it.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import trimesh

from inob.cfa.array import CFAArray
from inob.cfa.design import CONDUCTIVITY_SM, Design
from inob.cfa.fem import FemSpec, build_conductor_fem
from inob.cfa.forward import ConductorSolver, SolverOptions
from inob.cfa.posture import NeckModel, physiological_fraction
from inob.cfa.sources import CorticalSource, MyocardialSource

logger = logging.getLogger(__name__)

__all__ = ["OrientationResult", "solve_orientation", "DEFORM_RULE"]

#: How each compartment follows the head. See :mod:`inob.cfa.posture`.
DEFORM_RULE: dict[str, str] = {
    "skin": "soft", "spinal_cord": "soft", "muscle": "soft",
    "bone": "body",
    "brain": "head",
    "heart": "fixed", "lung_left": "fixed", "lung_right": "fixed",
}


def deform_compartments(meshes: dict[str, trimesh.Trimesh], neck: NeckModel,
                        ypr) -> dict[str, trimesh.Trimesh]:
    out = {}
    for name, m in meshes.items():
        rule = DEFORM_RULE.get(name, "fixed")
        if rule == "fixed":
            out[name] = m
        elif rule == "head":
            out[name] = neck.deform_mesh(m, ypr, rigid=True)
        elif rule == "body":
            out[name] = neck.deform_mesh(m, ypr, per_body=True)
        else:
            out[name] = neck.deform_mesh(m, ypr)
    return out


@dataclass
class OrientationResult:
    """Leadfields and diagnostics for one head orientation."""

    ypr_deg: tuple[float, float, float]
    B_myo: np.ndarray        # (S, 3, 3 * n_myo)
    B_eq: np.ndarray         # (S, 3, 6)
    B_ins: np.ndarray        # (S, 3, 3)
    chanori: np.ndarray      # (3S, 3) nominal axes, rotated with the head
    site_pos_mm: np.ndarray  # (S, 3)
    n_nodes: int
    n_tets: int
    seconds: float

    def meta(self) -> dict:
        return {
            "ypr_deg": list(map(float, self.ypr_deg)),
            "physiological_fraction": round(physiological_fraction(self.ypr_deg), 3),
            "n_nodes": int(self.n_nodes), "n_tets": int(self.n_tets),
            "seconds": round(self.seconds, 1),
        }


def solve_orientation(
    *,
    meshes: dict[str, trimesh.Trimesh],
    neck: NeckModel,
    array: CFAArray,
    myo: MyocardialSource,
    eq_pos_mm: np.ndarray,
    insula: CorticalSource,
    ypr,
    fem_spec: FemSpec | None = None,
    solver_opts: SolverOptions | None = None,
) -> OrientationResult:
    """Deform, mesh and solve one head orientation."""
    t0 = time.time()
    ypr = np.asarray(ypr, float).ravel()
    tag = "{:+.0f}/{:+.0f}/{:+.0f}".format(*ypr)
    logger.info("=== orientation %s (%.2f x cervical ROM)", tag,
                physiological_fraction(ypr))

    dm = deform_compartments(meshes, neck, ypr)
    arr = array.transformed(
        lambda p, d: (neck.deform_points(p, ypr, weights=1.0),
                      neck.deform_directions(d, p, ypr, weights=1.0)))
    # The insula rides with the brain; the heart does not move at all.
    ins_pos = neck.deform_points(insula.pos_mm[None], ypr, weights=1.0)

    fem = build_conductor_fem(dm, fem_spec or FemSpec())
    sol = ConductorSolver.for_sites(fem, CONDUCTIVITY_SM, arr.site_pos_mm,
                                    options=solver_opts)
    B_myo = sol.site_field(myo.pos_mm)
    B_eq = sol.site_field(eq_pos_mm)
    B_ins = sol.site_field(ins_pos)

    return OrientationResult(
        ypr_deg=tuple(ypr), B_myo=B_myo, B_eq=B_eq, B_ins=B_ins,
        chanori=arr.chanori, site_pos_mm=arr.site_pos_mm,
        n_nodes=len(fem.nodes), n_tets=len(fem.tets), seconds=time.time() - t0,
    )


def save_orientation(path: Path, r: OrientationResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, ypr_deg=np.asarray(r.ypr_deg), B_myo=r.B_myo, B_eq=r.B_eq,
        B_ins=r.B_ins, chanori=r.chanori, site_pos_mm=r.site_pos_mm,
        n_nodes=r.n_nodes, n_tets=r.n_tets, seconds=r.seconds,
    )
    logger.info("[saved] %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path
