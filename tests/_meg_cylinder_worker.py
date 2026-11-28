"""Worker for tests/test_meg_cylinder_real_duneuro.py: a homogeneous finite cylinder.

A dipole lying along the axis of any axially symmetric conductor makes no
magnetic field outside it: the currents are symmetric about the axis, so B is
azimuthal, and by Ampere's law its circulation round a circle outside the
conductor equals the net current through that circle's disc, which is zero.
This is an exact, non-spherical reference, and it is the geometry of a
cord-aligned source in the neck. With the volume-current field added instead
of subtracted the result is about twice the free-space field.

Usage: python tests/_meg_cylinder_worker.py OUT.npz
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

RADIUS, HALF_LENGTH, PITCH = 60.0, 150.0, 2.0


def build_cylinder_fem():
    import iso2mesh as im

    from inob.io.hdf5 import FemMesh

    span_r, span_z = RADIUS + 20.0, HALF_LENGTH + 20.0
    nx = int(np.ceil(2 * span_r / PITCH)) + 1
    nz = int(np.ceil(2 * span_z / PITCH)) + 1
    mn = np.array([-span_r, -span_r, -span_z])
    xs = mn[0] + (np.arange(nx) + 0.5) * PITCH
    zs = mn[2] + (np.arange(nz) + 0.5) * PITCH
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    vol = ((X**2 + Y**2 <= RADIUS**2) & (np.abs(Z) <= HALF_LENGTH)).astype(np.uint8)
    node, elem, _ = im.cgalv2m(vol, 1.5, 4.0)
    tets = np.asarray(elem, dtype=np.int64)[:, :4] - 1
    return FemMesh(
        nodes=(node[:, :3] * PITCH + mn).astype(np.float64),
        tets=tets.astype(np.int32),
        tissue=np.ones(len(tets), dtype=np.int32),
        tissue_labels=("homogeneous",),
        unit="mm",
    )


def main() -> None:
    out = sys.argv[1]
    import duneuropy as dp

    from inob.config import SOURCE_MODEL_TYPES, SourceModelCfg
    from inob.forward.duneuro_driver import build_source_model_config, compute_meg_field_components

    fem = build_cylinder_fem()
    # Triaxial sites on two rings round the cylinder, 15 mm outside it.
    th = 2 * np.pi * np.arange(8) / 8
    site = np.array(
        [[(RADIUS + 15) * np.cos(a), (RADIUS + 15) * np.sin(a), z] for z in (0.0, 40.0) for a in th]
    )
    pos = np.vstack([site, site, site])
    ori = np.repeat(np.eye(3), len(site), axis=0)
    # On the axis, and 25 mm off it.
    src = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 30.0], [25.0, 0.0, 0.0]])

    cfg = {
        "type": "fitted",
        "solver_type": "cg",
        "element_type": "tetrahedron",
        "post_process": "false",
        "post_process_meg": "true",
        "subtract_mean": "false",
        "solver": {
            "reduction": "1e-10",
            "edge_norm_type": "houston",
            "penalty": "20",
            "scheme": "sipg",
            "weights": "tensorOnly",
        },
        "volume_conductor": {
            "grid": {"nodes": fem.nodes, "elements": fem.tets.astype(np.int64)},
            "tensors": {
                "labels": np.zeros(len(fem.tets), dtype=np.int64),
                "conductivities": np.array([0.33e-3]),
            },
        },
        "meg": {"intorderadd": "5", "type": "physical"},
    }
    drv = dp.MEEGDriver3d(cfg)
    drv.setCoilsAndProjections(
        [dp.FieldVector3D(p) for p in pos], [[dp.FieldVector3D(o)] for o in ori]
    )
    T = np.array(drv.computeMEGTransferMatrix(cfg)[0])
    dips = [dp.Dipole3d(p, e) for p in src for e in np.eye(3)]
    save = {"n_tets": np.array(len(fem.tets)), "src": src, "site": site}
    for name in SOURCE_MODEL_TYPES:
        ns = SimpleNamespace(forward=SimpleNamespace(source_model=SourceModelCfg(type=name)))
        cfg["source_model"] = build_source_model_config(ns)
        save[f"pri__{name}"], save[f"sec__{name}"] = compute_meg_field_components(
            drv, T, dips, cfg
        )
    np.savez(out, **save)
    print("WORKER DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
