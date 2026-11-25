"""Worker for tests/test_meg_secondary_sign_real_duneuro.py.

Runs every real-DUNEuro solve the sign tests need in ONE process and writes an
npz. It is a separate process because DUNEuro can abort while tearing a grid
down; the npz is written first and the process then exits without teardown, so
a crash there cannot lose results or kill the pytest session.

Usage: python tests/_meg_sign_worker.py OUT.npz [maxvol]
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np


def geometry() -> dict[str, np.ndarray]:
    """Coils and sources. All coils are outside the 100 mm sphere."""
    rng = np.random.default_rng(11)
    u = rng.normal(size=(400, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u = u[u[:, 2] > 0.1][:12]
    site = u * 120.0
    # Triaxial at every site: radial, then two tangential directions.
    t1 = np.cross(u, [0.0, 0.0, 1.0])
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(u, t1)
    pos = [site, site, site]
    ori = [u, t1, t2]
    kind = [np.zeros(12, int), np.ones(12, int), np.ones(12, int)]  # 0 radial, 1 tangential

    # Two closed loops of tangential coils, for the Ampere circulation test.
    def loop(centre, a, b, radius, n):
        th = 2 * np.pi * np.arange(n) / n
        p = centre + radius * (np.outer(np.cos(th), a) + np.outer(np.sin(th), b))
        t = -np.outer(np.sin(th), a) + np.outer(np.cos(th), b)
        return p, t

    ex, ey, ez = np.eye(3)
    pA, tA = loop(np.array([0.0, 0.0, 135.0]), ex, ey, 40.0, 12)  # small, above the sphere
    pB, tB = loop(np.zeros(3), ey, ez, 125.0, 16)  # great circle round the sphere
    pos += [pA, pB]
    ori += [tA, tB]
    kind += [np.full(12, 2), np.full(16, 3)]

    # Sources at four depths, none on a symmetry axis.
    d = np.array([[0.3, 0.2, 0.93], [-0.5, 0.3, 0.81], [0.2, -0.6, 0.77], [0.45, 0.45, 0.77]])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    src = d * np.array([30.0, 50.0, 70.0, 85.0])[:, None]
    return {
        "coilpos": np.vstack(pos),
        "coilori": np.vstack(ori),
        "kind": np.concatenate(kind),
        "src": src,
    }


def source_model_cfgs() -> dict[str, dict[str, str]]:
    """The DUNEuro source-model dicts exactly as the production code builds them."""
    from inob.config import SOURCE_MODEL_TYPES, SourceModelCfg
    from inob.forward.duneuro_driver import build_source_model_config

    out = {}
    for name in SOURCE_MODEL_TYPES:
        cfg = SimpleNamespace(forward=SimpleNamespace(source_model=SourceModelCfg(type=name)))
        out[name] = build_source_model_config(cfg)
    return out


def driver_cfg(fem, labels: np.ndarray, sigma_S_per_m: np.ndarray) -> dict:
    return {
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
                "labels": labels.astype(np.int64),
                "conductivities": np.asarray(sigma_S_per_m, float) * 1e-3,
            },
        },
        "meg": {"intorderadd": "5", "type": "physical"},
    }


def main() -> None:
    out = sys.argv[1]
    maxvol = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    import duneuropy as dp

    from inob.analysis.sphere_calibration import build_sphere_fem
    from inob.forward.duneuro_driver import MEG_MM_MODE_TO_SI, compute_meg_leadfield

    g = geometry()
    fem = build_sphere_fem(radius_mm=100.0, pitch_mm=2.0, radbound=1.5, maxvol=maxvol)
    eye = np.eye(3)
    dips = [dp.Dipole3d(p, e) for p in g["src"] for e in eye]
    keep = []  # hold every driver until exit: no teardown before the npz is written

    def solve(labels, sigma, coil_idx, models):
        cfg = driver_cfg(fem, labels, sigma)
        drv = dp.MEEGDriver3d(cfg)
        keep.append(drv)
        drv.setCoilsAndProjections(
            [dp.FieldVector3D(p) for p in g["coilpos"][coil_idx]],
            [[dp.FieldVector3D(o)] for o in g["coilori"][coil_idx]],
        )
        T = np.array(drv.computeMEGTransferMatrix(cfg)[0])
        res = {}
        for name, sm in models.items():
            cfg["source_model"] = sm
            sec, _ = drv.applyMEGTransfer(T, dips, cfg)
            pri = drv.computeMEGPrimaryField(dips, cfg)
            res[name] = {
                "pri": np.column_stack([np.asarray(f) for f in pri]) * MEG_MM_MODE_TO_SI,
                "sec": np.column_stack([np.asarray(f) for f in sec]) * MEG_MM_MODE_TO_SI,
                # the production combination, whatever sign it currently uses
                "L": compute_meg_leadfield(drv, T, dips, cfg),
            }
        return res

    models = source_model_cfgs()
    save: dict[str, np.ndarray] = dict(g)
    save["n_tets"] = np.array(len(fem.tets))
    save["models"] = np.array(list(models))

    all_idx = np.arange(len(g["coilpos"]))
    homog = np.zeros(len(fem.tets), int)
    for name, r in solve(homog, [0.33], all_idx, models).items():
        for k, v in r.items():
            save[f"{k}__{name}"] = v
    np.savez(out, **save)  # checkpoint: the main result is safe from here on

    # Same sphere, different conductivity, on a subset of the triaxial sites.
    sub = np.concatenate([np.arange(0, 4), np.arange(12, 16), np.arange(24, 28)])
    save["sub"] = sub
    pi = {"partial_integration": models["partial_integration"]}
    save["L_sigma1"] = solve(homog, [1.0], sub, pi)["partial_integration"]["L"]
    np.savez(out, **save)

    # Layered sphere (brain / skull-like / scalp-like shells by element centroid).
    rc = np.linalg.norm(fem.nodes[fem.tets].mean(axis=1), axis=1)
    lab = np.where(rc < 80.0, 0, np.where(rc < 90.0, 1, 2))
    save["L_layered"] = solve(lab, [0.33, 0.01, 0.43], sub, pi)["partial_integration"]["L"]
    np.savez(out, **save)
    print("WORKER DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
