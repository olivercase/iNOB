#!/usr/bin/env python3
"""How much of the cardiac activation sequence can a chest OPM array recover?

Forward: the propagating myocardial source (:mod:`inob.cfa.cardiac`) in the
BodyParts3D torso FEM. Inverse: a moving equivalent dipole and a depth-weighted
minimum-norm current density (:mod:`inob.cfa.inverse`). Scored against the truth
the forward model was built from, so the number reported is a reconstruction
error, not a residual.

Arrays compared, all triaxial:

* the **measured 8-site plate**, in its real geometry, placed over the precordium;
* **conformal arrays** of 8 / 16 / 32 / 64 sites, same standoff. The 8-site
  conformal array isolates what the real plate loses to being flat; the rest
  answer how much site count buys.

    python scripts/run_chest_inverse.py
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inob.cfa.cardiac import PropagatingCardiacSource
from inob.cfa.chest import ChestArray, conformal_array, load_real_chest_array
from inob.cfa.design import CONDUCTIVITY_SM
from inob.cfa.fem import FemSpec, build_conductor_fem
from inob.cfa.forward import ConductorSolver
from inob.cfa.inverse import (
    fit_moving_dipole, localisation_error, minimum_norm, point_spread,
    resolution_matrix,
)
from inob.cfa.sources import MyocardialSource, load_cdm
from inob.geometry.composites import default_manifest_path, load_composite

log = logging.getLogger("chest_inverse")
CDM = Path("/Users/olivercase/Desktop/CFA/replication pack/data/cache/cdm_correction.mat")


def scan_grid(heart: trimesh.Trimesh, pitch: float = 8.0) -> np.ndarray:
    """Dipole scan grid filling the heart, at ``pitch`` mm."""
    lo, hi = heart.bounds
    ax = [np.arange(lo[i] + pitch / 2, hi[i], pitch) for i in range(3)]
    p = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    return p[heart.contains(p)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--out", type=Path, default=root / "outputs/chest")
    ap.add_argument("--n-myo", type=int, default=400)
    ap.add_argument("--noise-ft", type=float, default=15.0 * np.sqrt(100.0),
                    help="per-channel per-beat noise, fT (15 fT/sqrtHz over 100 Hz)")
    ap.add_argument("--beats", type=int, default=100)
    a = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    a.out.mkdir(parents=True, exist_ok=True)

    d = root / "outputs/cfa_thorax"
    meshes = {q.stem: trimesh.load(q, process=True) for q in sorted(d.glob("*.stl"))}
    T = np.loadtxt(d / "bp3d_to_mni.txt", delimiter=",")
    mr, mp = root / "internal_meshes", default_manifest_path(root)
    myo_mesh = load_composite("myocardium", mesh_root=mr, manifest_path=mp)
    myo_mesh.apply_transform(T)
    heart, skin = meshes["heart"], meshes["skin"]

    src = MyocardialSource.from_wall(myo_mesh, n_sites=a.n_myo)
    cdm = load_cdm(CDM)
    prop = PropagatingCardiacSource.build(src, cdm["time"], cdm["q"])
    grid = scan_grid(heart)
    log.info("dipole scan grid: %d points inside the heart", len(grid))

    target = np.asarray(heart.centroid) + np.array([0.0, 120.0, 0.0])
    arrays: dict[str, ChestArray] = {
        "real 8 (flat plate)": load_real_chest_array(
            a.out / "arrays_frames.mat").placed_on(skin, target, standoff_mm=6.5),
    }
    for n in (8, 16, 32, 64):
        arrays[f"conformal {n}"] = conformal_array(skin, target, n_site=n)

    fem = build_conductor_fem(meshes, FemSpec(pitch_mm=3.0, radbound_mm=6.0,
                                              maxvol_mm3=120.0))
    truth_cog = np.einsum("nt,nd->td", np.abs(prop.amps), src.pos_mm) / \
        np.maximum(np.abs(prop.amps).sum(0)[:, None], 1e-30)
    rng = np.random.default_rng(0)
    rows = []
    for name, arr in arrays.items():
        sol = ConductorSolver.for_sites(fem, CONDUCTIVITY_SM, arr.site_pos_mm)
        b_myo = sol.site_field(src.pos_mm)
        b_grid = sol.site_field(grid)
        n_site = arr.n_site
        # contract each myocardial site's three moment columns with its own
        # dipole orientation, then read the per-site field vector out along the
        # array's three real sensitive axes
        per = np.einsum("scnk,nk->scn",
                        b_myo.reshape(n_site, 3, -1, 3), prop.ori)
        ori = arr.chanori
        L_myo = np.concatenate([np.einsum("scn,sc->sn", per, ori[k * n_site:(k + 1) * n_site])
                                for k in range(3)], 0)             # (3S, n_myo)
        L_grid = np.concatenate(
            [np.einsum("sck,sc->sk", b_grid.reshape(n_site, 3, -1),
                       ori[k * n_site:(k + 1) * n_site]) for k in range(3)], 0)
        y = L_myo @ prop.amps * 1e15                               # fT
        y_noisy = y + (a.noise_ft / np.sqrt(a.beats)) * rng.standard_normal(y.shape)

        amp = np.linalg.norm(y_noisy, axis=0)
        thr = 0.25 * amp.max()
        fit = fit_moving_dipole(y_noisy, grid, L_grid, min_amplitude=thr)
        ok = np.isfinite(fit.pos_mm[:, 0])
        loc = localisation_error(fit.pos_mm[ok], truth_cog[ok])

        j = minimum_norm(y_noisy, L_myo)
        w = np.abs(j)
        cog = np.einsum("nt,nd->td", w, src.pos_mm) / np.maximum(w.sum(0)[:, None], 1e-30)
        mn_loc = localisation_error(cog[ok], truth_cog[ok])
        corr = float(np.corrcoef(np.abs(j).ravel(), np.abs(prop.amps).ravel())[0, 1])
        ps = point_spread(resolution_matrix(L_myo), src.pos_mm)

        rows.append({
            "array": name, "n_site": n_site, "n_chan": 3 * n_site,
            "peak_field_pT": float(np.abs(y).max() / 1e3),
            "dipole_loc_err_mm": float(np.median(loc)),
            "dipole_gof": float(np.nanmedian(fit.gof[ok])),
            "mne_cog_err_mm": float(np.median(mn_loc)),
            "mne_corr": corr,
            "psf_spread_mm": ps["mean_spread_mm"],
            "psf_bias_mm": ps["mean_bias_mm"],
        })
        log.info("%-22s %3d sites | peak %5.1f pT | dipole %5.1f mm | "
                 "MNE cog %5.1f mm | PSF %5.1f mm", name, n_site,
                 rows[-1]["peak_field_pT"], rows[-1]["dipole_loc_err_mm"],
                 rows[-1]["mne_cog_err_mm"], ps["mean_spread_mm"])
        np.savez_compressed(a.out / f"inv_{name.replace(' ', '_')}.npz",
                            pos=arr.site_pos_mm, dip_pos=fit.pos_mm,
                            dip_gof=fit.gof, mne=j, cog=cog, truth_cog=truth_cog,
                            field=y)
        del sol, b_myo, b_grid, L_myo, L_grid
        gc.collect()

    (a.out / "inverse_summary.json").write_text(json.dumps(rows, indent=2))
    print("\n%-22s %5s %10s %12s %12s %10s" % ("array", "sites", "peak pT",
                                               "dipole mm", "MNE cog mm", "PSF mm"))
    for r in rows:
        print("%-22s %5d %10.1f %12.1f %12.1f %10.1f" % (
            r["array"], r["n_site"], r["peak_field_pT"], r["dipole_loc_err_mm"],
            r["mne_cog_err_mm"], r["psf_spread_mm"]))
    print(f"\nwrote {a.out / 'inverse_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
