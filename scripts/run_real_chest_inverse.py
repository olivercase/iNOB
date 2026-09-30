#!/usr/bin/env python3
"""Invert the REAL measured chest MCG through the iNOB FEM torso.

Pipeline, all four legs measured rather than assumed where possible:

1. plate pose  -- reproduced from the CFA pilot's own torso fiducials, relative
   to the heart (:func:`inob.cfa.chest.placed_from_cfa`);
2. real data   -- R-locked average of the 24 chest channels from the raw .lvm;
3. forward     -- DUNEuro FEM on the BodyParts3D thorax, real sensor axes;
4. inverse     -- moving equivalent dipole + depth-weighted minimum norm on a
   grid filling the heart.

    python scripts/run_real_chest_inverse.py
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import numpy as np, trimesh, scipy.io as sio

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from inob.cfa.chest import ChestArray, load_real_chest_array, placed_from_cfa
from inob.cfa.design import CONDUCTIVITY_SM
from inob.cfa.fem import FemSpec, build_conductor_fem
from inob.cfa.forward import ConductorSolver
from inob.cfa.inverse import fit_moving_dipole, minimum_norm

log = logging.getLogger("real_chest")
SP = "/private/tmp/claude-501/-Users-olivercase-Desktop-CFA/8e26989c-c5b0-4b6c-a490-4ecb2d71b975/scratchpad"
PACK = Path("/Users/olivercase/Desktop/CFA/replication pack")


def area_centroid(v, f):
    v1, v2, v3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    c = (v1 + v2 + v3) / 3.0
    a = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
    return (c * a[:, None]).sum(0) / a.sum()


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = root / "outputs/chest_real"; out.mkdir(parents=True, exist_ok=True)
    d = root / "outputs/cfa_thorax"
    meshes = {q.stem: trimesh.load(q, process=True) for q in sorted(d.glob("*.stl"))}
    heart, skin = meshes["heart"], meshes["skin"]

    # --- 1. the plate, at the CFA pilot's pose --------------------------------
    m = sio.loadmat(PACK / "data/cache/bem_and_grad.mat", squeeze_me=True,
                    struct_as_record=False)
    bm = np.atleast_1d(m["vol_bem_torso"].bmeshes)
    v_t = np.asarray(bm[2].p) * 1000.0
    f_t = np.asarray(bm[2].e).astype(int) - 1
    pack_torso_c = area_centroid(v_t, f_t)
    pack_heart_c = (np.asarray(bm[0].p) * 1000.0).mean(0)
    measured = load_real_chest_array(root / "outputs/chest/arrays_frames.mat")
    arr, slide = placed_from_cfa(measured, pack_heart_c, pack_torso_c, skin,
                                 np.asarray(heart.centroid))
    log.info("plate placed at the CFA pose, slid %.1f mm out to clear the skin", slide)
    pq = trimesh.proximity.ProximityQuery(skin)
    ph = trimesh.proximity.ProximityQuery(heart)
    log.info("  %d/8 sites inside the body | epicardium %.1f-%.1f mm",
             int((pq.signed_distance(arr.site_pos_mm) > 0).sum()),
             *(lambda x: (x.min(), x.max()))(-ph.signed_distance(arr.site_pos_mm)))

    # --- 2. the real R-locked average ----------------------------------------
    ev = np.load(f"{SP}/real_chest_evoked.npz", allow_pickle=True)
    y = ev["avg"] * 1e-12                      # pT -> T, (24, n_t)
    t_ms, labels = ev["t"], list(ev["labels"])
    log.info("real data: %d beats, %d channels, peak %.1f pT",
             int(ev["n_beats"]), y.shape[0], np.abs(ev["avg"]).max())
    # channel order of the array is site-major per axis: X1..X8, Y1..Y8, Z1..Z8,
    # which is exactly ChestArray.chanori's [axis0 sites, axis1 sites, axis2 sites]
    assert labels == [f"{a}{s}" for a in "XYZ" for s in range(1, 9)], labels

    # --- 3. forward -----------------------------------------------------------
    fem = build_conductor_fem(meshes, FemSpec(pitch_mm=3.0, radbound_mm=6.0,
                                              maxvol_mm3=120.0))
    sol = ConductorSolver.for_sites(fem, CONDUCTIVITY_SM, arr.site_pos_mm)
    lo, hi = heart.bounds
    ax = [np.arange(lo[i] + 4.0, hi[i], 8.0) for i in range(3)]
    grid = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    grid = grid[heart.contains(grid)]
    log.info("heart grid: %d points", len(grid))
    b = sol.site_field(grid)                                  # (S, 3, 3*n_grid)
    n_site = arr.n_site
    ori = arr.chanori
    lead = np.concatenate([np.einsum("sck,sc->sk", b.reshape(n_site, 3, -1),
                                     ori[k * n_site:(k + 1) * n_site])
                           for k in range(3)], 0)             # (24, 3*n_grid)
    log.info("leadfield %s", lead.shape)

    # --- 4. inverse -----------------------------------------------------------
    amp = np.linalg.norm(y, axis=0)
    thr = 0.25 * amp.max()
    fit = fit_moving_dipole(y, grid, lead, min_amplitude=thr)
    ok = np.isfinite(fit.pos_mm[:, 0])
    j = minimum_norm(y, lead)
    w = np.abs(j).reshape(len(grid), 3, -1).sum(1)
    cog = np.einsum("nt,nd->td", w, grid) / np.maximum(w.sum(0)[:, None], 1e-30)
    k = int(amp.argmax())
    log.info("R-peak sample t=%.0f ms", t_ms[k])
    log.info("  moving dipole at R-peak: %s mm, GoF %.3f",
             np.round(fit.pos_mm[k], 1), fit.gof[k])
    log.info("  MNE centre of gravity  : %s mm", np.round(cog[k], 1))
    log.info("  heart centroid         : %s mm", np.round(heart.centroid, 1))
    log.info("  dipole -> heart centroid: %.1f mm",
             np.linalg.norm(fit.pos_mm[k] - heart.centroid))
    log.info("  fits above threshold: %d / %d samples", int(ok.sum()), len(ok))
    np.savez(out / "real_inverse.npz", pos=arr.site_pos_mm, chanori=ori,
             grid=grid, dip_pos=fit.pos_mm, dip_gof=fit.gof, mne=j, cog=cog,
             field=ev["avg"], t_ms=t_ms, slide_mm=slide,
             heart_centroid=np.asarray(heart.centroid))
    json.dump({"slide_mm": slide, "n_beats": int(ev["n_beats"]),
               "peak_pT": float(np.abs(ev["avg"]).max()),
               "r_peak_ms": float(t_ms[k]),
               "dipole_gof_at_R": float(fit.gof[k]),
               "dipole_to_heart_centroid_mm":
                   float(np.linalg.norm(fit.pos_mm[k] - heart.centroid)),
               "n_grid": int(len(grid))},
              open(out / "real_inverse.json", "w"), indent=2)
    log.info("saved -> %s", out / "real_inverse.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
