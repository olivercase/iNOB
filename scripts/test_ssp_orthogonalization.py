#!/usr/bin/env python3
"""Test: does a movement-free orthogonalization (SSP) arm beat/lose to LCMV
as the user's hypothesis predicts, across the precision x movement 2x2?

Reuses the EXACT generative loop from ``engine.run_2x2`` (same rng seeding,
same perturbation, same cells) so the numbers below are directly comparable to
the cached sim_2x2.pkl arms. Not a refactor of engine.py -- a standalone
comparison script, run once as a test.

    python scripts/test_ssp_orthogonalization.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inob.cfa.arms import arm_glm, arm_hfc_lcmv, arm_none, arm_ssp
from inob.cfa.array import CFAArray
from inob.cfa.design import DESIGN
from inob.cfa.engine import Forward, _reproject_ins, hep_waveform
from inob.cfa.hfc import harmonic_projector
from inob.cfa.perturb import perturb_array
from inob.cfa.stats import error_stats

logging.basicConfig(level="INFO", format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ssp_test")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    d = root / "outputs/cfa_forward"
    arr = CFAArray.from_reference(root / "outputs/_cfa_ref.npz")
    fwd = Forward.load(d, arr)
    design = DESIGN

    nom_ori = arr.chanori
    chanpos = arr.chanpos_mm
    t = fwd.time_s
    k = fwd.erp_index
    n_t = len(t)

    m_dir = np.ones(3) / np.sqrt(3.0)
    moment = np.outer(m_dir, hep_waveform(t, float(t[k]))) * 10.0 * 1e-9
    l_ins_model = fwd.l_ins_head[0]
    phi = np.linalg.qr(l_ins_model)[0]
    truth = (l_ins_model @ moment)[:, k] * 1e15

    ssp_ranks = (3, 4, 6)
    arm_names = ("none", "glm", "hfc1_lcmv", *[f"ssp{r}" for r in ssp_ranks])

    print(f"\n{'cell':<12}{'arm':<12}{'bias_fT':>10}{'sigma_fT':>10}{'topo_r':>8}")
    results = {}
    for c, (pname, mname) in enumerate(design.cells):
        gain_sd, tilt_sd = design.precision(pname)
        idx = [fwd.index_of(a) for a in design.angles(mname)]
        n_or = len(idx)
        beats = design.n_beats / n_or
        nb = design.noise_ft / np.sqrt(beats)

        l_mod = [arr.project(fwd.b_eq[n], nom_ori) for n in idx]
        gdes = np.concatenate([np.tile(phi, (n_or, 1)),
                               np.concatenate(l_mod, axis=0)], axis=1)
        proj1 = harmonic_projector(chanpos, nom_ori, 1)

        rec_bias = {a: np.full((arr.n_chan, design.n_rep), np.nan) for a in arm_names}
        rec_noisy = {a: np.full((arr.n_chan, design.n_rep), np.nan) for a in arm_names}

        rng = np.random.default_rng(design.rng_seed + c)
        for rep in range(design.n_rep):
            true_ori, gain = perturb_array(nom_ori, gain_sd, tilt_sd, rng)
            g = gain[:, None]
            yc = np.empty((n_or, arr.n_chan, n_t))
            ccfa = np.empty((n_or, arr.n_chan, n_t))
            for j, n in enumerate(idx):
                lin = fwd.l_ins_head[0]
                ccfa[j] = g * (arr.project(fwd.b_myo[n], true_ori) @ fwd.amps) * 1e15
                hep_meas = g * (_reproject_ins(arr, lin, nom_ori, true_ori) @ moment) * 1e15
                yc[j] = hep_meas + ccfa[j]
            yn = yc + nb * rng.standard_normal(yc.shape)

            # REALISTIC reference: the same recording, re-averaged locked to
            # the R-peak instead of the stimulus, using the SAME per-block beat
            # budget -- so it carries the SAME finite-averaging noise `nb` as
            # the stimulus-locked data, drawn independently (a different lock
            # point sees uncorrelated noise). This is the fair test: no oracle
            # access to the noiseless model, and per-orientation reference
            # quality degrades exactly as movement splits the beat budget into
            # more, shorter blocks.
            ccfa_ref = ccfa + nb * rng.standard_normal(ccfa.shape)

            out = {
                "none": arm_none(yc, yn),
                "glm": arm_glm(yc, yn, gdes, phi),
                "hfc1_lcmv": arm_hfc_lcmv(yc, yn, proj1, phi,
                                          lambda_reg=design.lambda_reg,
                                          noise_var=nb ** 2),
            }
            for r in ssp_ranks:
                out[f"ssp{r}"] = arm_ssp(yc, yn, ccfa_ref, phi, rank=r)
            for a, (rb, rn) in out.items():
                rec_bias[a][:, rep] = rb[:, k]
                rec_noisy[a][:, rep] = rn[:, k]

        cell_label = design.cell_labels[c]
        results[cell_label] = {}
        for a in arm_names:
            st = error_stats(rec_bias[a], rec_noisy[a], truth)
            results[cell_label][a] = st
            print(f"{cell_label:<12}{a:<12}{st.bias:>10.2f}{st.sigma:>10.2f}{st.topo_r:>8.3f}")
        print()

    print(f"signal rms = {float(np.sqrt((truth**2).mean())):.1f} fT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
