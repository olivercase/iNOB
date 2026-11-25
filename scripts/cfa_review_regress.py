"""Regression test: reproduce the manuscript's 2x2 coherent bias from the stored lead
fields with our own generator, draw by draw, against outputs/cfa_forward/sim_2x2.pkl."""
from __future__ import annotations
import json, pickle, sys, time
import numpy as np
from cfa_review_core import *          # noqa

def regression(frame="head", nrep=None, verbose=True):
    st = load_setup(frame)
    res = pickle.load(open(FWD / "sim_2x2.pkl", "rb"))
    paper = json.loads((FWD / "paper_numbers.json").read_text())["bias_ft"]
    nrep = nrep or D.n_rep
    k = st.k
    names = ["none", "orth", "hfc1", "hfc5", "cdm"]
    arm_of = {"none": "none", "orth": "glm", "hfc1": "hfc1_lcmv", "hfc5": "hfc5_lcmv", "cdm": "cdm"}
    wave = hep_wave(st.time, float(st.time[k]))
    out = {}
    for c in range(D.n_cells):
        cell = Cell(st, c)
        rng = np.random.default_rng(D.rng_seed + c)        # engine's stream
        tab = method_table(cell, names)
        per = {n: [] for n in names}
        maxdiff = {n: 0.0 for n in names}
        for rep in range(nrep):
            true_ori, gain = perturb_array(st.nom, cell.gain_sd, cell.tilt_sd, rng)
            ccfa = cardiac_fields(st, cell.idx, true_ori, gain)
            hef = 10.0 * unit_hef(st, true_ori, gain)[:, None] * wave[None]
            yc = ccfa + hef[None]
            _ = rng.standard_normal(yc.shape)             # keep the stream aligned
            truth = 10.0 * (st.l_ins0 @ (np.ones(3) / np.sqrt(3))) * 1e-9 * 1e15 * wave[k]
            ai = res.arms.index
            for n in names:
                rb, _rn = tab[n][0](yc, yc)
                per[n].append(rb)
                ref = res.rec_bias[(c, ai(arm_of[n]))][:, rep]
                maxdiff[n] = max(maxdiff[n], float(np.abs(rb - ref).max() / max(np.abs(ref).max(), 1e-12)))
        label = D.cell_labels[c]
        for n in names:
            bias = rms(np.array(per[n]).T - truth[:, None])
            out[(label, n)] = dict(ours=bias, paper=paper[label][arm_of[n]], max_rel_draw_diff=maxdiff[n])
    return out

if __name__ == "__main__":
    frame = sys.argv[1] if len(sys.argv) > 1 else "head"
    nrep = int(sys.argv[2]) if len(sys.argv) > 2 else None
    t0 = time.time()
    r = regression(frame, nrep)
    for (lab, n), v in r.items():
        print(f"{lab:11s} {n:5s} ours {v['ours']:9.4f} paper {v['paper']:9.4f} ratio {v['ours']/v['paper']:.4f} maxreldiff/draw {v['max_rel_draw_diff']:.2e}")
    print("seconds", round(time.time() - t0, 1))
