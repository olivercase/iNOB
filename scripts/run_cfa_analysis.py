#!/usr/bin/env python3
"""Every post-processing step the paper needs, from one cached simulation.

    python scripts/run_cfa_analysis.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inob.cfa.analysis import (
    anova_over_arms, axis_split, bootstrap_compare, detection_table,
)
from inob.cfa.array import CFAArray
from inob.cfa.design import DESIGN
from inob.cfa.engine import Forward, run_2x2

log = logging.getLogger("cfa_analysis")

#: Subsets of the eight solved yaws, chosen so ROTATION SPAN and ORIENTATION
#: COUNT can be varied independently. The replication pack's sweep found that
#: span, not count, is what does the work; this repeats that test on the FEM
#: without solving any new conductors.
YAWS = [-100.0, -70.0, -45.0, -15.0, 15.0, 45.0, 70.0, 100.0]
SWEEP = {
    "span +/-15, n=2":  [-15.0, 15.0],
    "span +/-45, n=4":  [-45.0, -15.0, 15.0, 45.0],
    "span +/-100, n=8": YAWS,
    "span +/-100, n=2": [-100.0, 100.0],
    "span +/-100, n=4": [-100.0, -45.0, 45.0, 100.0],
}


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    root = Path(__file__).resolve().parents[1]
    d = root / "outputs/cfa_forward"
    res = pickle.load(open(d / "sim_2x2.pkl", "rb"))
    arr = CFAArray.from_reference(root / "outputs/_cfa_ref.npz")
    fwd = Forward.load(d, arr)
    out: dict = {"signal_ft": res.sig_rms, "cfa_ft": res.cfa_rms,
                 "source_model_mismatch_pct": res.mismatch_pct,
                 "n_channels": arr.n_chan, "n_sites": arr.n_site}

    # --- axis families ------------------------------------------------------
    cfa = (arr.project(fwd.b_myo[0], arr.chanori) @ fwd.amps)[:, fwd.erp_index] * 1e15
    h, c = axis_split(res.truth, arr.n_site), axis_split(cfa, arr.n_site)
    out["axis_split"] = {k: {"hep_ft": h[k], "cfa_ft": c[k], "ratio": c[k] / h[k]}
                         for k in h}

    # --- detection ----------------------------------------------------------
    rows = detection_table(res)
    out["detection"] = [{"cell": r.cell, "arm": r.arm, "bias_ft": r.bias_ft,
                         "sigma_ft": r.sigma_ft,
                         "beats": None if not r.gate_open else r.beats,
                         "gate_open": r.gate_open} for r in rows]
    print(f"\n=== detection (SNR {DESIGN.snr_target:.0f}, signal {res.sig_rms:.1f} fT) ===")
    print("%-12s %-11s %9s %9s %14s" % ("cell", "arm", "bias fT", "sigma fT", "beats"))
    for r in rows:
        print("%-12s %-11s %9.2f %9.3f %14s" % (
            r.cell, r.arm, r.bias_ft, r.sigma_ft,
            "IMPOSSIBLE" if not r.gate_open else f"{r.beats:,.0f}"))

    # --- ANOVA --------------------------------------------------------------
    an = anova_over_arms(res)
    out["anova"] = {k: {"eta2_precision": v.eta2_a, "eta2_movement": v.eta2_b,
                        "eta2_interaction": v.eta2_ab,
                        "cellmean": v.cellmean.tolist()} for k, v in an.items()}

    # --- bootstrap ----------------------------------------------------------
    out["bootstrap"] = {}
    for a, b in (("glm", "hfc5_lcmv"), ("glm", "hfc1_lcmv"), ("hfc5_lcmv", "hfc1_lcmv")):
        out["bootstrap"][f"{a}_vs_{b}"] = bootstrap_compare(res, a, b)
    print("\n=== paired bootstrap, bias difference (95% CI) ===")
    for k, v in out["bootstrap"].items():
        for cell, s in v.items():
            print("%-24s %-12s %8.2f - %8.2f = %+8.2f fT  CI [%+.2f, %+.2f]" % (
                k, cell, s["bias_a"], s["bias_b"], s["diff"], *s["ci95"]))

    # --- rotation sweep: span vs count --------------------------------------
    print("\n=== rotation sweep: does SPAN or COUNT do the work? ===")
    print("%-20s %3s %10s %10s" % ("orientation set", "n", "GLM bias", "HFC5 bias"))
    sweep = {}
    for name, yaws in SWEEP.items():
        ang = np.array([a for a in DESIGN.ang_high if a[0] in yaws])
        dz = replace(DESIGN, ang_high=ang, n_rep=8)
        r = run_2x2(fwd, dz)
        gi, hi = r.arms.index("glm"), r.arms.index("hfc5_lcmv")
        cur = [i for i, (p, m) in enumerate(dz.cells) if m == "high"]
        g = float(np.mean([r.stats(i, gi).bias for i in cur]))
        hh = float(np.mean([r.stats(i, hi).bias for i in cur]))
        sweep[name] = {"n": len(ang), "span_deg": float(np.ptp(ang[:, 0])),
                       "glm_bias_ft": g, "hfc5_bias_ft": hh}
        print("%-20s %3d %10.2f %10.2f" % (name, len(ang), g, hh))
    out["rotation_sweep"] = sweep

    (d / "analysis.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {d / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
