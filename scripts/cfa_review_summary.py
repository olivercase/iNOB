#!/usr/bin/env python3
"""Turn outputs/cfa_review/sim{1..5}.json into outputs/cfa_review/summary.md.

    cfa_review_summary.py [--dir outputs/cfa_review] [--out summary.md]

Files flagged smoke=true are labelled as tiny-run output, never as final.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CELLS = ["SQUID", "future OPM", "worst case", "current OPM"]
PRETTY = {"SQUID": "Cryogenic MEG (high, still)", "future OPM": "Future OPM (high, tracked)",
          "worst case": "Worst case (low, still)", "current OPM": "Current OPM (low, tracked)"}


def f(x, nd=2):
    if x is None:
        return "never"
    return f"{x:.{nd}f}" if abs(x) < 1000 else f"{x:,.0f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def load(d, i):
    p = d / f"sim{i}.json"
    return json.loads(p.read_text()) if p.exists() else None


def s1(j):
    r = j["results"]
    L = ["## Sim 1. Null (HEF = 0)", "",
         "Array-RMS of the estimated 'HEF' at 250 ms with no HEF present. leak = coherent part; "
         "est = leak + noise (sensor noise 15 fT/rtHz, and the recorded between-beat level, x12.34). "
         "Limit = s/3 = 34 fT (10 nAm). sd = noise SD of the estimate at 500 beats.", ""]
    for c in CELLS:
        L += [f"**{PRETTY[c]}**", ""]
        rows = []
        for m in r[c] and sorted({k.split('@')[0] for k in r[c]}, key=lambda x: list(r[c]).index(x + "@0")):
            n, h = r[c][f"{m}@0"], r[c][f"{m}@10"]
            rows.append([m, f(n["bias"], 3), f(n["est_rms"], 3), f(n["est_rms_rec"], 3),
                         f(n["bias"] / h["limit"], 4), f(n["bias"] / max(n["sigma"], 1e-12), 2),
                         f(n["bias"] / max(n["sigma_rec"], 1e-12), 2), f(h["bias"], 3)])
        L += table(["method", "leak (fT)", "est, sensor noise", "est, recorded noise", "leak / (s/3)",
                    "leak / sd (sensor)", "leak / sd (recorded)", "bias with 10 nAm HEF"], rows) + [""]
    L += ["Linearity check of orthogonalisation (est(HEF+CFA) - est(CFA) - est(HEF only), fT):", ""]
    L += table(["condition", "null leak", "residual"], [[c, f(v["null_leak_fT"], 3),
               f"{v['full_minus_null_minus_hefonly_fT']:.1e}"] for c, v in j["check"].items()]) + [""]
    return L


def s2(j):
    r = j["results"]
    inf = j["info"]
    L = ["## Sim 2. QRS-window SSP (empirical, per head orientation)", "",
         f"Subspace from the block-averaged data in -40..+60 ms around the R-peak (SVD, k components); data and "
         f"brain lead field projected off it; joint least squares across orientations. Spatial rank of the "
         f"simulated static cardiac source: **{inf['rank_static_amps']}**, so k >= 3 can remove the CFA at any "
         f"latency by construction (a property of the non-propagating source). Supplementary propagating source "
         f"(repo's PropagatingCardiacSource, T/QRS 0.25): rank {inf['rank_propagating_amps']}; CFA at 250 ms "
         f"{f(inf['cfa_rms_propagating_250ms'], 0)} fT vs static {f(inf['cfa_rms_static_250ms'], 0)} fT. "
         "bias = noise-free estimate; tot_s / tot_r = total error with the realistic (noisy) subspace at "
         "sensor / recorded noise. Best k is chosen on tot_s (oracle).", ""]
    for name, cfgr in r.items():
        L += [f"**{name}**", ""]
        rows = []
        for c in CELLS:
            row = cfgr[c]
            ks = [f"ssp_k{k}" for k in range(1, 7)]
            tots = [row[f"{k}@10"]["total_err"] for k in ks]
            totr = [row[f"{k}@10"]["total_err_rec"] for k in ks]
            kb = min(range(6), key=lambda i: tots[i])
            rows.append([PRETTY[c], f(row["none@10"]["bias"], 1), f(row["orth@10"]["bias"], 2),
                         f(row["orth@10"]["total_err_rec"], 1),
                         f(row["ssp_k2@10"]["total_err"], 2), f(row["ssp_k3@10"]["bias"], 2),
                         f(row["ssp_k3@10"]["total_err"], 2), f(row["ssp_k3@10"]["total_err_rec"], 1),
                         f"k={kb+1}: {f(tots[kb],2)} / {f(totr[kb],1)}",
                         f(row["ssp_k3@0"]["bias"], 3)])
        L += table(["condition", "none bias", "orth bias", "orth tot_r", "SSP k=2 tot_s", "SSP k=3 bias",
                    "SSP k=3 tot_s", "SSP k=3 tot_r", "best k: tot_s / tot_r", "SSP k=3 null leak"], rows) + [""]
    return L


def s3(j):
    r = j["results"]
    ms = ["orth", "null_white", "null_sample", "null_sample_rc", "hfc1"]
    L = ["## Sim 3. Nulling LCMV", "",
         "Stacked constraints [Phi L]: unit gain on the insula, null on the stacked cardiac lead fields, 10 nAm. "
         "null_white = identity covariance; null_sample = stacked sample covariance + single-trial noise, 5% "
         "regularisation; _rc = noise term at the recorded level.", ""]
    L += table(["condition"] + [f"{m} bias (fT)" for m in ms],
               [[PRETTY[c]] + [f(r[c][f"{m}@10"]["bias"], 3) for m in ms] for c in CELLS]) + [""]
    L += ["Equivalence check (identity covariance vs orthogonalisation):", ""]
    L += table(["condition", "max rel. weight difference", "max estimate difference (fT)"],
               [[c, f"{v['max_abs_w_diff_rel']:.1e}", f"{v['max_abs_est_diff_fT']:.1e}"]
                for c, v in j["check"].items()]) + [""]
    return L


def s4(j):
    r = j["results"]
    amps = ["10", "5", "2", "1", "0.5"]
    methods = sorted({k.split("@")[0] for k in r["SQUID"]})
    L = ["## Sim 4. Source-strength sweep", "",
         "Heartbeats to reach SNR = 3 at the sensor-noise level / at the recorded noise level; 'never' = bias >= s/3. "
         "In brackets: the B1-style analytic value (bias and sigma frozen at their 10 nAm values).", ""]
    for m in methods:
        L += [f"**{m}**", ""]
        rows = []
        for a in amps:
            row = [a]
            for c in CELLS:
                v = r[c][f"{m}@{a}"]
                b1 = j["b1_analytic"][f"{c}|{m}@{a}"]
                row.append(f"b={f(v['bias'],2)}; {f(v['beats'],1)} / {f(v['beats_rec'],0)} [{f(b1['beats_rec'],0)}]")
            rows.append(row)
        L += table(["nAm"] + CELLS, rows) + [""]
    L += ["REVIEW_2026-11-25.md B1 table (orthogonalisation, recorded noise):", ""]
    L += table(["nAm"] + CELLS, [[a] + [f(v, 0) if v is not None else "never" for v in row.values()]
                                  for a, row in j["b1_review_table"].items()]) + [""]
    return L


def s5(j):
    r = j["results"]
    ns = j["n_list"]
    L = ["## Sim 5. Nested T-wave mismatch (REVIEW B5)", "",
         f"T-gain CV {j['nested']['t_cv']}, QRS CV {j['nested']['q_cv']}, T-peak jitter SD {j['nested']['t_jit']} ms; "
         "beats split over the orientations and averaged per orientation; HEF fixed (independent of the T-wave). "
         "Noise-free bias (fT). 'inf' = expected waveform. 'one beat/ori' = a set of one beat per orientation: "
         "coherent = mean over sets, fluctuating = beat-to-beat part.", ""]
    for m in ("orth", "none", "cdm", "hfc1"):
        L += [f"**{m}**", ""]
        rows = []
        for c in CELLS:
            e = r[c][m]
            rows.append([PRETTY[c], f(e["no_variability"], 3)] + [f(e[f"N{n}"], 3) for n in ns] +
                        [f(e["Ninf"], 3), f(e["single_beat_set_total"], 3), f(e["single_beat_set_coherent"], 3),
                         f(e["single_beat_set_fluct"], 3)])
        L += table(["condition", "no variability"] + [f"N={n}" for n in ns] +
                   ["N=inf", "one beat/ori, total", "coherent", "fluctuating"], rows) + [""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parents[1] / "outputs/cfa_review"))
    ap.add_argument("--out", default="summary.md")
    a = ap.parse_args()
    d = Path(a.dir)
    L = ["# CFA/HEF review simulations: summary", ""]
    smoke = False
    for i, fn in enumerate((s1, s2, s3, s4, s5), 1):
        j = load(d, i)
        if j is None:
            L += [f"## Sim {i}", "", "not available", ""]
            continue
        m = j["meta"]
        smoke = smoke or m["smoke"]
        L += [f"*sim {i}: {m['nrep']} calibration draws, {m['procs']} process(es), {m['seconds']:.0f} s"
              f"{'; **SMOKE TEST, TINY RUN, NOT FINAL**' if m['smoke'] else ''}*", ""] + fn(j)
    if smoke:
        L.insert(2, "> **Contains tiny smoke-test output (2 draws). Not final.**\n")
    (d / a.out).write_text("\n".join(L) + "\n")
    print("wrote", d / a.out)


if __name__ == "__main__":
    main()
