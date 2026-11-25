#!/usr/bin/env python3
"""Cheap review-response simulations for the CFA/HEF orthogonalisation paper.

    cfa_review_sims.py --sim 1|2|3|4|5|all [--nrep 16] [--procs 4] [--outdir DIR] [--smoke]
    cfa_review_sims.py --regression          (reproduce the paper's 2x2 bias, then exit)

1 null (HEF = 0)            2 QRS-window SSP (+ R-peak jitter, propagating source)
3 nulling LCMV              4 source-strength sweep
5 nested T-wave mismatch (REVIEW B5)

Outputs (DIR/simN.json, DIR/simN.txt).  Worker processes: --procs (<= 4 in the queue).
Seeds are per (cell, draw), so results do not depend on --procs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfa_review_core import (D, FWD, REC_SCALE, ROOT, Cell, aggregate, beat_mean_coeff,  # noqa: E402
                             build_propagating, detect_beats, load_setup, qrs_t_split,
                             rms, run_draw)

OUT = ROOT / "outputs/cfa_review"
ST = None
CELLS = {}


def ensure_init():
    if ST is None:
        _init()


def _init():
    global ST, CELLS
    ST = load_setup("head")
    build_propagating(ST)
    CELLS = {c: Cell(ST, c) for c in range(D.n_cells)}


def _rngs(c, rep, tag=0):
    ss = np.random.SeedSequence([D.rng_seed, c, rep, tag])
    a, b, d = ss.spawn(3)
    return np.random.default_rng(a), np.random.default_rng(b), np.random.default_rng(d)


def _coeffs(cell, rng, cfg):
    """Per-orientation mean cardiac moment for this draw (None = identical every beat)."""
    t_cv, q_cv = cfg.get("t_cv", 0.0), cfg.get("q_cv", 0.0)
    tj, rj = cfg.get("t_jit", 0.0), cfg.get("r_jit", 0.0)
    if t_cv == q_cv == tj == rj == 0:
        return None
    nbo = max(int(cfg.get("n_beats", D.n_beats) // cell.n_or), 1)
    parts = qrs_t_split(ST)
    return [beat_mean_coeff(ST, nbo, rng, t_cv=t_cv, q_cv=q_cv, t_jit_ms=tj,
                            r_jit_ms=rj, parts=parts) for _ in range(cell.n_or)]


# ---------------------------------------------------------------------------
# generic grid task: one (config, cell, draw)
# ---------------------------------------------------------------------------
def grid_task(a):
    cfg, c, rep = a
    cell = CELLS[c]
    r_cal, r_noise, r_beat = _rngs(c, rep, cfg.get("tag", 0))
    coeffs = _coeffs(cell, r_beat, cfg)
    out = run_draw(ST, cell, r_cal, r_noise, amps_nam=cfg["amps"], methods=cfg["methods"],
                   coeffs=coeffs, cardiac=cfg.get("cardiac", "static"))
    return cfg["name"], c, rep, out


def run_grid(cfgs, nrep, procs, cells=range(4)):
    tasks = [(cfg, c, rep) for cfg in cfgs for c in cells for rep in range(nrep)]
    res = pmap(grid_task, tasks, procs)
    bucket = {}
    for name, c, rep, out in res:
        bucket.setdefault((name, c), []).append(out)
    summary = {}
    for (name, c), recs in bucket.items():
        for (m, a), v in aggregate(recs).items():
            summary.setdefault(name, {}).setdefault(D.cell_labels[c], {})[f"{m}@{a:g}"] = v
    return summary


def pmap(fn, tasks, procs):
    if procs <= 1:
        ensure_init()
        return [fn(t) for t in tasks]
    with mp.get_context("spawn").Pool(procs, initializer=_init) as p:
        return p.map(fn, tasks, chunksize=1)


def fnum(x, nd=2):
    return "never" if x is None else (f"{x:.{nd}f}" if x < 1000 else f"{x:.0f}")


# ---------------------------------------------------------------------------
# 1  null
# ---------------------------------------------------------------------------
def sim1(nrep, procs):
    methods = ["none", "orth", "hfc1", "hfc5", "hfc1_rc", "cdm", "ssp_k2", "ssp_k3",
               "null_white", "null_sample", "null_sample_rc"]
    cfg = dict(name="null", amps=[0.0, 10.0], methods=methods)
    s = run_grid([cfg], nrep, procs)["null"]
    # linearity check of orthogonalisation: est(HEF)-est(null) == est(HEF only)
    ensure_init()
    chk = {}
    from cfa_review_core import cardiac_fields, hep_wave, unit_hef
    from inob.cfa.perturb import perturb_array
    wave = hep_wave(ST.time, float(ST.time[ST.k]))
    for c in range(4):
        cell = CELLS[c]
        r1, _, _ = _rngs(c, 0)
        to, g = perturb_array(ST.nom, cell.gain_sd, cell.tilt_sd, r1)
        cc = cardiac_fields(ST, cell.idx, to, g)
        hef = (10.0 * unit_hef(ST, to, g)[:, None] * wave[None])[None] * np.ones((cell.n_or, 1, 1))
        v_null, v_full, v_hef = (cell.orth(x, x)[0] for x in (cc, cc + hef, hef))
        chk[D.cell_labels[c]] = dict(
            null_leak_fT=rms(v_null), full_minus_null_minus_hefonly_fT=rms(v_full - v_null - v_hef),
            hef_only_recovery_err_fT=rms(v_hef - 10.0 * (ST.l_ins0 @ (np.ones(3) / np.sqrt(3))) * 1e6 * wave[ST.k]))
    text = ["SIM 1  NULL (HEF = 0) -- array-RMS of the estimated 'HEF' at 250 ms (bias + noise)",
            f"draws {nrep}; sensor noise = 94.9 fT/beat/channel at 500 beats; recorded noise x{REC_SCALE:.2f}",
            "leak = coherent bias with HEF = 0; limit s/3 = 34 fT (10 nAm); noise SD at 500 beats",
            ""]
    for cl in D.cell_labels:
        text.append(f"[{cl}]")
        text.append(f"{'method':15s} {'leak':>8s} {'est@sens':>9s} {'est@rec':>9s} {'sd_sens':>8s} "
                    f"{'sd_rec':>8s} {'leak/lim':>9s} {'leak/sd_s':>9s} {'leak/sd_r':>9s} {'bias@10':>8s}")
        for m in methods:
            n = s[cl][f"{m}@0"]
            h = s[cl][f"{m}@10"]
            text.append(f"{m:15s} {n['bias']:8.3f} {n['est_rms']:9.3f} {n['est_rms_rec']:9.3f} "
                        f"{n['sigma']:8.3f} {n['sigma_rec']:8.3f} {n['bias']/h['limit']:9.4f} "
                        f"{n['bias']/max(n['sigma'],1e-12):9.2f} {n['bias']/max(n['sigma_rec'],1e-12):9.2f} "
                        f"{h['bias']:8.3f}")
    return dict(results=s, check=chk), "\n".join(text)


# ---------------------------------------------------------------------------
# 2  QRS-window SSP
# ---------------------------------------------------------------------------
def sim2(nrep, procs):
    ks = [f"ssp_k{k}" for k in range(1, 7)]
    base = ["none", "orth"] + ks
    cfgs = [dict(name=f"static_jit{j}", amps=[0.0, 10.0], methods=base, r_jit=float(j),
                 n_beats=D.n_beats, tag=2) for j in (0, 4, 10)]
    cfgs.append(dict(name="propagating_jit0", amps=[0.0, 10.0], methods=base,
                     cardiac="propagating", tag=2))
    s = run_grid(cfgs, nrep, procs)
    ensure_init()
    info = dict(rank_static_amps=int(np.linalg.matrix_rank(ST.amps)),
                rank_propagating_amps=int(ST.prop.spatial_rank()),
                rank_propagating_calibrated=int(np.linalg.matrix_rank(ST.prop_amps)))
    # CFA magnitude at the HEF latency, still head, nominal array (fT RMS)
    nom = ST.nom
    f_s = ST.arr.project(ST.b_myo[0], nom) @ ST.amps[:, ST.k] * 1e15
    f_p = ST.arr.project(ST.prop_b[0], nom) @ ST.prop_amps[:, ST.k] * 1e15
    info.update(cfa_rms_static_250ms=rms(f_s), cfa_rms_propagating_250ms=rms(f_p))
    text = ["SIM 2  QRS-window SSP (window -40..+60 ms, per orientation, k components)",
            f"rank of the static cardiac source (amps): {info['rank_static_amps']} -> the CFA spans <=3 spatial "
            "patterns at ANY latency, so SSP k>=3 removes it almost by construction. "
            f"Propagating source rank {info['rank_propagating_amps']} (supplementary check, T/QRS fixed at 0.25).",
            f"CFA at 250 ms: static {info['cfa_rms_static_250ms']:.0f} fT, propagating "
            f"{info['cfa_rms_propagating_250ms']:.0f} fT RMS",
            ""]
    text.append("bias = noise-free estimate (subspace from noise-free window; for k > 3 the extra singular "
                "vectors are numerically the HEF tail, so bias for k>3 is degenerate); tot = total error "
                "incl. noise, subspace from the NOISY block average (the realistic SSP); best k picked by tot (oracle).")
    for cfg in cfgs:
        text.append(f"== {cfg['name']} (R-peak jitter SD {cfg.get('r_jit', 0):g} ms; HEF 10 nAm) ==")
        text.append(f"{'cell':11s} {'metric':8s} " + " ".join(f"{m:>8s}" for m in base) + "   best-k by tot_s / tot_r")
        for cl in D.cell_labels:
            row = s[cfg["name"]][cl]
            g = lambda f, m: row[f"{m}@10"][f]
            tots = [g("total_err", m) for m in ks]
            totr = [g("total_err_rec", m) for m in ks]
            for lab, f in (("bias", "bias"), ("tot_s", "total_err"), ("tot_r", "total_err_rec")):
                extra = (f"k={int(np.argmin(tots)) + 1} ({min(tots):.2f} fT) / k={int(np.argmin(totr)) + 1} "
                         f"({min(totr):.2f} fT)") if lab == "tot_s" else ""
                text.append(f"{cl if lab == 'bias' else '':11s} {lab:8s} " +
                            " ".join(f"{g(f, m):8.2f}" for m in base) + "   " + extra)
    return dict(results=s, info=info), "\n".join(text)


# ---------------------------------------------------------------------------
# 3  nulling LCMV
# ---------------------------------------------------------------------------
def sim3(nrep, procs):
    methods = ["orth", "null_white", "null_sample", "null_sample_rc", "hfc1"]
    s = run_grid([dict(name="nulling", amps=[10.0], methods=methods)], nrep, procs)["nulling"]
    ensure_init()
    chk = {}
    for c in range(4):
        cell = CELLS[c]
        w0 = cell.null_weights(None)
        chk[D.cell_labels[c]] = dict(
            max_abs_w_diff_rel=float(np.abs(w0 - cell.w_orth).max() / np.abs(cell.w_orth).max()))
        r1, r2, _ = _rngs(c, 0)
        o = run_draw(ST, cell, r1, r2, amps_nam=[10.0], methods=["orth", "null_white"],
                     keep_vec=True, with_noise=False)
        chk[D.cell_labels[c]]["max_abs_est_diff_fT"] = float(
            np.abs(o[("orth", 10.0)]["vec"] - o[("null_white", 10.0)]["vec"]).max())
    text = ["SIM 3  NULLING LCMV (stacked [Phi L] constraints, unit gain on Phi, null on L; 10 nAm)",
            "identity covariance == orthogonalisation (check below); sample covariance = stacked data "
            "covariance + single-trial noise, regularised 5%",
            f"{'cell':11s} " + " ".join(f"{m:>15s}" for m in methods) + "   (bias, fT)"]
    for cl in D.cell_labels:
        text.append(f"{cl:11s} " + " ".join(f"{s[cl][f'{m}@10']['bias']:15.3f}" for m in methods))
    text.append("")
    for cl, v in chk.items():
        text.append(f"identity check {cl}: max|W_null-W_orth|/max|W_orth| = {v['max_abs_w_diff_rel']:.1e}; "
                    f"max|est diff| = {v['max_abs_est_diff_fT']:.1e} fT")
    return dict(results=s, check=chk), "\n".join(text)


# ---------------------------------------------------------------------------
# 4  source-strength sweep
# ---------------------------------------------------------------------------
B1_TABLE = {   # REVIEW_2026-11-25.md section B1, recorded-noise beats (orth), analytic
    10.0: dict(zip(D.cell_labels, (17, 22, 16, 24))),
    2.0: dict(zip(D.cell_labels, (425, 550, 462, 744))),
    1.0: dict(zip(D.cell_labels, (1700, 2200, 3600, 11900))),
    0.5: dict(zip(D.cell_labels, (6800, 8800, None, None))),
}


def sim4(nrep, procs):
    amps = [0.5, 1.0, 2.0, 5.0, 10.0]
    methods = ["orth", "hfc1", "hfc5", "hfc1_rc", "cdm", "null_sample", "ssp_k3"]
    s = run_grid([dict(name="sweep", amps=amps, methods=methods)], nrep, procs)["sweep"]
    # B1-style analytic extrapolation: bias and sigma frozen at their 10 nAm values
    b1 = {}
    for cl in D.cell_labels:
        for m in methods:
            ref = s[cl][f"{m}@10"]
            for a in amps:
                sa = ref["s"] * a / 10.0
                b1[f"{cl}|{m}@{a:g}"] = dict(
                    beats=detect_beats(sa, ref["bias"], ref["sigma"]),
                    beats_rec=detect_beats(sa, ref["bias"], ref["sigma_rec"]))
    text = ["SIM 4  SOURCE-STRENGTH SWEEP; heartbeats to SNR 3 ('never' if bias >= s/3)",
            "columns: bias fT | beats at sensor noise | beats at recorded noise | [B1-analytic recorded]", ""]
    for m in methods:
        text.append(f"-- {m} --")
        text.append(f"{'nAm':>5s} " + " ".join(f"{cl:>34s}" for cl in D.cell_labels))
        for a in amps[::-1]:
            cells_ = []
            for cl in D.cell_labels:
                v = s[cl][f"{m}@{a:g}"]
                bb = b1[f"{cl}|{m}@{a:g}"]["beats_rec"]
                cells_.append(f"{v['bias']:6.2f}|{fnum(v['beats'],1):>6s}|{fnum(v['beats_rec'],0):>6s}|[{fnum(bb,0):>6s}]")
            text.append(f"{a:5g} " + " ".join(f"{c_:>34s}" for c_ in cells_))
    text.append("")
    text.append("B1 table of REVIEW_2026-11-25.md (orth, recorded noise) for comparison:")
    for a, row in B1_TABLE.items():
        text.append(f"{a:5g} nAm: " + ", ".join(f"{k}={fnum(v,0)}" for k, v in row.items()))
    return dict(results=s, b1_analytic=b1, b1_review_table={str(k): v for k, v in B1_TABLE.items()}), "\n".join(text)


# ---------------------------------------------------------------------------
# 5  nested mismatch
# ---------------------------------------------------------------------------
NESTED = dict(t_cv=0.08, q_cv=0.02, t_jit=4.0)
N_LIST = (125, 500, 2000, 8000)
N_REAL = {125: 4, 500: 4, 2000: 3, 8000: 2}
N_INF = 40000
N_SETS = 12


def nested_task(a):
    methods, c, rep = a
    cell = CELLS[c]
    r_cal, r_noise, r_beat = _rngs(c, rep, 5)
    from inob.cfa.perturb import perturb_array
    calib = perturb_array(ST.nom, cell.gain_sd, cell.tilt_sd, r_cal)
    parts = qrs_t_split(ST)

    def errs(coeffs):
        o = run_draw(ST, cell, None, None, amps_nam=[10.0], methods=methods, coeffs=coeffs,
                     with_noise=False, calib=calib, keep_vec=True)
        return {m: o[(m, 10.0)]["vec"] for m in methods}

    res = {"base": errs(None)}
    for n in N_LIST:
        res[f"N{n}"] = []
        for _ in range(N_REAL[n]):
            cf = [beat_mean_coeff(ST, max(n // cell.n_or, 1), r_beat, parts=parts, **_nest_kw())
                  for _ in range(cell.n_or)]
            res[f"N{n}"].append(errs(cf))
    inf = beat_mean_coeff(ST, N_INF, r_beat, parts=parts, **_nest_kw())
    res["Ninf"] = errs([inf] * cell.n_or)
    # one beat per orientation, many sets: is the error the same on every beat?
    res["sets"] = [errs([beat_mean_coeff(ST, 1, r_beat, parts=parts, **_nest_kw())
                         for _ in range(cell.n_or)]) for _ in range(N_SETS)]
    return c, rep, res


def _nest_kw():
    return dict(t_cv=NESTED["t_cv"], q_cv=NESTED["q_cv"], t_jit_ms=NESTED["t_jit"])


def sim5(nrep, procs):
    methods = ["none", "orth", "cdm", "hfc1"]
    tasks = [(methods, c, rep) for c in range(4) for rep in range(nrep)]
    raw = pmap(nested_task, tasks, procs)
    out = {}
    txt = ["SIM 5  NESTED MISMATCH (REVIEW B5): T-gain CV 0.08, QRS CV 0.02, T-peak jitter SD 4 ms,",
           "beats split over the orientations and AVERAGED PER ORIENTATION before correction; HEF 10 nAm, "
           "independent of the T-wave. bias = array-RMS error of the noise-free estimate at 250 ms.",
           "N = beats in the recording; 'inf' = expected waveform (40000 beats); 1/ori = a set of one beat "
           "per orientation.",
           ""]
    for c in range(4):
        cl = D.cell_labels[c]
        rows = [r for (cc, rep, r) in raw if cc == c]
        out[cl] = {}
        for m in methods:
            ent = {"no_variability": rms(np.stack([r["base"][m] for r in rows]))}
            for n in N_LIST:
                ent[f"N{n}"] = rms(np.stack([v[m] for r in rows for v in r[f"N{n}"]]))
            ent["Ninf"] = rms(np.stack([r["Ninf"][m] for r in rows]))
            # one-beat sets: coherent (mean over sets) vs fluctuating part
            coh, fl = [], []
            for r in rows:
                e = np.stack([s_[m] for s_ in r["sets"]])
                mu = e.mean(0)
                coh.append(rms(mu))
                fl.append(rms(e - mu))
            ent["single_beat_set_total"] = rms(np.stack([s_[m] for r in rows for s_ in r["sets"]]))
            ent["single_beat_set_coherent"] = float(np.sqrt(np.mean(np.square(coh))))
            ent["single_beat_set_fluct"] = float(np.sqrt(np.mean(np.square(fl))))
            out[cl][m] = ent
        txt.append(f"[{cl}]  bias (fT)")
        txt.append(f"{'method':6s} {'novar':>8s} " + " ".join(f"{'N'+str(n):>8s}" for n in N_LIST) +
                   f" {'inf':>8s} {'1/ori tot':>9s} {'coh':>8s} {'fluct':>8s}")
        for m in methods:
            e = out[cl][m]
            txt.append(f"{m:6s} {e['no_variability']:8.3f} " + " ".join(f"{e[f'N{n}']:8.3f}" for n in N_LIST) +
                       f" {e['Ninf']:8.3f} {e['single_beat_set_total']:9.3f} {e['single_beat_set_coherent']:8.3f} "
                       f"{e['single_beat_set_fluct']:8.3f}")
    return dict(results=out, nested=NESTED, n_list=list(N_LIST), n_real=N_REAL, n_sets=N_SETS), "\n".join(txt)


SIMS = {1: sim1, 2: sim2, 3: sim3, 4: sim4, 5: sim5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="all")
    ap.add_argument("--nrep", type=int, default=D.n_rep)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--smoke", action="store_true", help="tiny run: 2 draws, 1 process, label smoke")
    ap.add_argument("--regression", action="store_true")
    a = ap.parse_args()
    if a.regression:
        from cfa_review_regress import regression
        for (lab, n), v in regression("head").items():
            print(f"{lab:11s} {n:5s} ours {v['ours']:.4f} paper {v['paper']:.4f} ratio {v['ours']/v['paper']:.4f}")
        return 0
    if a.procs > 4:
        raise SystemExit("at most 4 worker processes")
    outdir = Path(a.outdir)
    nrep, procs = a.nrep, a.procs
    if a.smoke:
        nrep, procs = 2, 1
        outdir = OUT / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    which = list(SIMS) if a.sim == "all" else [int(x) for x in a.sim.split(",")]
    rc = 0
    for i in which:
        t0 = time.time()
        res, text = SIMS[i](nrep, procs)
        meta = dict(sim=i, nrep=nrep, procs=procs, smoke=bool(a.smoke), seconds=time.time() - t0,
                    rec_scale=REC_SCALE, n_beats=D.n_beats)
        (outdir / f"sim{i}.json").write_text(json.dumps(dict(meta=meta, **res), indent=1, default=float))
        (outdir / f"sim{i}.txt").write_text(("[SMOKE TEST - TINY RUN, NOT FINAL]\n" if a.smoke else "") + text + "\n")
        print(f"sim {i} done in {meta['seconds']:.0f} s -> {outdir}/sim{i}.json", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
