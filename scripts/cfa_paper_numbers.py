#!/usr/bin/env python3
"""Every number and figure array the HEF/CFA manuscript uses, from the iNOB simulation.

Reads the calibrated forward model and the cached 2x2 (outputs/cfa_forward) and
writes
    outputs/cfa_forward/paper_numbers.json   every number quoted in the text
    outputs/cfa_forward/paper_figdata.npz    every array the figures draw

so the manuscript text and figures come from one place. Run after
``python -m inob.cfa.calibrate``, the 2x2 engine and ``run_cfa_analysis.py``.

    python scripts/cfa_paper_numbers.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inob.cfa.arms import arm_hfc_lcmv                          # noqa: E402
from inob.cfa.array import CFAArray                             # noqa: E402
from inob.cfa.calibrate import recorded_rms                     # noqa: E402
from inob.cfa.design import DESIGN as D                         # noqa: E402
from inob.cfa.engine import Forward, _reproject_ins, hep_waveform  # noqa: E402
from inob.cfa.hfc import harmonic_projector                     # noqa: E402
from inob.cfa.perturb import perturb_array                      # noqa: E402

PACK = Path("/Users/olivercase/Desktop/CFA/replication pack")
OUT = ROOT / "outputs/cfa_forward"
R_IDX_MS = 400.0                      # R-peak position in the simulation window


def rms(x, axis=None):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def main() -> int:
    arr = CFAArray.from_reference(ROOT / "outputs/_cfa_ref.npz")
    fwd = Forward.load(OUT, arr)
    res = pickle.load(open(OUT / "sim_2x2.pkl", "rb"))
    ana = json.loads((OUT / "analysis.json").read_text())
    cal = json.loads((OUT / "calibration.json").read_text())
    nom, k, t = arr.chanori, fwd.erp_index, fwd.time_s
    ns = arr.n_site
    t_ms = t * 1000.0 - R_IDX_MS
    i_r = int(np.argmin(np.abs(t_ms)))

    # --- the two fields, still head, nominal array -------------------------
    moment = np.outer(np.ones(3) / np.sqrt(3), hep_waveform(t, float(t[k]))) * 10e-9
    lin = fwd.l_ins_head[0]
    phi = np.linalg.qr(lin)[0]
    hef = (lin @ moment) * 1e15
    i0 = fwd.index_of((0.0, 0.0, 0.0))
    cfa = (arr.project(fwd.b_myo[i0], nom) @ fwd.amps) * 1e15
    best = int(np.argmax(np.abs(hef[:, k])))
    s = float(rms(hef[:, k]))
    gate = s / D.snr_target

    # --- geometry: overlap and principal angle for orientation sets --------
    def geom(idx):
        L = np.concatenate([arr.project(fwd.b_eq[n], nom) for n in idx], 0)
        pb = np.tile(phi, (len(idx), 1))
        q = np.linalg.qr(L)[0]
        r = pb - q @ (q.T @ pb)
        return (float(1 - np.linalg.norm(r) ** 2 / np.linalg.norm(pb) ** 2),
                float(np.degrees(subspace_angles(pb, L)).min()))

    yaw = {int(round(a[0])): fwd.index_of(a) for a in fwd.angles}
    sets = {"still": [0], "+/-15, 2": [-15, 15], "+/-45, 4": [-45, -15, 15, 45],
            "+/-100, 2": [-100, 100], "+/-100, 4": [-100, -45, 45, 100],
            "+/-100, 8": [-100, -70, -45, -15, 15, 45, 70, 100]}
    geometry = {}
    for name, ys in sets.items():
        ov, ang = geom([yaw[y] for y in ys])
        geometry[name] = {"n": len(ys), "overlap_pct": 100 * ov, "min_angle_deg": ang}

    # --- CFA topography across the tracked orientations ---------------------
    tracked = [fwd.index_of(a) for a in D.angles("high")]
    cfa_k = {n: (arr.project(fwd.b_myo[n], nom) @ fwd.amps)[:, k] * 1e15
             for n in range(len(fwd.angles))}
    r_all = {int(round(fwd.angles[n][0])): float(np.corrcoef(cfa_k[n], cfa_k[i0])[0, 1])
             for n in tracked}

    # --- 2x2 results ----------------------------------------------------------
    det = {(d["cell"], d["arm"]): d for d in ana["detection"]}
    cells = list(res.cell_labels)
    arms = list(res.arms)
    bias = {c: {a: det[(c, a)]["bias_ft"] for a in arms if (c, a) in det} for c in cells}
    sigma = {c: {a: det[(c, a)]["sigma_ft"] for a in arms if (c, a) in det} for c in cells}
    beats = {c: {a: det[(c, a)]["beats"] for a in arms if (c, a) in det} for c in cells}

    # beats if the noise were the recorded between-beat level instead of the
    # 15 fT/sqrtHz sensor floor (N scales with noise^2)
    tr, rec, n_rec = recorded_rms(PACK)
    noise_scale = 1171.0 / D.noise_ft          # recorded between-beat RMS (manuscript Fig.)
    def n_needed(b, sg, scale=1.0):
        rhs = gate ** 2 - b ** 2
        return None if rhs <= 0 else float(D.n_beats * (sg * scale) ** 2 / rhs)
    beats_recnoise = {c: {a: (None if beats[c][a] is None else
                              n_needed(bias[c][a], sigma[c][a], noise_scale))
                          for a in bias[c]} for c in cells}

    # --- HFC + LCMV diagnostic (low precision, still head, order 1) --------
    P = harmonic_projector(arr.chanpos_mm, nom, 1)
    nb = D.noise_ft / np.sqrt(D.n_beats)
    def lcmv_bias(gain_sd, tilt_sd, cfa_on=True, sig_free=False, lam=D.lambda_reg,
                  noise_var=D.noise_ft ** 2):
        rng = np.random.default_rng(D.rng_seed + 2)
        b = []
        for _ in range(D.n_rep):
            ori, g = perturb_array(nom, gain_sd, tilt_sd, rng)
            g = g[:, None]
            cf = g * (arr.project(fwd.b_myo[i0], ori) @ fwd.amps) * 1e15 * float(cfa_on)
            hm = g * (_reproject_ins(arr, lin, nom, ori) @ moment) * 1e15
            y = (hm + cf)[None]
            if sig_free:
                z = P @ cf
                cov = np.cov(z) + noise_var * P
                cov = cov + lam * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
                ci = np.linalg.pinv(cov)
                pp = P @ phi
                w = np.linalg.solve(pp.T @ ci @ pp, pp.T @ ci)
                rb = phi @ (w @ (P @ y[0]))
            else:
                rb, _ = arm_hfc_lcmv(y, y, P, phi, lambda_reg=lam, noise_var=noise_var)
            b.append(rms(rb[:, k] - hef[:, k]))
        return float(np.sqrt(np.mean(np.square(b))))
    lo, hi = D.precision("low"), D.precision("high")
    lcmv = {"low_baseline": lcmv_bias(*lo), "low_no_cfa": lcmv_bias(*lo, cfa_on=False),
            "low_signal_free_cov": lcmv_bias(*lo, sig_free=True),
            "low_lambda_0.5": lcmv_bias(*lo, lam=0.5), "low_lambda_5": lcmv_bias(*lo, lam=5.0),
            "high_baseline": lcmv_bias(*hi), "high_no_cfa": lcmv_bias(*hi, cfa_on=False),
            "high_signal_free_cov": lcmv_bias(*hi, sig_free=True),
            # covariance noise level: block average (too clean) vs recorded between-beat
            "low_block_average_cov": lcmv_bias(*lo, noise_var=nb ** 2),
            "high_block_average_cov": lcmv_bias(*hi, noise_var=nb ** 2),
            "low_recorded_noise_cov": lcmv_bias(*lo, noise_var=1171.0 ** 2),
            "high_recorded_noise_cov": lcmv_bias(*hi, noise_var=1171.0 ** 2)}

    # --- estimates for the figure (last calibration draw, HEF latency) -----
    ci = {c: i for i, c in enumerate(cells)}
    ai = {a: i for i, a in enumerate(arms)}
    est = {f"{c}|{a}": res.rec_noisy[(ci[c], ai[a])][:, -1] for c in cells for a in ("none", "glm")}

    # --- 2-D layout of the 85 sites (azimuthal projection from the vertex) --
    pos = arr.site_pos_mm - arr.site_pos_mm.mean(0)
    top = pos[:, 2].max()
    v = pos - np.array([0, 0, top - np.linalg.norm(pos, axis=1).max()])
    th = np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))
    ph = np.arctan2(v[:, 1], v[:, 0])
    lay = np.c_[th * np.cos(ph), th * np.sin(ph)]
    lay = lay / np.abs(lay).max() * 0.45

    numbers = {
        "calibration": cal,
        "signal_rms_ft": s, "signal_radial_rms_ft": float(rms(hef[:ns, k])),
        "signal_best_channel_ft": float(abs(hef[best, k])),
        "cfa_rms_hef_latency_ft": float(rms(cfa[:, k])),
        "cfa_radial_rms_hef_latency_ft": float(rms(cfa[:ns, k])),
        "cfa_rms_r_peak_ft": float(rms(cfa[:, i_r])),
        "cfa_radial_rms_r_peak_ft": float(rms(cfa[:ns, i_r])),
        "ratio_all": float(rms(cfa[:, k]) / s),
        "ratio_radial": float(rms(cfa[:ns, k]) / rms(hef[:ns, k])),
        "gate_ft": gate, "noise_per_beat_ft": D.noise_ft, "n_channels": arr.n_chan,
        "n_sites": ns, "source_mismatch_pct": float(res.mismatch_pct),
        "geometry": geometry, "cfa_topography_r_vs_still": r_all,
        "cfa_topography_r_min": min(r_all.values()),
        "bias_ft": bias, "sigma_ft": sigma, "beats": beats,
        "beats_at_recorded_noise": beats_recnoise, "recorded_noise_scale": noise_scale,
        "anova_glm": ana["anova"]["glm"], "lcmv_diagnostic_ft": lcmv,
        "axis_split_hef_latency": ana["axis_split"],
        "recorded_beats": n_rec,
    }
    (OUT / "paper_numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
    np.savez(OUT / "paper_figdata.npz", t_ms=t_ms, hef=hef, cfa=cfa, best=best, k=k,
             i_r=i_r, lay=lay, n_site=ns,
             cfa_by_ori=np.stack([cfa_k[n] for n in range(len(fwd.angles))]),
             angles=fwd.angles, truth=res.truth,
             **{f"est_{k_.replace(' ', '_').replace('|', '__')}": v for k_, v in est.items()},
             rec_t=tr, rec_rms=rec, sim_rad_rms=rms(cfa[:ns], 0))
    print(json.dumps({k_: numbers[k_] for k_ in ("signal_rms_ft", "cfa_rms_hef_latency_ft",
                      "ratio_all", "ratio_radial", "gate_ft", "cfa_topography_r_min")}, indent=1))
    print(json.dumps(geometry, indent=1))
    print(json.dumps(lcmv, indent=1))
    print(json.dumps(beats_recnoise, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
