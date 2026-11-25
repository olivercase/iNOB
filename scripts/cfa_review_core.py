"""Core of the CFA/HEF review-response simulations (no existing file is modified).

Reproduces the simulation of the manuscript (precision x movement 2x2, 500 beats per
condition, 16 calibration draws, 10 nAm insula source) from the STORED lead fields in
outputs/cfa_forward, with its own data generator, and adds the estimators the review
asks for (empirical QRS-window SSP, nulling LCMV) plus a per-beat cardiac-modulation
generator for the nested T-wave case.

Everything is linear in the sources, so the cardiac field of orientation n is built as
    ccfa_n(t) = G_n  coeff_n(t),     G_n = gain * project(C_n, true_axes)   (255 x K)
with K = 3 (static distributed source: amps = B3 q(t), so C_n = b_myo_n B3 * scale) or
K = 400 (propagating source, one time course per wall site).

The loader below is an independent copy of the head-frame logic of
inob.cfa.engine.Forward.load; `regression()` checks it against the repo.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inob.cfa.arms import arm_cdm, arm_hfc_lcmv           # noqa: E402
from inob.cfa.array import CFAArray                       # noqa: E402
from inob.cfa.design import DESIGN as D                   # noqa: E402
from inob.cfa.hfc import harmonic_projector               # noqa: E402
from inob.cfa.perturb import perturb_array                # noqa: E402
from inob.cfa.posture import rotation_matrix              # noqa: E402

FWD = ROOT / "outputs/cfa_forward"
R_MS = 400.0                                  # R-peak in the simulation window, ms
REC_NOISE_FT = 1171.0                         # recorded between-beat array-RMS noise
REC_SCALE = REC_NOISE_FT / D.noise_ft         # 12.34
N0 = D.n_beats
CELL_ORDER = D.cell_labels                    # SQUID, future OPM, worst case, current OPM


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def hep_wave(t, latency, fwhm=0.10):
    s = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * ((np.asarray(t, float) - latency) / s) ** 2)


def reproject_ins(arr, l_nom, nom_ori, true_ori):
    """Own copy of the engine's re-reading of the insula leadfield on perturbed axes."""
    s = arr.n_site
    i = np.arange(s)
    o_nom = np.stack([nom_ori[i], nom_ori[i + s], nom_ori[i + 2 * s]], 1)
    l3 = np.stack([l_nom[i], l_nom[i + s], l_nom[i + 2 * s]], 1)
    field = np.einsum("scd,sck->sdk", o_nom, l3)
    return np.einsum("sdk,sd->sk", field[arr.site_of()], true_ori)


@dataclass
class Setup:
    arr: CFAArray
    angles: np.ndarray
    l_ins0: np.ndarray            # (255,3) head-frame insula leadfield, still head
    b_raw: list                   # per orientation (S,3,N,3) head-frame raw myocardial field
    b_myo: list                   # per orientation (S,3,N) contracted on wall normals
    b_eq: list                    # per orientation (S,3,6)
    q: np.ndarray                 # (3,n_t) cardiac moment
    amps: np.ndarray              # (N,n_t) calibrated static site amplitudes
    b3: np.ndarray                # (N,3), amps = b3 @ q
    c3: list                      # per orientation (S,3,3) = b_myo @ b3 (calibrated)
    time: np.ndarray
    k: int
    i_r: int
    amp_scale: float
    myo_ori: np.ndarray
    myo_pos: np.ndarray
    myo_area: np.ndarray
    prop: object = None           # optional PropagatingCardiacSource (built lazily)
    prop_b: list = None

    @property
    def nom(self):
        return self.arr.chanori

    def index_of(self, ypr):
        d = np.linalg.norm(self.angles - np.asarray(ypr, float), axis=1)
        i = int(d.argmin())
        assert d[i] < 1e-6, ypr
        return i


def load_setup(frame: str = "head") -> Setup:
    arr = CFAArray.from_reference(ROOT / "outputs/_cfa_ref.npz")
    s = np.load(FWD / "sources.npz")
    myo_ori, q, w = s["myo_ori"], s["q"], s["myo_area_mm2"]
    w = w / w.mean()
    nw = (myo_ori * w[:, None]).T
    b3 = w[:, None] * (nw.T @ np.linalg.inv(nw @ nw.T))            # amps = b3 @ q
    scale = float(json.loads((FWD / "calibration.json").read_text())["amp_scale"])
    amps = scale * (b3 @ q)
    ns = arr.n_site
    angles, l_ins, b_raw, b_myo, b_eq = [], [], [], [], []
    for f in sorted(FWD.glob("orient_*.npz")):
        z = np.load(f)
        ypr = np.asarray(z["ypr_deg"], float)
        angles.append(ypr)
        rot = rotation_matrix(ypr)
        l_ins.append(arr.project(z["B_ins"], z["chanori"]) @ rot)
        braw = z["B_myo"].reshape(ns, 3, -1, 3)
        bw = np.einsum("scnk,nk->scn", braw, myo_ori)
        be = z["B_eq"]
        if frame == "head":
            bw = np.einsum("cd,scn->sdn", rot, bw)
            be = np.einsum("cd,scn->sdn", rot, be)
            braw = np.einsum("cd,scnk->sdnk", rot, braw)
        b_raw.append(braw)
        b_myo.append(bw)
        b_eq.append(be)
    order = np.argsort([np.linalg.norm(a) for a in angles], kind="stable")
    pick = lambda xs: [xs[i] for i in order]
    t = s["time"]
    st = Setup(arr=arr, angles=np.array(angles)[order], l_ins0=pick(l_ins)[0],
               b_raw=pick(b_raw), b_myo=pick(b_myo), b_eq=pick(b_eq), q=q,
               amps=amps, b3=scale * b3, c3=[], time=t, k=int(s["erp_index"]),
               i_r=int(np.argmin(np.abs(t * 1000 - R_MS))), amp_scale=scale,
               myo_ori=myo_ori, myo_pos=s["myo_pos_mm"], myo_area=s["myo_area_mm2"])
    st.c3 = [np.einsum("scn,nk->sck", b, st.b3) for b in st.b_myo]
    return st


def build_propagating(st: Setup):
    """Propagating myocardial source (repo's own generator) in the stored lead fields."""
    from inob.cfa.cardiac import PropagatingCardiacSource
    from inob.cfa.sources import MyocardialSource
    src = MyocardialSource(pos_mm=st.myo_pos, ori=st.myo_ori, area_mm2=st.myo_area)
    prop = PropagatingCardiacSource.build(src, st.time, st.q)
    ns = st.arr.n_site
    st.prop = prop
    st.prop_b = [np.einsum("scnk,nk->scn", b, prop.ori) for b in st.b_raw]
    st.prop_amps = prop.amps * st.amp_scale
    return st


# ---------------------------------------------------------------------------
# cardiac waveform modulation (nested case, R-peak jitter)
# ---------------------------------------------------------------------------
def qrs_t_split(st: Setup, split_ms=100.0, taper_ms=40.0):
    """q = qQ + qT with a raised-cosine taper between R+split-taper/2 and R+split+taper/2."""
    trel = st.time * 1000 - R_MS
    x = np.clip((trel - (split_ms - taper_ms / 2)) / taper_ms, 0, 1)
    wT = 0.5 - 0.5 * np.cos(np.pi * x)
    return st.q * (1 - wT), st.q * wT


def beat_mean_coeff(st: Setup, n_beats: int, rng, *, t_cv=0.0, q_cv=0.0,
                    t_jit_ms=0.0, r_jit_ms=0.0, parts=None, chunk=2000):
    """Mean over n_beats beats of the per-beat cardiac moment (3,n_t).

    beat b: qQ(t - dR) * gQ + qT(t - dR - dT) * gT; gQ~N(1,q_cv), gT~N(1,t_cv),
    dT~N(0,t_jit) (T-peak latency jitter relative to the R-peak), dR~N(0,r_jit)
    (trigger jitter of the whole cardiac waveform). Sub-sample shifts by cubic spline.
    """
    if t_cv == q_cv == t_jit_ms == r_jit_ms == 0:
        return st.q.copy()
    qq, qt = parts if parts is not None else qrs_t_split(st)
    t = st.time
    csq, cst = CubicSpline(t, qq, axis=1), CubicSpline(t, qt, axis=1)
    acc = np.zeros_like(st.q)
    done = 0
    while done < n_beats:
        m = min(chunk, n_beats - done)
        gq = 1 + q_cv * rng.standard_normal(m)
        gt = 1 + t_cv * rng.standard_normal(m)
        dr = r_jit_ms * 1e-3 * rng.standard_normal(m)
        dt = t_jit_ms * 1e-3 * rng.standard_normal(m)
        tq = np.clip(t[None] - dr[:, None], t[0], t[-1])
        tt = np.clip(t[None] - (dr + dt)[:, None], t[0], t[-1])
        vq = csq(tq)            # (3, m, n_t)
        vt = cst(tt)
        acc += (vq * gq[None, :, None]).sum(1) + (vt * gt[None, :, None]).sum(1)
        done += m
    return acc / n_beats


# ---------------------------------------------------------------------------
# data generation
# ---------------------------------------------------------------------------
def unit_hef(st: Setup, true_ori, gain):
    """Spatial pattern (255,) of the measured HEF per nAm, fT, direction (1,1,1)/sqrt3."""
    m_dir = np.ones(3) / np.sqrt(3.0)
    return gain * (reproject_ins(st.arr, st.l_ins0, st.nom, true_ori) @ m_dir) * 1e-9 * 1e15


def cardiac_fields(st: Setup, idx, true_ori, gain, coeffs=None, cardiac="static"):
    """(n_or,255,n_t) cardiac field at the sensors, fT. coeffs: list of (K,n_t) per
    orientation (static: K=3 moment time courses) or None for the common source."""
    out = []
    for j, n in enumerate(idx):
        if cardiac == "static":
            g3 = gain[:, None] * st.arr.project(st.c3[n], true_ori)          # (255,3)
            c = st.q if coeffs is None else coeffs[j]
            out.append(g3 @ c * 1e15)
        else:
            gk = gain[:, None] * st.arr.project(st.prop_b[n], true_ori)      # (255,N)
            out.append(gk @ st.prop_amps * 1e15)
    return np.stack(out)


# ---------------------------------------------------------------------------
# estimators.  All return (rb, rn): (255,) recovered HEF field at the HEF latency from
# the noise-free data and from the noisy data, with the SAME weights unless stated.
# ---------------------------------------------------------------------------
class Cell:
    """Per-(precision, movement) constants: orientation set, model lead fields."""

    def __init__(self, st: Setup, c: int):
        self.st = st
        self.c = c
        pname, mname = D.cells[c]
        self.pname, self.mname = pname, mname
        self.gain_sd, self.tilt_sd = D.precision(pname)
        self.idx = [st.index_of(a) for a in D.angles(mname)]
        self.n_or = len(self.idx)
        self.beats = D.n_beats / self.n_or
        self.nb = D.noise_ft / np.sqrt(self.beats)
        self.l_mod = [st.arr.project(st.b_eq[n], st.nom) for n in self.idx]
        self.phi = np.linalg.qr(st.l_ins0)[0]
        self.n_chan = st.arr.n_chan
        self.lbar = np.concatenate(self.l_mod, 0)                           # (N,6)
        self.phibar = np.tile(self.phi, (self.n_or, 1))                     # (N,3)
        self.A = np.concatenate([self.phibar, self.lbar], 1)                # (N,9)
        self.scale = np.linalg.norm(self.A, axis=0)
        self.As = self.A / self.scale
        # orthogonalisation weights (3 x N):  Eq. 2 of the manuscript
        q_ = np.linalg.qr(self.lbar)[0]
        mphi = self.phibar - q_ @ (q_.T @ self.phibar)
        self.w_orth = np.linalg.solve(mphi.T @ mphi, mphi.T)
        self.proj = {o: harmonic_projector(st.arr.chanpos_mm, st.nom, o)
                     for o in (1, 5)}
        trel = st.time * 1000 - R_MS
        self.qrs_win = (trel >= -40) & (trel <= 60)

    # --- own orthogonalisation (Eq. 2) ----------------------------------------
    def orth(self, yc, yn):
        k = self.st.k
        f = lambda y: self.phi @ (self.w_orth @ y[:, :, k].reshape(-1))
        return f(yc), f(yn)

    def none(self, yc, yn):
        k = self.st.k
        return yc.mean(0)[:, k], yn.mean(0)[:, k]

    def hfc(self, order, noise_ft=None):
        nv = (D.noise_ft if noise_ft is None else noise_ft) ** 2

        def f(yc, yn):
            rb, rn = arm_hfc_lcmv(yc, yn, self.proj[order], self.phi,
                                  lambda_reg=D.lambda_reg, noise_var=nv)
            return rb[:, self.st.k], rn[:, self.st.k]
        return f

    def cdm(self, yc, yn):
        rb, rn = arm_cdm(yc, yn, self.l_mod)
        return rb[:, self.st.k], rn[:, self.st.k]

    # --- nulling LCMV on stacked data -----------------------------------------
    def null_weights(self, cov):
        """W (3 x N): min w'Cw s.t. W'[Phibar Lbar] = [I 0]; scaled-column solve."""
        if cov is None:                                    # identity covariance
            cia = self.As
        else:
            cia = np.linalg.solve(cov, self.As)
        m = self.As.T @ cia
        w = (cia @ np.linalg.solve(m, np.eye(9)[:, :3])).T  # (3,N) in scaled coordinates
        return w / self.scale[:3, None]

    def nulling(self, kind, noise_ft=None):
        nv = (D.noise_ft if noise_ft is None else noise_ft) ** 2

        def f(yc, yn):
            k = self.st.k
            if kind == "white":
                w = self.null_weights(None)
            else:
                x = yc.reshape(-1, yc.shape[2])            # (N, n_t)
                cov = np.cov(x) + nv * np.eye(x.shape[0])
                cov += D.lambda_reg * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
                w = self.null_weights(cov)
            g = lambda y: self.phi @ (w @ y[:, :, k].reshape(-1))
            return g(yc), g(yn)
        return f

    # --- empirical QRS-window SSP -----------------------------------------------
    def ssp(self, rank, use_noise_subspace):
        """Per-orientation SSP: cardiac subspace = top-`rank` left singular vectors of the
        block-averaged data in -40..+60 ms around the R-peak; data and brain lead field
        projected off it; joint least squares across orientations."""
        k = self.st.k
        win = self.qrs_win

        def estimate(y_sub, y_eval):
            a_rows, y_rows = [], []
            for j in range(self.n_or):
                u = np.linalg.svd(y_sub[j][:, win], full_matrices=False)[0][:, :rank]
                p_phi = self.phi - u @ (u.T @ self.phi)
                a_rows.append(p_phi)
                y_rows.append(y_eval[j][:, k] - u @ (u.T @ y_eval[j][:, k]))
            a = np.concatenate(a_rows, 0)
            y = np.concatenate(y_rows, 0)
            return self.phi @ np.linalg.lstsq(a, y, rcond=None)[0]

        def f(yc, yn):
            rb = estimate(yc, yc)
            rn = estimate(yn if use_noise_subspace else yc, yn)
            return rb, rn
        return f


def method_table(cell: Cell, names):
    tab = {}
    for nme in names:
        if nme == "none":
            tab[nme] = (cell.none, True)
        elif nme == "orth":
            tab[nme] = (cell.orth, True)
        elif nme.startswith("hfc") and nme.endswith("_rc"):
            tab[nme] = (cell.hfc(int(nme[3:-3]), REC_NOISE_FT), True)   # recorded-noise covariance
        elif nme.startswith("hfc"):
            tab[nme] = (cell.hfc(int(nme[3:])), True)          # lcmv weights from yc only
        elif nme == "cdm":
            tab[nme] = (cell.cdm, True)
        elif nme == "null_white":
            tab[nme] = (cell.nulling("white"), True)
        elif nme == "null_sample":
            tab[nme] = (cell.nulling("sample"), True)
        elif nme == "null_sample_rc":
            tab[nme] = (cell.nulling("sample", REC_NOISE_FT), True)
        elif nme.startswith("ssp_k"):
            tab[nme] = (cell.ssp(int(nme[5:]), True), False)    # non-linear in the noise
        else:
            raise KeyError(nme)
    return tab


def rms(x, axis=None):
    return float(np.sqrt(np.mean(np.square(x), axis=axis)))


# ---------------------------------------------------------------------------
# one calibration draw of one cell
# ---------------------------------------------------------------------------
def run_draw(st: Setup, cell: Cell, rng_cal, rng_noise, *, amps_nam, methods,
             coeffs=None, cardiac="static", with_noise=True, calib=None, keep_vec=False, n_noise=4):
    """Generate one calibration draw and score every method at every HEF amplitude.

    Returns {(method, amp): dict(eb, en, est, en_rec, est_rec, truth_rms)}.
    eb = array-RMS of (noise-free estimate - truth); en = array-RMS of (noisy - noise-free)
    estimate at the sensor-noise level; est = array-RMS of the noisy estimate; *_rec the
    same with noise at the recorded level (x REC_SCALE).
    """
    st_k = st.k
    true_ori, gain = calib if calib is not None else perturb_array(
        st.nom, cell.gain_sd, cell.tilt_sd, rng_cal)
    ccfa = cardiac_fields(st, cell.idx, true_ori, gain, coeffs, cardiac)
    hef_sp = unit_hef(st, true_ori, gain)
    wave = hep_wave(st.time, float(st.time[st_k]))
    zs = ([rng_noise.standard_normal(ccfa.shape) for _ in range(n_noise)] if with_noise
          else [np.zeros_like(ccfa)])
    truth_unit = (st.l_ins0 @ (np.ones(3) / np.sqrt(3.0))) * 1e-9 * 1e15
    tab = method_table(cell, methods)
    out = {}
    for a in amps_nam:
        yc = ccfa + (a * hef_sp[:, None] * wave[None])[None]
        truth = a * truth_unit * wave[st_k]
        for nme, (fn, linear) in tab.items():
            en, est, enr, estr, et, etr = [], [], [], [], [], []
            for z in zs:
                rb, rn = fn(yc, yc + cell.nb * z)
                if linear:
                    rr = rb + REC_SCALE * (rn - rb)
                else:
                    rr = fn(yc, yc + REC_SCALE * cell.nb * z)[1]
                en.append(rms(rn - rb))
                est.append(rms(rn))
                enr.append(rms(rr - rb))
                estr.append(rms(rr))
                et.append(rms(rn - truth))
                etr.append(rms(rr - truth))
            q = lambda v: float(np.sqrt(np.mean(np.square(v))))
            out[(nme, float(a))] = dict(eb=rms(rb - truth), en=q(en), est=q(est), en_rec=q(enr),
                                        est_rec=q(estr), et=q(et), et_rec=q(etr), truth_rms=rms(truth))
            if keep_vec:
                out[(nme, float(a))]["vec"] = rb - truth
    return out


def detect_beats(s_ft, bias, sigma):
    """Heartbeats for SNR = 3 (manuscript Eq. 4); None = never."""
    rhs = (s_ft / D.snr_target) ** 2 - bias ** 2
    return None if rhs <= 0 else float(N0 * sigma ** 2 / rhs)


def aggregate(recs):
    """recs: list over draws of {(m,a): dict}. -> {(m,a): summary}."""
    out = {}
    for key in recs[0]:
        rows = [r[key] for r in recs]
        g = lambda f: float(np.sqrt(np.mean([r[f] ** 2 for r in rows])))
        s = rows[0]["truth_rms"]
        bias, sig, sigr = g("eb"), g("en"), g("en_rec")
        out[key] = dict(
            bias=bias, sigma=sig, sigma_rec=sigr, est_rms=g("est"),
            est_rms_rec=g("est_rec"), total_err=g("et"), total_err_rec=g("et_rec"), s=s, limit=s / D.snr_target,
            beats=detect_beats(s, bias, sig), beats_rec=detect_beats(s, bias, sigr),
            n_draw=len(rows))
    return out
