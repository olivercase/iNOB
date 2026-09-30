"""The precision x movement 2x2: one simulation, every arm scored on the same data.

Port of ``CFA_Simulation_2x2_Engine.m`` onto the FEM leadfields.

This module only SIMULATES and CORRECTS. It computes no statistics and draws no
figures; the ANOVA, bootstrap, detection curves and rotation sweep are all pure
post-processing of its output, so the arms are paired by construction and the
reported numbers cannot drift between scripts.

GENERATIVE MODEL. Per orientation block n, with the array as it physically is
(tilted axes, gains r) and the heart as it actually is (a current density spread
over the myocardial wall):

    Y_n(t) = r * [ L_ins(head frame) m(t)  +  L_myo,n a(t) ]  +  noise_n(t)

Every estimator builds its model from the NOMINAL array and from a TWO-DIPOLE
equivalent heart at a template position. Those are the two mismatches and they are
the experiment:

* the calibration mismatch is the **precision** factor;
* the source mismatch is what gives **movement** work to do. It is torso-fixed, so
  it rotates relative to the head-fixed array, and a spread of orientations
  averages it down where a single orientation cannot. Note what has changed from
  the replication pack: there the mismatch was a hand-set 10.2 mm displacement of
  a point dipole. Here the truth is a distributed source over real myocardium, so
  what the 6-dof model cannot fit is the heart's actual spatial extent -- its
  quadrupole and higher content. The template position error is retained on top,
  because a real template carries one regardless of how well its shape is modelled.

**The insula leadfield is held fixed across orientations.** Array, skull and brain
are one rigid body, so the head-local geometry is identical at every orientation
and the HEF topography cannot drift; that invariance is what makes the joint GLM
identifiable. The FEM's measured departure from it (a 1.6-3.7 deg principal angle,
from the non-rotating neck and from remeshing) is reported by
``hep_invariance="measured"`` as a robustness check, not baked into the model.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from inob.cfa.array import CFAArray
from inob.cfa.arms import arm_cdm, arm_glm, arm_hfc_lcmv, arm_none
from inob.cfa.design import DESIGN, Design
from inob.cfa.hfc import harmonic_projector
from inob.cfa.perturb import perturb_array
from inob.cfa.posture import rotation_matrix
from inob.cfa.stats import ErrorStats, error_stats

logger = logging.getLogger(__name__)

__all__ = ["Forward", "SimResult", "run_2x2", "hep_waveform"]


def hep_waveform(time_s: np.ndarray, latency_s: float, fwhm_s: float = 0.10
                 ) -> np.ndarray:
    """Unit-peak Gaussian evoked response.

    The pack takes its HEF timecourse from a cached real average; there is no
    atlas equivalent, and the 2x2 never uses the waveform's shape -- every arm is
    evaluated at a single latency. A Gaussian of evoked-response width keeps the
    simulation self-contained without pretending to more realism than is used.
    """
    sigma = fwhm_s / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((np.asarray(time_s, float) - latency_s) / sigma) ** 2)


@dataclass
class Forward:
    """Everything the engine needs from the FEM stage, in the head frame."""

    array: CFAArray
    angles: np.ndarray               # (n_or, 3) yaw/pitch/roll, still first
    l_ins_head: list[np.ndarray]     # per orientation, (n_chan, 3) head-frame moments
    b_myo: list[np.ndarray]          # per orientation, (n_site, 3, n_myo) contracted
    b_eq: list[np.ndarray]           # per orientation, (n_site, 3, 6)
    amps: np.ndarray                 # (n_myo, n_t) myocardial site amplitudes, A.m
    time_s: np.ndarray
    erp_index: int

    @property
    def n_or(self) -> int:
        return len(self.angles)

    def index_of(self, ypr) -> int:
        d = np.linalg.norm(self.angles - np.asarray(ypr, float), axis=1)
        i = int(d.argmin())
        if d[i] > 1e-6:
            raise KeyError(f"orientation {ypr} was not solved")
        return i

    @classmethod
    def load(cls, forward_dir: Path, array: CFAArray) -> "Forward":
        d = Path(forward_dir)
        s = np.load(d / "sources.npz")
        myo_ori = s["myo_ori"]
        q, w = s["q"], s["myo_area_mm2"]
        w = w / w.mean()
        nw = (myo_ori * w[:, None]).T
        amps = w[:, None] * (nw.T @ np.linalg.solve(nw @ nw.T, q))
        # Amplitude calibration against the recorded head field (see
        # inob.cfa.calibrate). Applied here so every consumer sees it.
        cal = d / "calibration.json"
        if cal.exists():
            amps = amps * float(json.loads(cal.read_text())["amp_scale"])
        n_site = array.n_site
        angles, l_ins, b_myo, b_eq = [], [], [], []
        for f in sorted(d.glob("orient_*.npz")):
            z = np.load(f)
            ypr = np.asarray(z["ypr_deg"], float)
            angles.append(ypr)
            # Head-frame insula leadfield: read with the rotated nominal axes,
            # then express the moment in head coordinates so it is comparable
            # across orientations.
            l_ins.append(array.project(z["B_ins"], z["chanori"]) @ rotation_matrix(ypr))
            b_myo.append(np.einsum("scnk,nk->scn",
                                   z["B_myo"].reshape(n_site, 3, -1, 3), myo_ori))
            b_eq.append(z["B_eq"])
        order = np.argsort([np.linalg.norm(a) for a in angles], kind="stable")
        pick = lambda xs: [xs[i] for i in order]
        return cls(array=array, angles=np.array(angles)[order],
                   l_ins_head=pick(l_ins), b_myo=pick(b_myo), b_eq=pick(b_eq),
                   amps=amps, time_s=s["time"], erp_index=int(s["erp_index"]))


@dataclass
class SimResult:
    """Recovered fields per cell and arm, plus the diagnostics the papers use."""

    design: Design
    arms: tuple[str, ...]
    cell_labels: tuple[str, ...]
    rec_bias: dict                    # (cell, arm) -> (n_chan, n_rep)
    rec_noisy: dict
    applicable: np.ndarray            # (n_cell, n_arm) bool
    truth: np.ndarray                 # (n_chan,) HEF field at the latency
    sig_rms: float
    cfa_rms: float
    noise_block: np.ndarray           # (n_cell,)
    beats_block: np.ndarray           # (n_cell,)
    phi_keep: np.ndarray              # (n_cell, n_hfc) insula basis surviving HFC
    sf_hfc: np.ndarray                # (n_cell, n_hfc) CFA shielding factor
    mismatch_pct: float

    def stats(self, cell: int, arm: int) -> ErrorStats | None:
        if not self.applicable[cell, arm]:
            return None
        return error_stats(self.rec_bias[(cell, arm)], self.rec_noisy[(cell, arm)],
                           self.truth)


def run_2x2(fwd: Forward, design: Design = DESIGN, *,
            source_nam: float = 10.0,
            hep_invariance: str = "exact") -> SimResult:
    """Sweep the 2x2, scoring every arm on identical data."""
    arr = fwd.array
    n_chan = arr.n_chan
    nom_ori = arr.chanori
    chanpos = arr.chanpos_mm
    t = fwd.time_s
    k = fwd.erp_index
    n_t = len(t)

    # --- the HEF source ------------------------------------------------------
    # 3-dof insula source, fixed unit direction in the HEAD frame, driven by an
    # evoked waveform and scaled to an absolute neuronal dipole strength.
    m_dir = np.ones(3) / np.sqrt(3.0)
    moment = np.outer(m_dir, hep_waveform(t, float(t[k]))) * source_nam * 1e-9
    l_ins_model = fwd.l_ins_head[0]                       # held fixed: see module doc
    phi = np.linalg.qr(l_ins_model)[0]                    # (n_chan, 3) orthonormal
    truth = (l_ins_model @ moment)[:, k] * 1e15           # fT
    sig_rms = float(np.sqrt((truth ** 2).mean()))

    cfa_ref = (arr.project(fwd.b_myo[0], nom_ori) @ fwd.amps)[:, k] * 1e15
    cfa_rms = float(np.sqrt((cfa_ref ** 2).mean()))
    # how far the 6-dof model is from the distributed truth, at the nominal array
    l_eq0 = arr.project(fwd.b_eq[0], nom_ori)
    l_myo0 = arr.project(fwd.b_myo[0], nom_ori)
    resid = l_myo0 - l_eq0 @ np.linalg.lstsq(l_eq0, l_myo0, rcond=None)[0]
    mismatch_pct = 100.0 * np.linalg.norm(resid) / np.linalg.norm(l_myo0)
    logger.info("[engine] %d channels, HEF %.1f fT rms, CFA %.0f fT rms, "
                "source-model mismatch %.1f%% of the cardiac leadfield",
                n_chan, sig_rms, cfa_rms, mismatch_pct)

    arms = ("none", "glm", *[f"hfc{o}_lcmv" for o in design.hfc_orders], "cdm")
    i_none, i_glm = 0, 1
    i_hfc = tuple(2 + j for j in range(len(design.hfc_orders)))
    i_cdm = len(arms) - 1
    n_cell, n_arm = design.n_cells, len(arms)

    rec_bias: dict = {}
    rec_noisy: dict = {}
    applicable = np.zeros((n_cell, n_arm), bool)
    noise_block = np.zeros(n_cell)
    beats_block = np.zeros(n_cell)
    phi_keep = np.zeros((n_cell, len(design.hfc_orders)))
    sf_hfc = np.zeros((n_cell, len(design.hfc_orders)))

    for c, (pname, mname) in enumerate(design.cells):
        gain_sd, tilt_sd = design.precision(pname)
        idx = [fwd.index_of(a) for a in design.angles(mname)]
        n_or = len(idx)
        # Beats are held constant across the movement factor: N_BEATS split over
        # this cell's orientations, so total information is identical and movement
        # changes only the DIVERSITY of orientations.
        beats = design.n_beats / n_or
        nb = design.noise_ft / np.sqrt(beats)
        beats_block[c], noise_block[c] = beats, nb

        l_mod = [arr.project(fwd.b_eq[n], nom_ori) for n in idx]      # (n_chan, 6)
        gdes = np.concatenate([np.tile(phi, (n_or, 1)),
                               np.concatenate(l_mod, axis=0)], axis=1)

        # HFC projectors from the NOMINAL array. The span of solid harmonics up to
        # a given degree is invariant under rotation of the rigid array, so one
        # projector serves every orientation -- asserted, not assumed.
        proj = []
        for oi, order in enumerate(design.hfc_orders):
            p0 = harmonic_projector(chanpos, nom_ori, order)
            proj.append(p0)
            phi_keep[c, oi] = (np.linalg.norm(p0 @ phi) ** 2
                               / np.linalg.norm(phi) ** 2)

        # CDM runs in every cell. With one orientation it trains and tests on the
        # same orientation (Woelk et al. 2026 applied it to a still head too).
        have_cdm = True
        applicable[c, :] = True
        applicable[c, i_cdm] = have_cdm
        for a in range(n_arm):
            rec_bias[(c, a)] = np.full((n_chan, design.n_rep), np.nan)
            rec_noisy[(c, a)] = np.full((n_chan, design.n_rep), np.nan)

        rng = np.random.default_rng(design.rng_seed + c)
        sf_acc = np.zeros(len(design.hfc_orders))
        for rep in range(design.n_rep):
            true_ori, gain = perturb_array(nom_ori, gain_sd, tilt_sd, rng)
            g = gain[:, None]
            # measured HEF: head-fixed, so identical at every orientation
            l_ins_true = (fwd.l_ins_head[0] if hep_invariance == "exact"
                          else None)
            yc = np.empty((n_or, n_chan, n_t))
            ccfa = np.empty((n_or, n_chan, n_t))
            for j, n in enumerate(idx):
                lin = l_ins_true if l_ins_true is not None else fwd.l_ins_head[n]
                # The perturbed array reads the same physical fields along
                # different axes, and with different per-channel gains.
                ccfa[j] = g * (arr.project(fwd.b_myo[n], true_ori) @ fwd.amps) * 1e15
                hep_meas = g * (_reproject_ins(arr, lin, nom_ori, true_ori)
                                @ moment) * 1e15
                yc[j] = hep_meas + ccfa[j]
            yn = yc + nb * rng.standard_normal(yc.shape)

            for oi, p0 in enumerate(proj):
                pre = float((ccfa[:, :, k] ** 2).sum())
                post = float(((ccfa[:, :, k] @ p0.T) ** 2).sum())
                sf_acc[oi] += np.sqrt(pre / max(post, 1e-30))

            out = {i_none: arm_none(yc, yn),
                   i_glm: arm_glm(yc, yn, gdes, phi)}
            for oi, order in enumerate(design.hfc_orders):
                out[i_hfc[oi]] = arm_hfc_lcmv(yc, yn, proj[oi], phi,
                                              lambda_reg=design.lambda_reg,
                                              # Single-trial noise: a real beamformer's
                                              # covariance is estimated from single
                                              # trials, not from the block average.
                                              noise_var=design.noise_ft ** 2)
            if have_cdm:
                out[i_cdm] = arm_cdm(yc, yn, l_mod)
            for a, (rb, rn) in out.items():
                rec_bias[(c, a)][:, rep] = rb[:, k]
                rec_noisy[(c, a)][:, rep] = rn[:, k]

        sf_hfc[c] = sf_acc / design.n_rep
        logger.info("[engine] cell %-11s (prec %-4s / move %-4s): %d orient, "
                    "%.1f beats/block, block noise %.1f fT",
                    design.cell_labels[c], pname, mname, n_or, beats, nb)

    return SimResult(
        design=design, arms=arms, cell_labels=design.cell_labels,
        rec_bias=rec_bias, rec_noisy=rec_noisy, applicable=applicable,
        truth=truth, sig_rms=sig_rms, cfa_rms=cfa_rms,
        noise_block=noise_block, beats_block=beats_block,
        phi_keep=phi_keep, sf_hfc=sf_hfc, mismatch_pct=mismatch_pct)


def _reproject_ins(arr: CFAArray, l_nom: np.ndarray, nom_ori: np.ndarray,
                   true_ori: np.ndarray) -> np.ndarray:
    """Re-read the insula leadfield along PERTURBED axes.

    ``l_nom`` is the head-frame insula leadfield as the NOMINAL axes see it. The
    three axes at a triaxial site are orthonormal and span the full field vector,
    so the site's field vector is recoverable from its three nominal readings and
    can be re-projected onto any other axis. That is the whole reason the array is
    triaxial here: with a single-axis site the tilted axis would sample a
    component the site never measured.
    """
    s = arr.n_site
    idx = np.arange(s)
    # nominal axes at each site, as rows of a 3x3 orthonormal matrix
    o_nom = np.stack([nom_ori[idx], nom_ori[idx + s], nom_ori[idx + 2 * s]], 1)
    l3 = np.stack([l_nom[idx], l_nom[idx + s], l_nom[idx + 2 * s]], 1)  # (S,3,3)
    field = np.einsum("scd,sck->sdk", o_nom, l3)          # world field vector per site
    return np.einsum("sdk,sd->sk", field[arr.site_of()], true_ori)
