"""Post-processing of one 2x2 simulation: ANOVA, bootstrap, detection, sweep.

Ports of ``CFA_Simulation_2x2_ANOVA.m``, ``..._2x2_Bootstrap_Compare.m``,
``..._DetectionCurves.m`` and ``..._RotationSweep.m``. Everything here reads a
single :class:`~inob.cfa.engine.SimResult`, so the numbers in the ANOVA, the
bootstrap and the detection curves cannot disagree -- they are the same
per-replicate recoveries seen four ways.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from inob.cfa.engine import SimResult
from inob.cfa.stats import TwoWayAnova, error_stats, two_way_anova

logger = logging.getLogger(__name__)

__all__ = ["beats_to_detect", "detection_table", "anova_over_arms",
           "bootstrap_compare", "DetectionRow", "axis_split"]


def axis_split(field: np.ndarray, n_site: int) -> dict[str, float]:
    """RMS of a field over each axis family of a triaxial array.

    Channel order is all-radial, all-tangent-1, all-tangent-2
    (:func:`inob.sensors.build_triaxial`), so this is a reshape. Worth reporting
    separately because the two families do not carry the same thing: a source
    inside the head produces a large in-plane field near itself and a smaller
    radial one, so a radial-only array discards most of a shallow cortical
    signal -- while a distant source like the heart arrives as a near-uniform
    gradient that both families see.
    """
    f = np.asarray(field, float).ravel()
    r = lambda v: float(np.sqrt((v ** 2).mean()))
    return {"radial": r(f[:n_site]), "tangential_1": r(f[n_site:2 * n_site]),
            "tangential_2": r(f[2 * n_site:])}


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def beats_to_detect(signal_ft: float, bias_ft: float, sigma_ft: float,
                    beats_at_sigma: float, snr_target: float = 3.0) -> float:
    """Beats needed to reach ``snr_target``, or ``inf`` if the bias forbids it.

    The estimator's error has two parts that behave completely differently:

        SNR(N) = s / sqrt( bias^2 + sigma^2 * N0 / N )

    ``sigma`` was measured at ``N0 = beats_at_sigma`` beats and falls as
    1/sqrt(N); ``bias`` is R-locked, identical on every beat, and never averages
    down at all. So there is a hard gate: if ``bias >= s / snr_target`` no
    recording length whatsoever reaches the target, and returning a large finite
    number there would be a fiction. That gate -- not the beat count -- is the
    real result in the uncorrected arm (Dirlich et al. 1997).
    """
    if not np.isfinite(bias_ft) or not np.isfinite(sigma_ft):
        return float("inf")
    budget = (signal_ft / snr_target) ** 2 - bias_ft ** 2
    if budget <= 0:
        return float("inf")
    # At least one beat: the formula is a continuous approximation and can return
    # a fraction when the array is large enough that noise was never the limit.
    # Reporting "0 beats" would state something the experiment cannot do.
    return max(1.0, float(np.ceil(sigma_ft ** 2 * beats_at_sigma / budget)))


@dataclass(frozen=True)
class DetectionRow:
    cell: str
    arm: str
    bias_ft: float
    sigma_ft: float
    beats: float
    minutes: float
    gate_open: bool


def detection_table(res: SimResult) -> list[DetectionRow]:
    """Beats (and minutes at the design heart rate) to detect, per cell and arm."""
    out: list[DetectionRow] = []
    for c, cell in enumerate(res.cell_labels):
        for a, arm in enumerate(res.arms):
            st = res.stats(c, a)
            if st is None:
                continue
            n = beats_to_detect(res.sig_rms, st.bias, st.sigma,
                                res.beats_block[c] * len(res.design.angles(
                                    res.design.cells[c][1])),
                                res.design.snr_target)
            out.append(DetectionRow(
                cell=cell, arm=arm, bias_ft=st.bias, sigma_ft=st.sigma,
                beats=n, minutes=n / res.design.hr_bpm if np.isfinite(n) else np.inf,
                gate_open=np.isfinite(n)))
    return out


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------
def anova_over_arms(res: SimResult) -> dict[str, TwoWayAnova]:
    """Partition log10(signal / bias) over precision and movement, per arm.

    The dependent variable is a log ratio because the arms span three orders of
    magnitude of bias and an ANOVA on the raw fT would be dominated by whichever
    arm happens to be worst. ``log10(s/bias)`` is the headroom in decades: 0 means
    the bias equals the signal, positive means the signal is above it.

    Per-replicate values, not cell means, so ``SS_err`` has something to be: each
    Monte-Carlo calibration draw is one observation.
    """
    d = res.design
    a_prec = np.array([1 if p == "low" else 0 for p, _ in d.cells])
    b_move = np.array([1 if m == "high" else 0 for _, m in d.cells])
    out: dict[str, TwoWayAnova] = {}
    for ai, arm in enumerate(res.arms):
        if not res.applicable[:, ai].all():
            continue                                   # unbalanced: CDM has no still
        ys, A, B = [], [], []
        for c in range(d.n_cells):
            rb = res.rec_bias[(c, ai)]
            for r in range(d.n_rep):
                e = rb[:, r] - res.truth
                bias_r = float(np.sqrt((e ** 2).mean()))
                ys.append(np.log10(res.sig_rms / max(bias_r, 1e-30)))
                A.append(a_prec[c]); B.append(b_move[c])
        out[arm] = two_way_anova(np.array(A), np.array(B), np.array(ys), d.n_rep)
    return out


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def bootstrap_compare(res: SimResult, arm_a: str, arm_b: str, *,
                      n_boot: int = 10_000, seed: int = 7) -> dict:
    """Paired bootstrap of the bias difference between two arms, per cell.

    Paired over calibration draws, because both arms saw the SAME array and the
    same data on every replicate -- resampling them independently would throw away
    the pairing the engine went to some trouble to guarantee, and inflate the
    interval.
    """
    ia, ib = res.arms.index(arm_a), res.arms.index(arm_b)
    rng = np.random.default_rng(seed)
    out = {}
    for c, cell in enumerate(res.cell_labels):
        if not (res.applicable[c, ia] and res.applicable[c, ib]):
            continue
        ea = res.rec_bias[(c, ia)] - res.truth[:, None]
        eb = res.rec_bias[(c, ib)] - res.truth[:, None]
        ba = np.sqrt((ea ** 2).mean(0))
        bb = np.sqrt((eb ** 2).mean(0))
        n = len(ba)
        idx = rng.integers(0, n, size=(n_boot, n))
        diff = np.sqrt((ba[idx] ** 2).mean(1)) - np.sqrt((bb[idx] ** 2).mean(1))
        lo, hi = np.percentile(diff, [2.5, 97.5])
        out[cell] = {
            "bias_a": float(np.sqrt((ba ** 2).mean())),
            "bias_b": float(np.sqrt((bb ** 2).mean())),
            "diff": float(np.sqrt((ba ** 2).mean()) - np.sqrt((bb ** 2).mean())),
            "ci95": (float(lo), float(hi)),
            "p_a_worse": float((diff > 0).mean()),
        }
    return out
