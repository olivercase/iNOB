"""Error decomposition and the balanced 2x2 variance partition.

Ports of ``cfa_error_stats.m`` and ``cfa_two_way_anova.m``. Both live here so the
ANOVA, the bootstrap, the detection curves and the rotation sweep cannot disagree
about what "bias" means -- which they did, by up to 11%, before the MATLAB build
centralised it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ErrorStats", "error_stats", "TwoWayAnova", "two_way_anova"]


def _rms_over_replicates(e: np.ndarray) -> float:
    """Array-RMS within each replicate, then RMS across replicates.

    Quadratic at both stages, deliberately. BIAS and SIGMA enter the detection
    law ``SNR(N) = s / sqrt(bias^2 + sigma^2 * N0/N)`` in quadrature; an
    arithmetic mean of per-replicate RMSEs is a smaller and different statistic
    that understates the error whenever the calibration draw matters.
    """
    return float(np.sqrt(np.mean(np.mean(np.asarray(e, float) ** 2, axis=0))))


@dataclass(frozen=True)
class ErrorStats:
    bias: float      # coherent, R-locked error: never averages down
    sigma: float     # the estimator's output noise at this recording length
    topo_r: float    # topography correlation of the noise-free recovery vs truth


def error_stats(rec_bias: np.ndarray, rec_noisy: np.ndarray,
                truth: np.ndarray) -> ErrorStats:
    """Split an arm's recovery error into coherent bias and output noise.

    ``rec_bias`` and ``rec_noisy`` are ``(n_chan, n_rep)`` recoveries at the HEF
    latency computed with IDENTICAL weights, so their difference is exactly the
    estimator's linear noise propagation -- no assumption about noise gain is
    needed, which is what makes the beats-to-detection calculation well posed.
    """
    b = np.asarray(rec_bias, float)
    n = np.asarray(rec_noisy, float)
    t = np.asarray(truth, float).reshape(-1, 1)
    e_bias = b - t
    e_noise = n - b
    bc = b - b.mean(0, keepdims=True)
    tc = t - t.mean()
    denom = np.linalg.norm(bc, axis=0) * np.linalg.norm(tc)
    r = np.where(denom > 0, (bc * tc).sum(0) / np.maximum(denom, 1e-300), 0.0)
    return ErrorStats(bias=_rms_over_replicates(e_bias),
                      sigma=_rms_over_replicates(e_noise),
                      topo_r=float(np.mean(r)))


@dataclass(frozen=True)
class TwoWayAnova:
    ss_a: float
    ss_b: float
    ss_ab: float
    ss_err: float
    df_err: int
    f_a: float
    f_b: float
    f_ab: float
    eta2_a: float
    eta2_b: float
    eta2_ab: float
    cellmean: np.ndarray          # rows = A (0, 1), cols = B (0, 1)


def two_way_anova(a: np.ndarray, b: np.ndarray, y: np.ndarray,
                  n: int) -> TwoWayAnova:
    """Balanced 2x2 variance partition over two binary factors.

    **Read eta^2, not F.** The replicates are Monte-Carlo calibration draws, not
    independent subjects, so ``df_err = 4(n-1)`` is a simulation knob: every F
    scales roughly linearly with n and none is an inferential test about the
    world. F is returned only as a which-sum-of-squares-dominates cue; partial
    eta^2 is the effect size to report.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    y = np.asarray(y, float).ravel()
    grand = y.mean()
    m_a = np.array([y[a == 0].mean(), y[a == 1].mean()])
    m_b = np.array([y[b == 0].mean(), y[b == 1].mean()])
    ss_a = n * 2 * float(((m_a - grand) ** 2).sum())
    ss_b = n * 2 * float(((m_b - grand) ** 2).sum())
    ss_ab = ss_err = 0.0
    cell = np.zeros((2, 2))
    for ia in (0, 1):
        for ib in (0, 1):
            yc = y[(a == ia) & (b == ib)]
            cell[ia, ib] = yc.mean()
            ss_ab += n * (cell[ia, ib] - m_a[ia] - m_b[ib] + grand) ** 2
            ss_err += float(((yc - yc.mean()) ** 2).sum())
    df_err = 4 * (n - 1)
    ms_err = ss_err / df_err if df_err else np.inf

    def eta(ss: float) -> float:
        return ss / (ss + ss_err) if (ss + ss_err) > 0 else 0.0

    return TwoWayAnova(
        ss_a=ss_a, ss_b=ss_b, ss_ab=ss_ab, ss_err=ss_err, df_err=df_err,
        f_a=ss_a / ms_err, f_b=ss_b / ms_err, f_ab=ss_ab / ms_err,
        eta2_a=eta(ss_a), eta2_b=eta(ss_b), eta2_ab=eta(ss_ab), cellmean=cell)
