"""Inverse reconstruction of the cardiac source from a chest array.

The question this answers is a design question, not a data-analysis one: **given a
chest OPM array, how much of the cardiac activation sequence can you actually get
back, and what array do you need to get more?**

Two estimators, because they fail differently and the difference is the result:

``equivalent_dipole``
    Fit a single moving current dipole per time sample -- position and moment
    both free. This is what clinical MCG reports, and it is well posed even for a
    handful of sensors: 6 unknowns against 24 channels. What it cannot do is
    represent a wavefront; when the true source is an extended, moving sheet, the
    fitted dipole tracks its *centre of gravity*, which is a real trajectory but
    not the activation sequence.

``distributed``
    Minimum-norm current density over the myocardial wall, with the orientations
    fixed to the anatomy. Depth-weighted, because an unweighted minimum-norm
    solution over a source space that spans 6 cm of depth puts everything on the
    surface nearest the sensors -- an artefact of the norm, not of the heart.

Both are scored against the ground truth the forward model was built from, so the
error is a reconstruction error and not a residual: a residual can be driven to
zero by any sufficiently flexible model while the source estimate is wrong.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DipoleFit", "fit_moving_dipole", "minimum_norm", "localisation_error",
           "resolution_matrix", "point_spread"]


# ---------------------------------------------------------------------------
# moving equivalent dipole
# ---------------------------------------------------------------------------
@dataclass
class DipoleFit:
    pos_mm: np.ndarray        # (T, 3)
    moment: np.ndarray        # (T, 3) A.m
    residual: np.ndarray      # (T,) fraction of field variance unexplained
    gof: np.ndarray           # (T,) goodness of fit, 1 - residual^2


def fit_moving_dipole(
    data: np.ndarray, grid_mm: np.ndarray, leadfield: np.ndarray,
    *, min_amplitude: float = 0.0,
) -> DipoleFit:
    """Scan a grid for the best single dipole at each time sample.

    ``leadfield`` is ``(n_chan, 3 * n_grid)`` in source-major x/y/z order. The scan
    is exhaustive rather than gradient-based: the cost surface for a dipole in a
    real conductor is not convex, and a local optimiser started from the previous
    sample will happily follow a wrong minimum through the whole beat.

    Samples whose field falls below ``min_amplitude`` get NaN rather than a
    fitted position, because a dipole fitted to noise is a random point in the
    chest and plotting it as a trajectory is how spurious "movement" appears in
    MCG papers.
    """
    y = np.asarray(data, float)
    n_grid = len(grid_mm)
    lf = np.asarray(leadfield, float).reshape(len(y), n_grid, 3)
    # per-grid-point orthonormal basis for its 3-dof leadfield
    proj = np.empty((n_grid, len(y), 3))
    for g in range(n_grid):
        # lf is (n_chan, n_grid, 3): take this grid point's 3-column leadfield
        proj[g] = np.linalg.qr(lf[:, g, :])[0]
    amp = np.linalg.norm(y, axis=0)
    pos = np.full((y.shape[1], 3), np.nan)
    mom = np.full((y.shape[1], 3), np.nan)
    res = np.full(y.shape[1], np.nan)
    for t in range(y.shape[1]):
        if amp[t] <= min_amplitude:
            continue
        # explained variance at every grid point at once
        expl = np.einsum("gcd,c->gd", proj, y[:, t])
        e2 = (expl ** 2).sum(1)
        g = int(e2.argmax())
        pos[t] = grid_mm[g]
        mom[t] = np.linalg.lstsq(lf[:, g, :], y[:, t], rcond=None)[0]
        res[t] = np.sqrt(max(1.0 - e2[g] / max(amp[t] ** 2, 1e-30), 0.0))
    return DipoleFit(pos_mm=pos, moment=mom, residual=res, gof=1.0 - res ** 2)


# ---------------------------------------------------------------------------
# distributed minimum norm
# ---------------------------------------------------------------------------
def minimum_norm(data: np.ndarray, leadfield: np.ndarray, *,
                 snr: float = 3.0, depth_weight: float = 0.8) -> np.ndarray:
    """Depth-weighted minimum-norm current density. Returns ``(n_source, n_t)``.

    Regularisation is set from an assumed SNR rather than tuned to the answer:
    ``lambda2 = trace(L L') / (n_chan * snr^2)``, the standard MNE convention. A
    lambda chosen by looking at the reconstruction is a free parameter fitted to
    the truth, which is exactly what a sensitivity study must not do.
    """
    l = np.asarray(leadfield, float)
    w = np.linalg.norm(l, axis=0) ** depth_weight
    w[w == 0] = 1.0
    lw = l / w
    lam2 = np.trace(lw @ lw.T) / (l.shape[0] * snr ** 2)
    g = lw.T @ np.linalg.solve(lw @ lw.T + lam2 * np.eye(l.shape[0]),
                               np.asarray(data, float))
    return g / w[:, None]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def localisation_error(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Distance between an estimated and a true centre of gravity, per sample."""
    return np.linalg.norm(np.asarray(est) - np.asarray(truth), axis=-1)


def resolution_matrix(leadfield: np.ndarray, *, snr: float = 3.0,
                      depth_weight: float = 0.8) -> np.ndarray:
    """``R = G L``: what the array would recover from a unit source at each site.

    The resolution matrix is the array's answer to "what can this geometry see",
    independent of any particular beat. Its columns are point-spread functions and
    its rows cross-talk functions; a perfect array would give the identity.
    """
    l = np.asarray(leadfield, float)
    w = np.linalg.norm(l, axis=0) ** depth_weight
    w[w == 0] = 1.0
    lw = l / w
    lam2 = np.trace(lw @ lw.T) / (l.shape[0] * snr ** 2)
    g = lw.T @ np.linalg.solve(lw @ lw.T + lam2 * np.eye(l.shape[0]), np.eye(l.shape[0]))
    return (g / w[:, None]) @ l


def point_spread(res: np.ndarray, pos_mm: np.ndarray) -> dict:
    """Point-spread width and localisation bias, from a resolution matrix."""
    r2 = np.asarray(res, float) ** 2
    r2 = r2 / np.maximum(r2.sum(0, keepdims=True), 1e-30)
    cog = r2.T @ pos_mm                                   # (n_src, 3)
    bias = np.linalg.norm(cog - pos_mm, axis=1)
    d = np.linalg.norm(pos_mm[:, None, :] - pos_mm[None, :, :], axis=-1)
    spread = np.sqrt((r2 * d ** 2).sum(0))
    return {"bias_mm": bias, "spread_mm": spread,
            "mean_bias_mm": float(bias.mean()), "mean_spread_mm": float(spread.mean())}
