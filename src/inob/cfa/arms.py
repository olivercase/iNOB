"""The correction arms, all scored on identical data with identical weights.

Port of the local functions in ``CFA_Simulation_2x2_Engine.m``.

Every arm takes the clean and noisy beat blocks and returns the recovered HEF
field for **both**, computed with the SAME weights. That is what makes
``noisy - clean`` exactly the estimator's linear noise propagation, and it is
what makes the beats-to-detection calculation well posed: the coherent bias is a
floor that never averages down, the noise falls as ``1/sqrt(beats)``, and the two
are measured separately rather than assumed.

Data layout throughout: ``(n_or, n_chan, n_t)`` -- orientation-major, so stacking
orientations is a reshape rather than a transpose. (The MATLAB original is
``(ns, nT, Nor)`` and needs a permute before every stack; getting that permute
wrong silently scrambles the CDM training design, which is called out in its
source.)

The arms:

``none``       uncorrected average of every recorded beat. In the movement cells
               that average already cancels some CFA on its own, because the
               artefact topography differs by orientation -- a strictly fairer
               "do nothing" than picking one block.
``glm``        source-space joint GLM: solve ``Y_n = Phi a + L_n q`` for the
               shared insula amplitude and the cardiac moment across all
               orientations at once.
``hfc_lcmv``   harmonic field correction at a given order, then a 3-dof vector
               LCMV beamformer at the insula.
``cdm``        cardiac dipole moment method (Woelk et al.): per-channel
               pseudo-inverse trained on held-out rotations. Needs >= 2 distinct
               orientations, so it is undefined wherever movement is low -- itself
               a structural result.
"""
from __future__ import annotations

import numpy as np

__all__ = ["arm_none", "arm_glm", "arm_hfc_lcmv", "arm_ssp", "arm_ac", "arm_cdm"]


def _stack(y: np.ndarray) -> np.ndarray:
    """``(n_or, n_chan, n_t)`` -> ``(n_or * n_chan, n_t)``."""
    n_or, n_chan, n_t = y.shape
    return y.reshape(n_or * n_chan, n_t)


def arm_none(yc: np.ndarray, yn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The naive baseline an experimenter would actually produce."""
    return yc.mean(0), yn.mean(0)


def arm_glm(yc: np.ndarray, yn: np.ndarray, design: np.ndarray,
            phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Source-space joint GLM across all orientations.

    ``design`` is ``(n_or * n_chan, 3 + n_cardiac_dof)``: the insula basis
    repeated down the orientation blocks, beside the per-orientation cardiac
    leadfields. Columns are scaled to unit norm before the solve and unscaled
    after -- the insula and cardiac columns differ by orders of magnitude, and
    leaving that to the least-squares solver costs conditioning for nothing.
    """
    scale = np.linalg.norm(design, axis=0)
    scale[scale == 0] = 1.0
    a = design / scale
    cb = np.linalg.lstsq(a, _stack(yc), rcond=None)[0] / scale[:, None]
    cn = np.linalg.lstsq(a, _stack(yn), rcond=None)[0] / scale[:, None]
    return phi @ cb[:3], phi @ cn[:3]


def arm_hfc_lcmv(yc: np.ndarray, yn: np.ndarray, proj: np.ndarray,
                 phi: np.ndarray, *, lambda_reg: float,
                 noise_var: float) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic field correction, then a 3-dof vector LCMV beamformer.

    Two details that are easy to get wrong and change the answer:

    * **The beamformer's forward model is the leadfield seen THROUGH the
      projector**, ``P @ Phi``, not ``Phi``. HFC is a spatial projection applied
      to the data, so afterwards the insula source produces ``P Phi`` at the
      sensors. Passing the unprojected ``Phi`` would impose the unit-gain
      constraint on a subspace the data cannot occupy. The amplitude is estimated
      through ``P Phi`` and mapped back through ``Phi`` to compare against truth.
    * **The covariance's noise term is** ``noise_var * P``, **not**
      ``noise_var * I``. What reaches the beamformer is ``P n``, with covariance
      ``P (noise_var I) P' = noise_var P`` for the symmetric idempotent P. Using
      ``I`` would restore variance to the subspace the projector just emptied.
      The shrinkage below returns the matrix to full rank, so the rank deficiency
      of ``noise_var * P`` is harmless.

    The covariance uses the EXPECTED noise term rather than one realisation, so
    the weights are deterministic given the calibration draw and
    ``recn - recb`` is exactly the linear noise propagation.
    """
    zc = np.einsum("ij,ojt->oit", proj, yc)
    zn = np.einsum("ij,ojt->oit", proj, yn)
    n_chan = yc.shape[1]
    flat = zc.transpose(1, 0, 2).reshape(n_chan, -1)
    cov = np.cov(flat) + noise_var * proj
    cov = cov + lambda_reg * np.trace(cov) / n_chan * np.eye(n_chan)
    cinv = np.linalg.pinv(cov)
    phi_p = proj @ phi
    w = np.linalg.solve(phi_p.T @ cinv @ phi_p, phi_p.T @ cinv)      # (3, n_chan)
    return (phi @ np.einsum("ij,ojt->it", w, zc) / len(zc),
            phi @ np.einsum("ij,ojt->it", w, zn) / len(zn))


def arm_ssp(yc: np.ndarray, yn: np.ndarray, ccfa_ref: np.ndarray, phi: np.ndarray,
           *, rank: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Empirical signal-space projection: per-orientation, no cardiac model.

    Unlike ``arm_glm`` (parametric 6-dof cardiac model, shared moment across all
    orientations, jointly fit) or ``arm_hfc_lcmv`` (a rotation-invariant harmonic
    subspace, covariance-optimal beamformer), this arm uses NO forward model of
    the heart at all. It takes ``ccfa_ref`` -- an R-locked, effectively
    noise-free average CFA topography-over-time for THIS orientation block only
    (what an experimenter gets by averaging on the cardiac clock instead of the
    stimulus clock) -- and keeps the top ``rank`` left singular vectors of that
    ``(n_chan, n_t)`` matrix as an empirical CFA subspace. Data are projected onto
    its null space and the insula moment is read off by plain least squares
    (``pinv``, not LCMV's covariance weighting).

    It is done independently per orientation block, then averaged -- there is no
    joint fit across orientations, so any benefit from movement can only come
    from bias partially cancelling across orientations on average (the same
    mechanism that lets ``arm_none`` partly self-correct), not from a shared
    model becoming better identified.
    """
    n_or, n_chan, n_t = yc.shape
    rb = np.zeros((n_chan, n_t))
    rn = np.zeros((n_chan, n_t))
    for j in range(n_or):
        u, _, _ = np.linalg.svd(ccfa_ref[j], full_matrices=False)
        q = u[:, :rank]
        p = np.eye(n_chan) - q @ q.T
        phip = p @ phi
        w = np.linalg.pinv(phip)                  # (3, n_chan), plain OLS
        rb += phi @ (w @ (p @ yc[j]))
        rn += phi @ (w @ (p @ yn[j]))
    return rb / n_or, rn / n_or


def arm_cdm(yc: np.ndarray, yn: np.ndarray,
            lmod: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Woelk cardiac dipole moment: per-channel pinv trained on held-out rotations
    (or, with a still head, on the same orientation).

    It applies no source-space constraint and does not model the brain source at
    all, which is why it cannot separate a signal that contaminates every
    training beat.

    Leave-one-orientation-out, averaged over all orientations, so CDM sees the
    same total data as every other arm. Correcting a single held-out block would
    leave it carrying full single-block noise while the others average across all
    beats, making the comparison one of data quantity rather than method.
    """
    n_or, n_chan, n_t = yc.shape
    if n_or == 1:
        # Still head: train and test at the same orientation, as Woelk et al.
        # (2026) did with split beats. The training beats carry the HEF too.
        pinv = np.linalg.pinv(lmod[0])
        return (yc[0] - lmod[0] @ (pinv @ yc[0]),
                yn[0] - lmod[0] @ (pinv @ yn[0]))
    rb = np.zeros((n_chan, n_t))
    rn = np.zeros((n_chan, n_t))
    for test in range(n_or):
        train = [i for i in range(n_or) if i != test]
        l_train = np.concatenate([lmod[i] for i in train], axis=0)
        pinv = np.linalg.pinv(l_train)
        rb += yc[test] - lmod[test] @ (pinv @ _stack(yc[train]))
        rn += yn[test] - lmod[test] @ (pinv @ _stack(yn[train]))
    return rb / n_or, rn / n_or


def arm_ac(yc: np.ndarray, yn: np.ndarray, lmod: list[np.ndarray],
           phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lock-in on head rotation: fit the cardiac moment where the HEP cannot be.

    THE ASYMMETRY THIS EXPLOITS. Array, skull and brain are one rigid body, so
    the HEP leadfield is IDENTICAL in every orientation block; the heart is
    torso-fixed, so its leadfield ``L_n`` rotates. Split the orientation blocks
    into their mean (DC) and their variation about it (AC):

        DC  =  mean_n Y_n   contains the HEP in full, plus mean cardiac
        AC  =  Y_n - DC     contains ZERO HEP -- exactly, not approximately

    So AC is a guaranteed signal-free observation of the artefact alone. The
    cardiac moment ``q(t)`` is estimated there, where no amount of overfitting
    can steal HEP, and its DC contribution ``Lbar q`` is then subtracted from the
    block mean. Contrast the joint GLM, which fits ``q`` against data containing
    the HEP and therefore trades one against the other -- that trade is what
    makes the GLM's bias blow up when the array is poorly characterised.

    ``q`` is fitted on data measured through the REAL sensors, so the gain and
    tilt errors that define the precision factor are largely absorbed into the
    fitted amplitude rather than left as bias.

    Requires >= 2 distinct orientations and, unlike :func:`arm_ssp`, is
    STRUCTURALLY undefined for a still head -- with one block there is no AC.
    """
    n_or, n_chan, n_t = yc.shape
    if n_or < 2:
        raise ValueError("the AC lock-in needs at least 2 distinct orientations")
    lbar = np.mean(lmod, axis=0)
    dmat = np.concatenate([l - lbar for l in lmod], axis=0)      # (n_or*n_chan, 6)
    scale = np.linalg.norm(dmat, axis=0)
    scale[scale == 0] = 1.0
    w_phi = np.linalg.pinv(phi)

    def solve(y):
        dc = y.mean(0)
        ac = _stack(y - dc)
        q = np.linalg.lstsq(dmat / scale, ac, rcond=None)[0] / scale[:, None]
        return phi @ (w_phi @ (dc - lbar @ q))

    return solve(yc), solve(yn)
