"""Sensor-calibration error: the precision factor.

Port of ``cfa_precision_grad.m``, generalised to a triaxial array.

Two independent error channels turn the NOMINAL array into the array as it
physically is. They are not redundant:

* **gain** is a diagonal per-channel rescale. ``r_i (o_i . B) = (r_i o_i) . B`` --
  it can only rescale what a channel already sees.
* **axis tilt** rotates ``o_i``, mixing in field components the forward model
  treats that channel as blind to. No diagonal rescale reproduces that, and it is
  tilt -- not gain -- that caps the shielding factor of harmonic interference
  models (Taulu & Simola 2006; Tierney et al. 2024, SF ~ 1/(kappa*g)).

**Usage contract:** data are generated with the perturbed axes and gains; every
analysis leadfield (joint-GLM design, harmonic regressor, beamformer forward
model) is built from the NOMINAL array. That mismatch *is* the precision factor.

Positions are not perturbed: OPM position is fixed by the rigid scanner-cast
geometry and is known far better than the sensitive-axis direction, which depends
on each sensor's internal vapour-cell and beam alignment.

On a triaxial array the three axes at a site are perturbed **independently**.
They are separate readout channels with separate cells and separate beam paths;
a shared site does not make their calibration errors common-mode.
"""
from __future__ import annotations

import numpy as np

__all__ = ["perturb_array"]


def _tilt(ori: np.ndarray, sd_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate each unit row by a Gaussian-magnitude angle about a random perpendicular.

    Done exactly in the ``(o, w)`` plane -- that *is* the rotation, so there is no
    small-angle approximation and the result stays exactly unit norm.
    """
    o = np.asarray(ori, float)
    n = len(o)
    alpha = np.deg2rad(sd_deg) * rng.standard_normal((n, 1))
    w = rng.standard_normal((n, 3))
    w -= np.einsum("ij,ij->i", w, o)[:, None] * o          # component along o removed
    bad = np.linalg.norm(w, axis=1) < 1e-12
    if bad.any():                                          # vanishingly rare
        alt = np.where(np.abs(o[bad, 0:1]) > 0.9,
                       np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        w[bad] = np.cross(o[bad], alt)
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    out = np.cos(alpha) * o + np.sin(alpha) * w
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def perturb_array(
    chanori: np.ndarray, gain_sd: float, tilt_sd_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(perturbed_chanori, gain)`` for one calibration draw.

    Zero SDs return the nominal axes and unit gains, i.e. perfect knowledge.
    """
    o = np.asarray(chanori, float)
    gain = 1.0 + gain_sd * rng.standard_normal(len(o)) if gain_sd > 0 else np.ones(len(o))
    ori = _tilt(o, tilt_sd_deg, rng) if tilt_sd_deg > 0 else o.copy()
    return ori, gain
