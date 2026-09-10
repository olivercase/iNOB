"""Analytic forward solutions for spherical conductors.

Used as a reference benchmark in :mod:`tests.test_analytic_validation`.

References
----------
* Sarvas, J. (1987). Basic mathematical and electromagnetic concepts of the
  biomagnetic inverse problem. Phys Med Biol, 32(1), 11-22.
* Geselowitz, D. B. (1970). On the magnetic field generated outside an
  inhomogeneous volume conductor by internal current sources. IEEE Trans
  Magn, 6(2), 346-347.

The MEG forward in a *homogeneous* sphere is independent of the sphere's
conductivity (Geselowitz / Sarvas) — only radial source positions inside
the sphere and the field-point geometry matter. The EEG forward in a
homogeneous sphere is the Berg-Scherg / Wolters analytic series.

Forward-model ladder
--------------------
Two of the three MEG rungs used by :mod:`inob.analysis.sarvas_compare` live
here, in order of increasing volume-conductor realism:

  1. :func:`infinite_medium_meg_field` — free-space Biot–Savart. No boundary
     at all; primary current only.
  2. :func:`sarvas_meg_field` — homogeneous sphere. Adds the secondary
     (volume-current) field from a spherical boundary.
  3. FEM (DUNEuro, :mod:`inob.forward`) — realistic multi-tissue geometry.

Differencing consecutive rungs isolates one physical effect each: 1→2 is the
volume-current contribution, 2→3 is the effect of real geometry.
"""

from __future__ import annotations

import numpy as np

MU_0 = 4.0 * np.pi * 1e-7  # vacuum permeability, T·m/A


def infinite_medium_meg_field(
    dipole_pos_m: np.ndarray,
    dipole_moment_Am: np.ndarray,
    sensor_pos_m: np.ndarray,
) -> np.ndarray:
    """Magnetic field of a current dipole in an unbounded homogeneous medium.

    The free-space Biot–Savart law for a point current dipole::

        B(r) = (μ₀ / 4π) · Q × (r − r₀) / |r − r₀|³

    In an *infinite* homogeneous conductor the volume currents contribute
    nothing to the magnetic field, so this is simultaneously the vacuum
    result and the infinite-medium result: the forward field with no
    volume-conductor boundary anywhere. Differencing it against
    :func:`sarvas_meg_field` isolates the secondary (volume-current) field
    introduced by the spherical boundary.

    Unlike the sphere solutions this is translation invariant — only the
    source-to-sensor separation enters, so no centre need be chosen.

    Inputs/outputs are in **SI units** (m, A·m, T).

    Parameters
    ----------
    dipole_pos_m     : (3,) source position
    dipole_moment_Am : (3,) dipole moment vector
    sensor_pos_m     : (N, 3) sensor positions

    Returns
    -------
    (N, 3) magnetic-field vectors at each sensor in Tesla.
    """
    r0 = np.asarray(dipole_pos_m, dtype=np.float64)
    Q = np.asarray(dipole_moment_Am, dtype=np.float64)
    r = np.asarray(sensor_pos_m, dtype=np.float64)
    if r.ndim == 1:
        r = r[None, :]

    a = r - r0[None, :]
    a_norm = np.maximum(np.linalg.norm(a, axis=1), 1e-30)
    return (MU_0 / (4.0 * np.pi)) * np.cross(Q[None, :], a) / (a_norm**3)[:, None]


def sarvas_meg_field(
    dipole_pos_m: np.ndarray,
    dipole_moment_Am: np.ndarray,
    sensor_pos_m: np.ndarray,
) -> np.ndarray:
    """Magnetic field outside a homogeneous sphere from a current dipole.

    Sarvas (1987) closed-form. Inputs/outputs are in **SI units** (m, A·m, T).

    Parameters
    ----------
    dipole_pos_m   : (3,) source position, sphere centred at origin
    dipole_moment_Am : (3,) dipole moment vector
    sensor_pos_m   : (N, 3) sensor positions (must be outside the sphere)

    Returns
    -------
    (N, 3) magnetic-field vectors at each sensor in Tesla.
    """
    r0 = np.asarray(dipole_pos_m, dtype=np.float64)
    Q = np.asarray(dipole_moment_Am, dtype=np.float64)
    r = np.asarray(sensor_pos_m, dtype=np.float64)
    if r.ndim == 1:
        r = r[None, :]

    a = r - r0[None, :]
    a_norm = np.linalg.norm(a, axis=1)
    r_norm = np.linalg.norm(r, axis=1)
    # F = a * (r * a + r² - r·r0)
    F = a_norm * (r_norm * a_norm + r_norm * r_norm - (r * r0[None, :]).sum(axis=1))
    # ∇F = (a²/r + (a·r)/a + 2*a + 2*r) * r - (a + 2*r + (a·r)/a) * r0
    a_dot_r = (a * r).sum(axis=1)
    grad_F_r_coef = (a_norm**2 / r_norm + a_dot_r / a_norm + 2.0 * a_norm + 2.0 * r_norm)[:, None]
    grad_F_r0_coef = (a_norm + 2.0 * r_norm + a_dot_r / a_norm)[:, None]
    grad_F = grad_F_r_coef * r - grad_F_r0_coef * r0[None, :]

    Q_cross_r0 = np.cross(Q, r0)  # (3,)
    Q_cross_r0_dot_r = (r * Q_cross_r0[None, :]).sum(axis=1)[:, None]  # (N, 1)

    B = (MU_0 / (4.0 * np.pi * F[:, None] ** 2)) * (
        F[:, None] * np.tile(Q_cross_r0[None, :], (len(r), 1)) - Q_cross_r0_dot_r * grad_F
    )
    return B


def homogeneous_sphere_eeg_potential(
    dipole_pos_m: np.ndarray,
    dipole_moment_Am: np.ndarray,
    sensor_pos_m: np.ndarray,
    *,
    sphere_radius_m: float,
    sigma_S_per_m: float,
    n_terms: int = 60,
) -> np.ndarray:
    """Surface potential at points on a homogeneous sphere.

    Berg-Scherg series for a single-sphere head model.
    Inputs SI; output volts.
    """
    r0 = np.asarray(dipole_pos_m, dtype=np.float64)
    Q = np.asarray(dipole_moment_Am, dtype=np.float64)
    r = np.asarray(sensor_pos_m, dtype=np.float64)
    if r.ndim == 1:
        r = r[None, :]

    R = float(sphere_radius_m)
    sigma = float(sigma_S_per_m)
    f = float(np.linalg.norm(r0))  # source eccentricity
    if f >= R:
        raise ValueError("dipole must be inside the sphere")

    # Decompose Q into radial and tangential components w.r.t. r0
    if f > 0:
        e_r0 = r0 / f
    else:
        e_r0 = np.array([0.0, 0.0, 1.0])
    Q_rad = Q @ e_r0
    Q_tan_vec = Q - Q_rad * e_r0
    Q_tan = float(np.linalg.norm(Q_tan_vec))
    if Q_tan > 0:
        e_tan = Q_tan_vec / Q_tan
    else:
        e_tan = np.array([1.0, 0.0, 0.0])
        # ensure orthogonality to e_r0
        e_tan = e_tan - (e_tan @ e_r0) * e_r0
        nrm = np.linalg.norm(e_tan)
        e_tan = e_tan / max(nrm, 1e-12)

    # Spherical coordinates (θ, φ) of each sensor in the (e_r0, e_tan) frame
    rs = np.linalg.norm(r, axis=1)
    cos_theta = (r @ e_r0) / np.maximum(rs, 1e-30)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    # φ measured from e_tan in the plane orthogonal to e_r0
    proj = r - rs[:, None] * cos_theta[:, None] * e_r0[None, :]
    proj_norm = np.linalg.norm(proj, axis=1)
    proj_norm = np.maximum(proj_norm, 1e-30)
    cos_phi = (proj @ e_tan) / proj_norm
    # Legendre P_n and dP_n/dx at x = cos_theta, computed via recursion.
    n = np.arange(1, n_terms + 1)
    # Series coefficient for radial dipole (P_n) and tangential (P_n^1):
    # V_rad = (1/(4π σ R²)) Σ ((2n+1)/n) (f/R)^(n-1) P_n(cos θ)
    # V_tan = (1/(4π σ R²)) Σ ((2n+1)/(n(n+1))) (f/R)^(n-1) P_n^1(cos θ) cos φ
    f_over_R = f / R
    coef_rad = (2 * n + 1) / n * f_over_R ** (n - 1)
    coef_tan = (2 * n + 1) / (n * (n + 1)) * f_over_R ** (n - 1)

    out = np.zeros(len(r), dtype=np.float64)
    # P_{n-1}, P_n via recurrence
    P_prev = np.ones_like(cos_theta)  # P_0
    P_cur = cos_theta.copy()  # P_1
    P1_prev = np.zeros_like(cos_theta)  # P_0^1 = 0
    P1_cur = sin_theta.copy()  # P_1^1 = sinθ (using positive convention)
    for k in range(1, n_terms + 1):
        out += Q_rad * coef_rad[k - 1] * P_cur + Q_tan * coef_tan[k - 1] * P1_cur * cos_phi
        # advance: P_{k+1} = ((2k+1) x P_k − k P_{k-1}) / (k+1)
        P_next = ((2 * k + 1) * cos_theta * P_cur - k * P_prev) / (k + 1)
        P1_next = ((2 * k + 1) * cos_theta * P1_cur - (k + 1) * P1_prev) / k
        P_prev, P_cur = P_cur, P_next
        P1_prev, P1_cur = P1_cur, P1_next

    return out / (4.0 * np.pi * sigma * R * R)
