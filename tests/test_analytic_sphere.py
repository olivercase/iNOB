"""Analytic spherical-shell forward solutions (Sarvas + Berg-Scherg).

These are the gold-standard reference benchmarks that any FEM forward
solver should match to within ~5% RMS for sensors well outside the source.
We test the analytic formulas in isolation here (the FEM-vs-analytic test
lives behind the duneuro marker in test_analytic_validation.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from inob.analysis.analytic_sphere import (
    MU_0,
    homogeneous_sphere_eeg_potential,
    infinite_medium_meg_field,
    sarvas_meg_field,
)


def test_biot_savart_closed_form_value() -> None:
    """B = μ₀/4π · Q × r̂ / r² for a dipole in an unbounded medium."""
    r0 = np.zeros(3)
    Q = np.array([1e-9, 0.0, 0.0])  # 1 nA·m along +x
    sensors = np.array([[0.0, 0.05, 0.0]])  # 5 cm along +y
    B = infinite_medium_meg_field(r0, Q, sensors)
    # Q × r̂ = x̂ × ŷ = ẑ, so B is +z with magnitude μ₀ Q / (4π r²)
    expected = MU_0 * 1e-9 / (4.0 * np.pi * 0.05**2)
    np.testing.assert_allclose(B[0], [0.0, 0.0, expected], rtol=1e-12, atol=1e-20)


def test_biot_savart_inverse_square_falloff() -> None:
    r0 = np.zeros(3)
    Q = np.array([1e-9, 0.0, 0.0])
    near = infinite_medium_meg_field(r0, Q, np.array([[0.0, 0.05, 0.0]]))
    far = infinite_medium_meg_field(r0, Q, np.array([[0.0, 0.10, 0.0]]))
    ratio = np.linalg.norm(near) / np.linalg.norm(far)
    assert np.isclose(ratio, 4.0, rtol=1e-12)


def test_biot_savart_silent_along_dipole_axis() -> None:
    """Q × r̂ vanishes when the sensor lies along the dipole axis."""
    r0 = np.zeros(3)
    Q = np.array([0.0, 0.0, 1e-9])
    sensors = np.array([[0.0, 0.0, 0.08], [0.0, 0.0, -0.08]])
    B = infinite_medium_meg_field(r0, Q, sensors)
    np.testing.assert_allclose(B, 0.0, atol=1e-20)


def test_biot_savart_translation_invariant() -> None:
    """Unlike the sphere solutions, rung 1 depends only on the separation."""
    Q = np.array([1e-9, 2e-10, 0.0])
    r0 = np.array([0.01, -0.02, 0.03])
    sensors = np.array([[0.06, 0.01, 0.02], [-0.04, 0.05, 0.0]])
    shift = np.array([0.13, -0.07, 0.21])
    B_a = infinite_medium_meg_field(r0, Q, sensors)
    B_b = infinite_medium_meg_field(r0 + shift, Q, sensors + shift)
    np.testing.assert_allclose(B_a, B_b, rtol=1e-12, atol=1e-20)


def test_biot_savart_perpendicular_to_moment_and_separation() -> None:
    Q = np.array([1e-9, 3e-10, -2e-10])
    r0 = np.array([0.0, 0.0, 0.01])
    sensors = np.array([[0.05, 0.02, 0.03], [-0.03, 0.04, -0.01]])
    B = infinite_medium_meg_field(r0, Q, sensors)
    sep = sensors - r0[None, :]
    assert np.allclose(np.einsum("ij,j->i", B, Q), 0.0, atol=1e-24)
    assert np.allclose(np.einsum("ij,ij->i", B, sep), 0.0, atol=1e-24)


def test_biot_savart_radial_dipole_is_not_silent() -> None:
    """The key rung 1 vs rung 2 contrast: no sphere, so no silent sources.

    A dipole radial to the sphere centre gives exactly zero under Sarvas but
    a finite field under Biot-Savart, because the cancellation is a property
    of the spherical boundary rather than of the primary current.
    """
    r0 = np.array([0.0, 0.0, 0.05])
    Q_radial = r0 / np.linalg.norm(r0) * 1e-9
    sensors = np.array([[0.10, 0.0, 0.0], [0.08, 0.04, 0.05]])
    B_sphere = sarvas_meg_field(r0, Q_radial, sensors)
    B_free = infinite_medium_meg_field(r0, Q_radial, sensors)
    np.testing.assert_allclose(B_sphere, 0.0, atol=1e-15)
    assert (np.linalg.norm(B_free, axis=1) > 1e-16).all()


def test_sarvas_radial_dipole_zero_field() -> None:
    """A purely radial dipole produces zero magnetic field outside a sphere."""
    r0 = np.array([0.0, 0.0, 0.05])  # 5 cm radial offset
    Q_radial = r0 / np.linalg.norm(r0) * 1e-9
    sensors = np.array(
        [
            [0.10, 0.0, 0.0],
            [0.08, 0.04, 0.05],
            [0.0, 0.10, 0.10],
        ]
    )
    B = sarvas_meg_field(r0, Q_radial, sensors)
    np.testing.assert_allclose(B, 0.0, atol=1e-15)


def test_sarvas_tangential_dipole_finite_field() -> None:
    r0 = np.array([0.0, 0.0, 0.05])
    Q_tan = np.array([1e-9, 0.0, 0.0])  # tangential to r0
    sensors = np.array([[0.10, 0.0, 0.05], [0.08, 0.04, 0.05]])
    B = sarvas_meg_field(r0, Q_tan, sensors)
    assert np.isfinite(B).all()
    assert np.linalg.norm(B, axis=1).all()
    # Order-of-magnitude check: a 1 nA·m dipole at 5 cm radius → ~10–100 fT
    norms_fT = np.linalg.norm(B, axis=1) * 1e15
    assert (norms_fT > 0.1).all() and (norms_fT < 1e4).all()


def test_eeg_centred_dipole_dipolar_pattern() -> None:
    """A z-aligned dipole at the sphere centre produces a cosθ surface potential.

    Sensors on the equator (±x, ±y) are zero; sensors at ±z are antipodal
    extrema.
    """
    R = 0.1
    sensors = R * np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    V = homogeneous_sphere_eeg_potential(
        np.zeros(3),
        np.array([0.0, 0.0, 1e-9]),
        sensors,
        sphere_radius_m=R,
        sigma_S_per_m=0.33,
    )
    np.testing.assert_allclose(V[:2], 0.0, atol=1e-7)  # equator
    assert np.sign(V[2]) == -np.sign(V[3])  # poles antipodal
    np.testing.assert_allclose(abs(V[2]), abs(V[3]), rtol=1e-6)


def test_eeg_radial_dipole_axisymmetric_about_axis() -> None:
    """Radial dipole on +z: potential is axisymmetric about z."""
    R = 0.1
    r0 = np.array([0.0, 0.0, 0.05])
    Q = np.array([0.0, 0.0, 1e-9])  # radial w.r.t. r0
    sensors = R * np.array(
        [
            [np.cos(0), np.sin(0), 0.0],
            [np.cos(np.pi / 2), np.sin(np.pi / 2), 0.0],
            [np.cos(np.pi), np.sin(np.pi), 0.0],
        ]
    )
    V = homogeneous_sphere_eeg_potential(
        r0,
        Q,
        sensors,
        sphere_radius_m=R,
        sigma_S_per_m=0.33,
    )
    # All sensors at θ=π/2 → same potential by axisymmetry
    np.testing.assert_allclose(V, V[0], rtol=1e-9, atol=1e-15)


def test_eeg_tangential_dipole_dipolar_pattern() -> None:
    """Tangential dipole produces an antipodal sign change along its axis."""
    R = 0.1
    r0 = np.array([0.0, 0.0, 0.05])
    Q = np.array([1e-9, 0.0, 0.0])
    p_pos = np.array([R, 0.0, 0.0])
    p_neg = np.array([-R, 0.0, 0.0])
    V = homogeneous_sphere_eeg_potential(
        r0,
        Q,
        np.stack([p_pos, p_neg]),
        sphere_radius_m=R,
        sigma_S_per_m=0.33,
    )
    assert np.sign(V[0]) == -np.sign(V[1])
    np.testing.assert_allclose(abs(V[0]), abs(V[1]), rtol=1e-6)


@pytest.mark.parametrize("q_dir", [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.3, -0.5, 0.8]])
def test_eeg_centred_dipole_is_three_times_infinite_medium(q_dir) -> None:
    """A centred dipole in an insulated sphere reads 3 q·r̂ / (4π σ R²).

    Textbook (Malmivuo & Plonsey 1995, §8): the boundary triples the
    infinite-medium surface value, for every orientation. The earlier
    tangential coefficient read exactly half of this, which the symmetry
    tests above cannot see.
    """
    R, sigma = 0.1, 0.33
    rng = np.random.default_rng(0)
    rhat = rng.normal(size=(20, 3))
    rhat /= np.linalg.norm(rhat, axis=1)[:, None]
    # The series decomposes about r0, which is undefined at the centre.
    r0 = np.array([0.0, 0.0, 1e-9])
    q = 1e-9 * np.asarray(q_dir) / np.linalg.norm(q_dir)
    V = homogeneous_sphere_eeg_potential(r0, q, R * rhat, sphere_radius_m=R, sigma_S_per_m=sigma)
    expected = 3.0 * (rhat @ q) / (4.0 * np.pi * sigma * R**2)
    np.testing.assert_allclose(V, expected, rtol=1e-6, atol=1e-12 * np.abs(expected).max())


@pytest.mark.parametrize("q", [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.4, 0.7, -0.2]])
def test_eeg_eccentric_dipole_matches_numerical_neumann_solve(q) -> None:
    """Off-centre, the series matches an independent Neumann solve.

    The reference is built without any Legendre series: the infinite-medium
    dipole potential plus a harmonic correction Σ c_lm r^l Y_lm, with c fitted
    by least squares so that no current leaves the sphere. Surface potentials
    are compared after removing the mean, since only differences are defined.
    """
    from scipy.special import sph_harm_y

    R, sigma, L = 1.0, 1.0, 22
    r0 = np.array([0.1, -0.2, 0.3])  # eccentricity 0.37 → series tail < 1e-9
    q = np.asarray(q, dtype=np.float64)

    def harmonics(X):
        rr = np.linalg.norm(X, axis=1)
        th = np.arccos(np.clip(X[:, 2] / rr, -1.0, 1.0))
        ph = np.arctan2(X[:, 1], X[:, 0])
        B, dB = [], []
        for ell in range(1, L + 1):
            for m in range(-ell, ell + 1):
                Y = sph_harm_y(ell, abs(m), th, ph)
                Yr = Y.real if m >= 0 else Y.imag
                B.append(rr**ell * Yr)
                dB.append(ell * rr ** (ell - 1) * Yr)
        return np.array(B).T, np.array(dB).T

    rng = np.random.default_rng(1)
    fit = rng.normal(size=(3000, 3))
    fit /= np.linalg.norm(fit, axis=1)[:, None]
    d = fit - r0
    dn = np.linalg.norm(d, axis=1)
    grad_inf = (q[None, :] / dn[:, None] ** 3 - 3.0 * (d @ q)[:, None] * d / dn[:, None] ** 5) / (
        4.0 * np.pi * sigma
    )
    _, dB = harmonics(fit)
    c, *_ = np.linalg.lstsq(dB, -(grad_inf * fit).sum(axis=1), rcond=None)

    pts = rng.normal(size=(40, 3))
    pts /= np.linalg.norm(pts, axis=1)[:, None]
    B, _ = harmonics(pts)
    dp = pts - r0
    ref = (dp @ q) / (4.0 * np.pi * sigma * np.linalg.norm(dp, axis=1) ** 3) + B @ c
    V = homogeneous_sphere_eeg_potential(r0, q, pts, sphere_radius_m=R, sigma_S_per_m=sigma)
    ref -= ref.mean()
    V -= V.mean()
    np.testing.assert_allclose(V, ref, atol=1e-6 * np.abs(ref).max())


# ── MNE-Python cross-validation regression tests ──────────────────────────


def _mne_sphere_field_or_skip():
    """Return MNE's _do_sphere_field, skipping the test if unavailable."""
    # MNE moved the kernel into a numba module (1.7-ish); the old path is kept
    # for older installs. Without the new path this skipped on every current MNE.
    try:
        from mne.forward._compute_forward_numba import _do_sphere_field
    except ImportError:
        try:
            from mne.forward._compute_forward import _do_sphere_field
        except ImportError:  # pragma: no cover
            pytest.skip("mne-python not available")
    return _do_sphere_field


@pytest.mark.parametrize(
    "src_pos_m,Q_Am",
    [
        (np.array([0.0, 0.0, 0.05]), np.array([1e-9, 0.0, 0.0])),  # tangential
        (np.array([0.03, 0.02, 0.04]), np.array([0.0, 1e-9, 0.0])),
        (np.array([0.04, 0.0, 0.0]), np.array([0.0, 0.0, 1e-9])),  # longitudinal
        (np.array([-0.02, 0.03, 0.04]), np.array([2.0e-9, -1.0e-9, 0.5e-9])),  # mixed
    ],
)
def test_sarvas_matches_mne_to_machine_precision(src_pos_m, Q_Am) -> None:
    """Our Sarvas field must agree with MNE-Python's reference implementation.

    MNE's ``_do_sphere_field`` is the field of Sarvas (1987) as further
    optimised by Hämäläinen 1990 and used in MNE since the 1990s. Any drift
    between our analytic_sphere and MNE points to a bug in either our
    formula or our unit handling — this test catches both.
    """
    do_sphere_field = _mne_sphere_field_or_skip()
    r0_sphere = np.zeros(3, dtype=np.float64)
    sensors_m = np.array(
        [
            [0.10, 0.00, 0.05],
            [0.08, 0.04, 0.05],
            [0.0, 0.10, 0.10],
            [-0.07, 0.05, 0.06],
            [0.06, -0.07, 0.04],
        ]
    )
    n_sensors = sensors_m.shape[0]
    ws = np.ones(n_sensors)
    bins = np.arange(n_sensors, dtype=np.int64)

    Bs_mne = np.zeros((n_sensors, 3))
    for axis in range(3):
        cosmag = np.zeros((n_sensors, 3))
        cosmag[:, axis] = 1.0
        # _do_sphere_field returns (3 * n_src, n_coils), each row being
        # dB/dQ_i projected onto cosmag (so along this axis when cosmag = ê_axis).
        M = do_sphere_field(
            src_pos_m[None, :].astype(np.float64),
            sensors_m.astype(np.float64),
            cosmag.astype(np.float64),
            ws.astype(np.float64),
            bins,
            r0_sphere,
        )
        Bs_mne[:, axis] = M.T @ Q_Am

    Bs_ours = sarvas_meg_field(src_pos_m, Q_Am, sensors_m)
    np.testing.assert_allclose(Bs_ours, Bs_mne, rtol=1e-12, atol=1e-30)


@pytest.mark.parametrize("src_pos_m", [np.array([0.01, -0.02, 0.05]), np.array([0.0, 0.0, 0.03])])
def test_eeg_series_matches_mne_single_shell(src_pos_m) -> None:
    """Our homogeneous-sphere EEG must agree with MNE's sphere EEG.

    MNE evaluates the multi-shell series by the Berg approximation, which is
    exact to ~1e-5 when every shell has the same conductivity; the two are
    compared after common-average referencing.
    """
    try:
        import mne
        from mne.forward._compute_forward_numba import _do_eeg_spherepot_coil
    except ImportError:  # pragma: no cover
        pytest.skip("mne-python not available")
    R, sigma = 0.09, 0.33
    sphere = mne.make_sphere_model(
        r0=(0, 0, 0), head_radius=R, relative_radii=(0.5, 1.0), sigmas=(sigma, sigma), verbose=False
    )
    rng = np.random.default_rng(0)
    el = rng.normal(size=(30, 3))
    el *= R / np.linalg.norm(el, axis=1)[:, None]
    M = _do_eeg_spherepot_coil(
        src_pos_m[None, :],
        el,
        np.ones(30),
        np.arange(30),
        np.zeros(3),
        R,
        sphere["mu"],
        sphere["lambda"],
    )
    for k in range(3):
        ours = homogeneous_sphere_eeg_potential(
            src_pos_m, np.eye(3)[k], el, sphere_radius_m=R, sigma_S_per_m=sigma
        )
        np.testing.assert_allclose(
            ours - ours.mean(), M[k] - M[k].mean(), atol=1e-4 * np.abs(M[k]).max()
        )
