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
    Q = np.array([1e-9, 0.0, 0.0])             # 1 nA·m along +x
    sensors = np.array([[0.0, 0.05, 0.0]])     # 5 cm along +y
    B = infinite_medium_meg_field(r0, Q, sensors)
    # Q × r̂ = x̂ × ŷ = ẑ, so B is +z with magnitude μ₀ Q / (4π r²)
    expected = MU_0 * 1e-9 / (4.0 * np.pi * 0.05 ** 2)
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
    r0 = np.array([0.0, 0.0, 0.05])           # 5 cm radial offset
    Q_radial = r0 / np.linalg.norm(r0) * 1e-9
    sensors = np.array([
        [0.10, 0.0, 0.0],
        [0.08, 0.04, 0.05],
        [0.0, 0.10, 0.10],
    ])
    B = sarvas_meg_field(r0, Q_radial, sensors)
    np.testing.assert_allclose(B, 0.0, atol=1e-15)


def test_sarvas_tangential_dipole_finite_field() -> None:
    r0 = np.array([0.0, 0.0, 0.05])
    Q_tan = np.array([1e-9, 0.0, 0.0])         # tangential to r0
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
    sensors = R * np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    V = homogeneous_sphere_eeg_potential(
        np.zeros(3), np.array([0.0, 0.0, 1e-9]), sensors,
        sphere_radius_m=R, sigma_S_per_m=0.33,
    )
    np.testing.assert_allclose(V[:2], 0.0, atol=1e-7)         # equator
    assert np.sign(V[2]) == -np.sign(V[3])                    # poles antipodal
    np.testing.assert_allclose(abs(V[2]), abs(V[3]), rtol=1e-6)


def test_eeg_radial_dipole_axisymmetric_about_axis() -> None:
    """Radial dipole on +z: potential is axisymmetric about z."""
    R = 0.1
    r0 = np.array([0.0, 0.0, 0.05])
    Q = np.array([0.0, 0.0, 1e-9])         # radial w.r.t. r0
    sensors = R * np.array([
        [np.cos(0),         np.sin(0),         0.0],
        [np.cos(np.pi / 2), np.sin(np.pi / 2), 0.0],
        [np.cos(np.pi),     np.sin(np.pi),     0.0],
    ])
    V = homogeneous_sphere_eeg_potential(
        r0, Q, sensors, sphere_radius_m=R, sigma_S_per_m=0.33,
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
        r0, Q, np.stack([p_pos, p_neg]),
        sphere_radius_m=R, sigma_S_per_m=0.33,
    )
    assert np.sign(V[0]) == -np.sign(V[1])
    np.testing.assert_allclose(abs(V[0]), abs(V[1]), rtol=1e-6)


# ── MNE-Python cross-validation regression tests ──────────────────────────


def _mne_sphere_field_or_skip():
    """Return MNE's _do_sphere_field, skipping the test if unavailable."""
    try:
        from mne.forward._compute_forward import _do_sphere_field
    except ImportError:                    # pragma: no cover
        pytest.skip("mne-python not available")
    return _do_sphere_field


@pytest.mark.parametrize(
    "src_pos_m,Q_Am",
    [
        (np.array([0.0, 0.0, 0.05]),   np.array([1e-9, 0.0, 0.0])),     # tangential
        (np.array([0.03, 0.02, 0.04]), np.array([0.0, 1e-9, 0.0])),
        (np.array([0.04, 0.0, 0.0]),   np.array([0.0, 0.0, 1e-9])),     # longitudinal
        (np.array([-0.02, 0.03, 0.04]),
         np.array([2.0e-9, -1.0e-9, 0.5e-9])),                         # mixed
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
    sensors_m = np.array([
        [0.10, 0.00, 0.05],
        [0.08, 0.04, 0.05],
        [0.0, 0.10, 0.10],
        [-0.07, 0.05, 0.06],
        [0.06, -0.07, 0.04],
    ])
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
