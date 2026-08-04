"""Tests for the homogeneous-sphere EEG calibration helpers.

``build_sphere_fem`` (needs iso2mesh) and ``run_sphere_eeg_forward`` /
``calibrate_eeg_factor`` (need duneuropy) are not exercised here — they
require heavy external mesh/FEM dependencies not available in the unit-test
environment. We cover the pure geometry and analytic-potential helpers.
"""
from __future__ import annotations

import numpy as np

from inob.analysis.sphere_calibration import (
    _per_dipole_analytic,
    build_sphere_electrodes,
)


def test_build_sphere_electrodes_on_sphere_surface() -> None:
    radius = 100.0
    electrodes = build_sphere_electrodes(radius, n_electrodes=64, seed=0)
    assert electrodes.coilpos.shape == (64, 3)
    r = np.linalg.norm(electrodes.coilpos, axis=1)
    np.testing.assert_allclose(r, radius, atol=1e-6)
    assert len(electrodes.labels) == 64
    assert electrodes.unit == "mm"
    assert all(t == "eeg" for t in electrodes.chantype)


def test_build_sphere_electrodes_outward_normals() -> None:
    radius = 50.0
    electrodes = build_sphere_electrodes(radius, n_electrodes=32, seed=1)
    # orientation should point radially outward: unit vector == pos / radius
    np.testing.assert_allclose(
        electrodes.coilori, electrodes.coilpos / radius, atol=1e-6,
    )
    np.testing.assert_allclose(np.linalg.norm(electrodes.coilori, axis=1), 1.0, atol=1e-6)


def test_build_sphere_electrodes_deterministic_with_seed() -> None:
    e1 = build_sphere_electrodes(80.0, n_electrodes=20, seed=42)
    e2 = build_sphere_electrodes(80.0, n_electrodes=20, seed=42)
    np.testing.assert_array_equal(e1.coilpos, e2.coilpos)


def test_per_dipole_analytic_shape_and_units() -> None:
    src = np.array([0.0, 0.0, 0.05])       # 50 mm depth, in metres
    elec = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    out = _per_dipole_analytic(src, elec, radius_mm=100.0, sigma_S_per_m=0.43)
    assert out.shape == (3, 3)
    assert np.isfinite(out).all()
    # Not all-zero: a dipole should induce nonzero potential somewhere
    assert np.any(np.abs(out) > 0)


def test_per_dipole_analytic_scales_inversely_with_conductivity() -> None:
    src = np.array([0.0, 0.0, 0.05])
    elec = np.array([[0.1, 0.0, 0.0]])
    v_low = _per_dipole_analytic(src, elec, radius_mm=100.0, sigma_S_per_m=0.2)
    v_high = _per_dipole_analytic(src, elec, radius_mm=100.0, sigma_S_per_m=0.4)
    # homogeneous-sphere potential is inversely proportional to sigma
    np.testing.assert_allclose(v_low, v_high * 2.0, rtol=1e-6)
