"""Triaxial sensor placement tests."""
from __future__ import annotations

import numpy as np
import trimesh

from inob.io.hdf5 import validate_sensors
from inob.sensors.triaxial import build_triaxial, cylindrical_raycast


def test_build_triaxial_orthogonality() -> None:
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((10, 3)) * 50.0
    nrm = rng.standard_normal((10, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    arr = build_triaxial(pos, nrm)
    n = len(pos)
    R = arr.coilori[:n]
    T1 = arr.coilori[n:2 * n]
    T2 = arr.coilori[2 * n:]
    assert np.allclose(np.einsum("ij,ij->i", R, T1), 0.0, atol=1e-9)
    assert np.allclose(np.einsum("ij,ij->i", R, T2), 0.0, atol=1e-9)
    assert np.allclose(np.einsum("ij,ij->i", T1, T2), 0.0, atol=1e-9)


def test_build_triaxial_unit_norm() -> None:
    pos = np.array([[100.0, 0, 0], [0, 100, 0], [0, 0, 100]])
    nrm = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    arr = build_triaxial(pos, nrm)
    norms = np.linalg.norm(arr.coilori, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)


def test_build_triaxial_label_format() -> None:
    pos = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    nrm = pos.copy()
    arr = build_triaxial(pos, nrm)
    # 2 positions × 3 axes = 6 channels
    assert len(arr.labels) == 6
    assert arr.labels[0] == "mag-0001-R"
    assert arr.labels[1] == "mag-0002-R"
    assert arr.labels[2] == "mag-0001-T1"
    assert arr.labels[5] == "mag-0002-T2"


def test_build_triaxial_validates() -> None:
    pos = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    nrm = pos.copy()
    arr = build_triaxial(pos, nrm)
    validate_sensors(arr)


def test_build_triaxial_handles_zero_normal() -> None:
    pos = np.array([[1.0, 0, 0]])
    nrm = np.array([[0.0, 0.0, 0.0]])   # degenerate
    arr = build_triaxial(pos, nrm)
    # Falls back to a default axis; orientations still unit-norm + orthogonal
    norms = np.linalg.norm(arr.coilori, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_cylindrical_raycast_on_cube() -> None:
    cube = trimesh.creation.box(extents=(100.0, 100.0, 200.0))
    positions, normals = cylindrical_raycast(
        cube, resolution_mm=20.0, z_min=-90.0, z_max=90.0,
    )
    assert len(positions) > 0
    # All hits land on the cube faces — components within ±50 mm in X/Y, ±100 in Z
    assert np.all(np.abs(positions[:, 0]) <= 51)
    assert np.all(np.abs(positions[:, 1]) <= 51)
    # Outward normals point away from the centroid (cube is centred at 0)
    out_dot = np.einsum("ij,ij->i", normals, positions)
    assert (out_dot >= -1e-6).all()
