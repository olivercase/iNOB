"""Triaxial sensor placement tests."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from inob.io.hdf5 import validate_sensors
from inob.sensors.triaxial import (
    build_triaxial,
    cylindrical_raycast,
    tangent_frame,
)


def test_build_triaxial_orthogonality() -> None:
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((10, 3)) * 50.0
    nrm = rng.standard_normal((10, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    arr = build_triaxial(pos, nrm)
    n = len(pos)
    R = arr.coilori[:n]
    T1 = arr.coilori[n : 2 * n]
    T2 = arr.coilori[2 * n :]
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
    nrm = np.array([[0.0, 0.0, 0.0]])  # degenerate
    arr = build_triaxial(pos, nrm)
    # Falls back to a default axis; orientations still unit-norm + orthogonal
    norms = np.linalg.norm(arr.coilori, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_cylindrical_raycast_on_cube() -> None:
    cube = trimesh.creation.box(extents=(100.0, 100.0, 200.0))
    positions, normals = cylindrical_raycast(
        cube,
        resolution_mm=20.0,
        z_min=-90.0,
        z_max=90.0,
    )
    assert len(positions) > 0
    # All hits land on the cube faces — components within ±50 mm in X/Y, ±100 in Z
    assert np.all(np.abs(positions[:, 0]) <= 51)
    assert np.all(np.abs(positions[:, 1]) <= 51)
    # Outward normals point away from the centroid (cube is centred at 0)
    out_dot = np.einsum("ij,ij->i", normals, positions)
    assert (out_dot >= -1e-6).all()


def test_global_up_tangents_are_orthonormal() -> None:
    rng = np.random.default_rng(3)
    nrm = rng.standard_normal((200, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    t1, t2 = tangent_frame(nrm, basis="global_up")
    np.testing.assert_allclose(np.linalg.norm(t1, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(t2, axis=1), 1.0, atol=1e-12)
    for a, b in ((nrm, t1), (nrm, t2), (t1, t2)):
        np.testing.assert_allclose(np.einsum("ij,ij->i", a, b), 0.0, atol=1e-12)
    # Right-handed: n x T1 == T2, so the frame has a consistent orientation.
    np.testing.assert_allclose(np.cross(nrm, t1), t2, atol=1e-12)


def test_global_up_t1_is_horizontal_and_t2_is_head_foot() -> None:
    """The point of the fixed reference: T1/T2 mean the same thing everywhere."""
    # Normals pointing outward around the body at one height.
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    nrm = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    t1, t2 = tangent_frame(nrm, basis="global_up")
    # T1 has no superior component at all — it is the circumferential axis.
    np.testing.assert_allclose(t1[:, 2], 0.0, atol=1e-12)
    # T2 is the superior axis, for every sensor around the ring.
    np.testing.assert_allclose(t2, np.tile([0.0, 0.0, 1.0], (len(nrm), 1)), atol=1e-12)


def _tilted_ring(step_deg: float = 2.0) -> np.ndarray:
    """Outward normals around the body, tilted as real skin tilts them."""
    theta = np.radians(np.arange(0.0, 360.0, step_deg))
    nrm = np.column_stack([np.cos(theta), np.sin(theta), 0.15 * np.sin(3.0 * theta)])
    return nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def _adjacent_angles_deg(v: np.ndarray) -> np.ndarray:
    dots = np.abs(np.einsum("ij,ij->i", v[:-1], v[1:]))
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def test_global_up_t1_tracks_the_normals_where_local_svd_jumps() -> None:
    """The defect this basis fixes: T1 must turn no faster than the surface.

    Sensors 2 degrees apart measure nearly the same physical direction, so
    their T1 axes should be about 2 degrees apart too. The per-sensor null
    space manages that on average but jumps by several times as much at
    isolated positions, which is what puts steps into a tangential field map
    that the field itself does not have.
    """
    nrm = _tilted_ring(step_deg=2.0)
    step = _adjacent_angles_deg(nrm).max()

    g1, _ = tangent_frame(nrm, basis="global_up")
    assert _adjacent_angles_deg(g1).max() <= step + 1e-6

    s1, _ = tangent_frame(nrm, basis="local_svd")
    assert _adjacent_angles_deg(s1).max() > 4.0 * step


def test_global_up_t1_stays_horizontal_under_tilted_normals() -> None:
    """T1 keeps one anatomical meaning; the null-space tangent drifts off it."""
    nrm = _tilted_ring(step_deg=2.0)

    g1, _ = tangent_frame(nrm, basis="global_up")
    np.testing.assert_allclose(g1[:, 2], 0.0, atol=1e-12)

    s1, _ = tangent_frame(nrm, basis="local_svd")
    drift_deg = np.degrees(np.arcsin(np.abs(s1[:, 2]))).max()
    assert drift_deg > 5.0


def test_tangent_basis_does_not_change_the_measurable_field() -> None:
    """Only the split between T1 and T2 moves; the physics does not.

    The radial channel and the total field the three axes capture are
    identical in both bases, so no leadfield magnitude depends on the choice.
    """
    rng = np.random.default_rng(11)
    nrm = rng.standard_normal((64, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    field = rng.standard_normal((64, 3)) * 1e-13

    captured = {}
    for basis in ("global_up", "local_svd"):
        t1, t2 = tangent_frame(nrm, basis=basis)
        proj = np.column_stack(
            [
                np.einsum("ij,ij->i", field, nrm),
                np.einsum("ij,ij->i", field, t1),
                np.einsum("ij,ij->i", field, t2),
            ]
        )
        captured[basis] = proj

    np.testing.assert_allclose(captured["global_up"][:, 0], captured["local_svd"][:, 0])
    np.testing.assert_allclose(
        np.linalg.norm(captured["global_up"], axis=1),
        np.linalg.norm(captured["local_svd"], axis=1),
        rtol=1e-12,
    )


def test_global_up_handles_normals_parallel_to_the_reference_axis() -> None:
    """Crown of the head and soles of the feet: up x n is degenerate there."""
    nrm = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    t1, t2 = tangent_frame(nrm, basis="global_up")
    assert np.isfinite(t1).all() and np.isfinite(t2).all()
    np.testing.assert_allclose(np.linalg.norm(t1, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(t2, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.einsum("ij,ij->i", nrm, t1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.einsum("ij,ij->i", t1, t2), 0.0, atol=1e-12)


def test_tangent_frame_rejects_an_unknown_basis() -> None:
    with pytest.raises(ValueError, match="unknown tangent basis"):
        tangent_frame(np.array([[0.0, 1.0, 0.0]]), basis="nope")


def test_build_triaxial_honours_the_requested_basis() -> None:
    # A normal off every coordinate plane — the two bases coincide on one.
    pos = np.array([[100.0, 0.0, 0.0]])
    nrm = np.array([[0.5, 0.6, 0.62]])
    glob = build_triaxial(pos, nrm, tangent_basis="global_up")
    svd = build_triaxial(pos, nrm, tangent_basis="local_svd")
    assert not np.allclose(glob.coilori[1], svd.coilori[1])
    # Radial is the same either way.
    np.testing.assert_allclose(glob.coilori[0], svd.coilori[0], atol=1e-12)
