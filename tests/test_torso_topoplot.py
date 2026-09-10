"""Unit tests for the body-surface field map's building blocks.

The figure itself needs a solved leadfield and the geometry mesh, so it is not
rendered here. What is tested is the handful of decisions that are easy to get
silently wrong and invisible in a finished PNG: which channels are used, where
the interpolation refuses to paint, and whether the source marker is drawn
through the body.
"""

from __future__ import annotations

import numpy as np
import pytest

from inob.viz import torso_topoplot as tt


class _Leadfield:
    """Minimal stand-in: 2 positions x 3 axes, 2 sources."""

    def __init__(self) -> None:
        # Rows 0,1 are the radial component of positions 0,1; rows 2-5 are the
        # other two axes, which the figure must ignore.
        self.L_fT_per_nAm = np.arange(6 * 6, dtype=float).reshape(6, 6)
        self.coil_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],  # radial
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],  # axis 2
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],  # axis 3
            ]
        )


def test_only_the_radial_third_of_a_triaxial_array_is_used() -> None:
    vals, pos = tt._signed_radial_field(_Leadfield(), source_idx=1, moment="z")
    # 6 channels / 3 axes = 2 radial positions.
    assert vals.shape == (2,) and pos.shape == (2, 3)
    # Source 1, z moment -> column 3*1 + 2 = 5, rows 0 and 1.
    np.testing.assert_allclose(vals, [5.0, 11.0])


def test_the_field_keeps_its_sign() -> None:
    lf = _Leadfield()
    lf.L_fT_per_nAm = -np.ones((6, 6))
    vals, _ = tt._signed_radial_field(lf, source_idx=0, moment="x")
    assert (vals < 0).all(), "a magnitude here would erase one lobe of the dipole"


def test_interpolation_reports_the_distance_to_the_nearest_sensor() -> None:
    sensors = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    vals = np.array([1.0, -1.0])
    verts = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [400.0, 0.0, 0.0]])
    out, nearest = tt._interpolate_to_surface(sensors, vals, verts, sigma_mm=30.0)
    np.testing.assert_allclose(nearest, [0.0, 50.0, 300.0])
    # On top of a sensor the interpolant is that sensor's value, up to the
    # small pull of the other sensor 100 mm away (weight e^-0.5(100/30)^2).
    assert out[0] == pytest.approx(1.0, abs=0.01)
    # Midway between two opposite sensors it is ~0.
    assert abs(out[1]) < 1e-6
    # The far vertex is what `max_extrap_mm` exists to leave unpainted.
    assert nearest[2] > 100.0


def test_crop_to_band_reindexes_faces_consistently() -> None:
    verts = np.array([[0.0, 0.0, z] for z in (0.0, 5.0, 10.0, 100.0)])
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    v, f = tt._crop_to_band(verts, faces, 0.0, 20.0)
    assert len(v) == 3
    # Only the face whose vertices all survive is kept, re-indexed into `v`.
    np.testing.assert_array_equal(f, [[0, 1, 2]])
    np.testing.assert_allclose(v[f][0, :, 2], [0.0, 5.0, 10.0])


def test_crop_near_keeps_a_cap_not_a_ring() -> None:
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [200.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    v, f = tt._crop_near(verts, faces, np.zeros(3), 10.0)
    assert len(v) == 3
    np.testing.assert_array_equal(f, [[0, 1, 2]])


def test_shading_darkens_without_changing_hue() -> None:
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    rgba = np.array([[0.8, 0.2, 0.2, 1.0]])
    out = tt._shade(verts, faces, rgba)
    assert (out[:, :3] <= rgba[:, :3] + 1e-12).all(), "shading must only darken"
    # Red stays the dominant channel: a shaded red face is still red.
    assert out[0, 0] > out[0, 1] and out[0, 0] > out[0, 2]
    assert out[0, 3] == 1.0


def test_source_behind_the_body_is_not_marked() -> None:
    axis = np.zeros(2)
    front = np.array([0.0, -100.0, 1300.0])
    back = np.array([0.0, 100.0, 1300.0])
    # Camera at azim -90 looks from -y, so the -y source faces it.
    assert tt._source_is_visible(front, axis, -90.0) is True
    assert tt._source_is_visible(back, axis, -90.0) is False
    # And the reverse from the other side.
    assert tt._source_is_visible(back, axis, 90.0) is True
