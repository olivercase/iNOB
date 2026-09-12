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


def _unit_body() -> tuple[np.ndarray, np.ndarray]:
    """A four-vertex scrap of "skin" spanning 100 mm, enough to frame."""
    verts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3]], dtype=np.int64)
    return verts, faces


def _panel(ax: object, sensor_val: np.ndarray, **kw: object) -> float:
    verts, faces = _unit_body()
    sensors = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    return tt._body_panel(
        ax,
        verts,
        faces,
        sensors,
        sensor_val,
        np.array([10.0, 10.0, 10.0]),
        sigma_mm=30.0,
        k_nearest=2,
        max_extrap_mm=1e9,
        elev=10.0,
        azim=-35.0,
        axis_xy=np.zeros(2),
        cmap=tt.divergent_cmap(),
        rng=np.random.default_rng(0),
        title="",
        **kw,
    )


def test_both_panels_get_the_same_camera() -> None:
    """The figure's whole claim is that only the physics differs between a and b."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    axa = fig.add_subplot(1, 2, 1, projection="3d")
    axb = fig.add_subplot(1, 2, 2, projection="3d")
    # Fields four orders of magnitude apart, as MEG and ESG actually are here.
    _panel(axa, np.array([1.0, -1.0]))
    _panel(axb, np.array([1e-4, -1e-4]))
    assert axa.get_xlim() == axb.get_xlim()
    assert axa.get_ylim() == axb.get_ylim()
    assert axa.get_zlim() == axb.get_zlim()
    assert (axa.elev, axa.azim) == (axb.elev, axb.azim)
    plt.close(fig)


def test_the_colour_limit_ignores_skin_the_array_never_reached() -> None:
    """Otherwise one distant, strong sensor sets the scale for the skin in frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verts = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [900.0, 0.0, 20.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    # A weak sensor on the two near vertices and a 100x stronger one 1 m away,
    # whose only vertex is beyond max_extrap_mm and so must not be painted.
    sensors = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
    vals = np.array([1.0, 100.0])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    vlim = tt._body_panel(
        ax,
        verts,
        faces,
        sensors,
        vals,
        np.array([0.0, 0.0, 0.0]),
        sigma_mm=30.0,
        k_nearest=2,
        max_extrap_mm=50.0,
        elev=10.0,
        azim=0.0,
        axis_xy=np.zeros(2),
        cmap=tt.divergent_cmap(),
        rng=np.random.default_rng(0),
        title="",
    )
    plt.close(fig)
    assert vlim == pytest.approx(1.0, abs=0.05), "the unmeasured vertex set the scale"


def test_array_pitch_is_the_arrays_own_spacing_not_a_config_number() -> None:
    """Panel b may be fed a 5 mm paddle or a ~40 mm whole-torso array."""
    grid = np.array([[x, y, 0.0] for x in range(0, 100, 20) for y in range(0, 100, 20)])
    assert tt._array_pitch_mm(grid) == pytest.approx(20.0)
    assert tt._array_pitch_mm(grid * 0.25) == pytest.approx(5.0)
    # One contact carries no spacing; it must not divide by zero downstream.
    assert tt._array_pitch_mm(np.zeros((1, 3))) > 0.0


def test_the_scale_bar_stays_inside_the_panel_at_every_azimuth() -> None:
    """It is drawn along the screen axis, which reverses as the camera swings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for azim in (-135.0, -45.0, 45.0, 135.0):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        ax.set_zlim(0, 400)
        tt._scale_bar(ax, azim, mm=50.0)
        (xs, ys, _) = ax.lines[-1].get_data_3d()
        assert 0.0 <= min(xs) and max(xs) <= 400.0, f"bar left the panel in x at {azim}"
        assert 0.0 <= min(ys) and max(ys) <= 400.0, f"bar left the panel in y at {azim}"
        plt.close(fig)
