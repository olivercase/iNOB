"""Regression tests for turning a hollow anatomical shell into a solid body.

The bug these lock down was silent: solidifying the BodyParts3D skin shell in the
atlas frame gave 78 L, and solidifying the *same* shell after a ~1 degree
registration rotation gave 19 L -- the wall alone. Both results were watertight,
one body, with sane bounds, so every topology check passed and the FEM was handed
a 2 mm rind around a vacuum. The cause is that the skin wall is thinner than the
voxel, so whether the splat leaves a 6-connected pinhole depends on how the
surface happens to cut the grid.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from inob.mesh.solidify import exterior_mask, solidify_shell


def _hollow_ball(r_out: float = 40.0, wall: float = 2.0) -> trimesh.Trimesh:
    """A closed double-walled sphere: the shell of a solid, like the skin organ."""
    outer = trimesh.creation.icosphere(subdivisions=4, radius=r_out)
    inner = trimesh.creation.icosphere(subdivisions=4, radius=r_out - wall)
    inner.invert()
    return trimesh.util.concatenate([outer, inner])


def test_fills_the_cavity_not_just_the_wall():
    shell = _hollow_ball()
    solid = solidify_shell(shell, pitch=3.0, n_samples=200_000, seal_mm=9.0,
                           decimate_target=None, smooth_iter=0, seed=0)
    expected = 4 / 3 * np.pi * 40.0 ** 3
    assert solid.volume > 0, "orientation must be outward, so volume is positive"
    # A 40 mm ball on a 3 mm grid carries a ~0.5-voxel outward surface bias, so
    # it reads ~15% large. Real anatomy is 10x the size at the same pitch.
    assert solid.volume == pytest.approx(expected, rel=0.20)
    # the failure mode: enclosing only the wall
    assert solid.volume > 5 * abs(shell.volume)


@pytest.mark.parametrize("angle_deg", [0.0, 1.0, 17.0, 45.0])
def test_result_does_not_depend_on_the_frame(angle_deg):
    """A rotation must not change the filled volume. This is the actual bug."""
    shell = _hollow_ball()
    rot = trimesh.transformations.rotation_matrix(
        np.deg2rad(angle_deg), [0.3, 0.5, 0.81])
    turned = shell.copy()
    turned.apply_transform(rot)
    kw = dict(pitch=3.0, n_samples=200_000, seal_mm=9.0,
              decimate_target=None, smooth_iter=0, seed=0)
    assert solidify_shell(turned, **kw).volume == pytest.approx(
        solidify_shell(shell, **kw).volume, rel=0.05)


def test_gate_rejects_a_leaked_fill():
    """A shell with an opening wider than the seal floods, and the gate says so.

    Without the gate this returns a watertight, single-body, correctly-bounded
    mesh of the *wall*, which is exactly how the real failure went unnoticed.
    """
    shell = _hollow_ball()
    # Cut a 20 mm-wide mouth through both walls: wider than seal_mm, so the
    # erosion cannot sever it and the exterior reaches the cavity legitimately.
    keep = ~((np.abs(shell.triangles_center[:, 0]) < 10.0)
             & (np.abs(shell.triangles_center[:, 1]) < 10.0)
             & (shell.triangles_center[:, 2] > 0))
    holed = shell.submesh([np.flatnonzero(keep)], append=True)
    with pytest.raises(RuntimeError, match="wall, not the body"):
        solidify_shell(holed, pitch=3.0, n_samples=200_000, seal_mm=6.0,
                       wall_voxels=1, decimate_target=None, smooth_iter=0, seed=0)


def test_exterior_mask_does_not_enter_a_sealed_cavity():
    occ = np.zeros((30, 30, 30), bool)
    occ[8:22, 8:22, 8:22] = True
    occ[10:20, 10:20, 10:20] = False          # cavity inside a thick wall
    ext = exterior_mask(occ, seal_voxels=1)
    assert not ext[15, 15, 15], "cavity is not outside"
    assert ext[0, 0, 0], "grid corner is outside"
