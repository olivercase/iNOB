"""Voxelisation strategies."""
from __future__ import annotations

import numpy as np
import trimesh

from vagus_fm.mesh.voxelize import (
    fill_internal_cavities,
    keep_largest_component,
    voxelize_mesh,
    voxelize_solid_for_mesh,
    voxelize_watertight,
)


def _grid(mn: np.ndarray, mx: np.ndarray, pitch: float):
    shape = tuple(((mx - mn) / pitch).astype(int) + 1)
    xs = mn[0] + (np.arange(shape[0]) + 0.5) * pitch
    ys = mn[1] + (np.arange(shape[1]) + 0.5) * pitch
    zs = mn[2] + (np.arange(shape[2]) + 0.5) * pitch
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return X, Y, Z, shape


def test_voxelize_watertight_unit_sphere() -> None:
    R = 50.0
    sphere = trimesh.creation.icosphere(radius=R, subdivisions=3)
    pitch = 5.0
    pad = 30.0
    mn = np.array([-R - pad] * 3, dtype=np.float64)
    mx = np.array([R + pad] * 3, dtype=np.float64)
    X, _, _, _ = _grid(mn, mx, pitch)
    occ = voxelize_watertight(sphere, X, pitch=pitch, mn=mn)
    n_vox = int(occ.sum())
    expected = (4 / 3) * np.pi * R**3 / pitch**3
    assert n_vox > 0
    rel_err = abs(n_vox - expected) / expected
    # 10 voxels across radius → ~20% bias from boundary discretization is normal.
    assert rel_err < 0.25, f"n_vox={n_vox} expected≈{expected:.0f} rel_err={rel_err:.3f}"


def test_voxelize_mesh_handles_open_surface() -> None:
    sphere = trimesh.creation.icosphere(radius=30.0, subdivisions=3)
    # Drop one face to open it (skin-like, with anatomical opening)
    sphere.update_faces(np.arange(1, len(sphere.faces)))
    pitch = 5.0
    mn = np.array([-60.0] * 3)
    mx = np.array([60.0] * 3)
    X, _, _, _ = _grid(mn, mx, pitch)
    occ = voxelize_mesh(sphere, X, pitch=pitch, mn=mn,
                        closing_mm=10.0, dilate_mm=2.0)
    # Closing + flood fill should still produce a body interior
    assert int(occ.sum()) > 0


def test_voxelize_solid_for_mesh_box() -> None:
    box = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    pitch = 2.0
    mn = np.array([-30.0] * 3)
    mx = np.array([30.0] * 3)
    X, Y, Z, _ = _grid(mn, mx, pitch)
    occ = voxelize_solid_for_mesh(box, X, Y, Z, pitch=pitch, mn=mn)
    expected = (20.0 ** 3) / pitch**3
    n_vox = int(occ.sum())
    assert abs(n_vox - expected) / expected < 0.15


def test_keep_largest_component() -> None:
    occ = np.zeros((10, 10, 10), dtype=bool)
    occ[1:5, 1:5, 1:5] = True   # 64 voxels
    occ[8, 8, 8] = True          # tiny component
    out = keep_largest_component(occ)
    assert int(out.sum()) == 64
    assert not out[8, 8, 8]


def test_fill_internal_cavities() -> None:
    occ = np.zeros((10, 10, 10), dtype=bool)
    occ[1:9, 1:9, 1:9] = True
    occ[4:6, 4:6, 4:6] = False   # internal cavity
    out = fill_internal_cavities(occ)
    assert int(out.sum()) > int(occ.sum())
    assert out[4, 4, 4]
