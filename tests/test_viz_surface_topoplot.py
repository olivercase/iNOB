"""Tests for inob.viz.surface_topoplot: geometry helpers + render smoke test."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import trimesh

from inob.viz.surface_topoplot import (
    _crop_skin_to_band,
    _cylindrical_unroll,
    gaussian_interpolate_surface,
    render_surface_topoplots,
)

from tests.viz_pipeline_helpers import build_pipeline_cfg


def test_gaussian_interpolate_surface_reproduces_constant_field() -> None:
    rng = np.random.default_rng(0)
    sensor_pos = rng.uniform(-50, 50, size=(20, 3))
    sensor_val = np.full(20, 3.5)
    target = rng.uniform(-50, 50, size=(10, 3))
    out = gaussian_interpolate_surface(sensor_pos, sensor_val, target, sigma_mm=20.0)
    np.testing.assert_allclose(out, 3.5, atol=1e-9)


def test_gaussian_interpolate_surface_shape() -> None:
    sensor_pos = np.random.default_rng(1).uniform(-10, 10, size=(5, 3))
    sensor_val = np.arange(5, dtype=float)
    target = np.random.default_rng(2).uniform(-10, 10, size=(7, 3))
    out = gaussian_interpolate_surface(sensor_pos, sensor_val, target, sigma_mm=5.0, k_nearest=3)
    assert out.shape == (7,)


def test_gaussian_interpolate_surface_nearest_dominates_with_small_sigma() -> None:
    sensor_pos = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
    sensor_val = np.array([1.0, 100.0])
    target = np.array([[0.1, 0.0, 0.0]])
    out = gaussian_interpolate_surface(sensor_pos, sensor_val, target, sigma_mm=1.0)
    assert out[0] == pytest.approx(1.0, abs=1e-6)


def test_crop_skin_to_band_keeps_only_in_range_vertices() -> None:
    sphere = trimesh.creation.icosphere(radius=50.0, subdivisions=3)
    v, f = _crop_skin_to_band(sphere, z_lo=-10.0, z_hi=10.0)
    assert len(v) > 0
    assert (v[:, 2] >= -10.0).all() and (v[:, 2] <= 10.0).all()
    # every face must reference only kept vertices
    assert f.max() < len(v)


def test_crop_skin_to_band_empty_when_out_of_range() -> None:
    sphere = trimesh.creation.icosphere(radius=50.0, subdivisions=2)
    v, f = _crop_skin_to_band(sphere, z_lo=1000.0, z_hi=2000.0)
    assert len(v) == 0
    assert len(f) == 0


def test_cylindrical_unroll_axis_aligned_point() -> None:
    pos = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, -5.0]])
    theta, z = _cylindrical_unroll(pos, axis_xy=(0.0, 0.0))
    np.testing.assert_allclose(theta, [0.0, 90.0])
    np.testing.assert_allclose(z, [5.0, -5.0])


def test_cylindrical_unroll_offset_axis() -> None:
    pos = np.array([[10.0, 0.0, 0.0]])
    theta, _ = _cylindrical_unroll(pos, axis_xy=(10.0, 0.0))
    # point coincides with axis -> arctan2(0, 0) == 0
    np.testing.assert_allclose(theta, [0.0])


def test_render_surface_topoplots_smoke(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    out = render_surface_topoplots(cfg, out_path=tmp_path / "surf.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_surface_topoplots_default_out_path(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    out = render_surface_topoplots(cfg, source_idx=2)
    assert out == cfg.outputs.base / "surface_topoplots_vagus.png"
    assert out.exists()
