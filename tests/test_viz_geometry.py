"""Tests for inob.viz.geometry (4-panel geometry overview render)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pytest

from inob.config import load_config
from inob.io.hdf5 import Geometry, save_geometry, save_sensors
from inob.viz.geometry import _subsample_faces, render_geometry

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


@pytest.fixture
def cfg(tmp_path: Path):
    return load_config(TINY_CFG, project_root=tmp_path)


def test_subsample_faces_no_op_when_under_limit() -> None:
    faces = np.arange(30).reshape(10, 3)
    out = _subsample_faces(faces, max_tris=20, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(out, faces)


def test_subsample_faces_reduces_to_limit() -> None:
    faces = np.arange(300).reshape(100, 3)
    out = _subsample_faces(faces, max_tris=10, rng=np.random.default_rng(0))
    assert out.shape == (10, 3)
    # every sampled row must have come from the original array
    orig_rows = {tuple(r) for r in faces}
    assert all(tuple(r) in orig_rows for r in out)


def test_render_geometry_missing_file_raises(cfg) -> None:
    with pytest.raises(FileNotFoundError):
        render_geometry(cfg, with_sensors=False)


def test_render_geometry_writes_png(cfg, tiny_geometry: Geometry) -> None:
    save_geometry(cfg.outputs.geometry_mat, tiny_geometry)
    out = render_geometry(cfg, with_sensors=False)
    assert out == cfg.outputs.geometry_png
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_geometry_with_sensors(cfg, tiny_geometry: Geometry, tiny_sensors) -> None:
    save_geometry(cfg.outputs.geometry_mat, tiny_geometry)
    save_sensors(cfg.outputs.sensors_mat, tiny_sensors)
    out = render_geometry(cfg, with_sensors=True)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_geometry_sensors_skipped_if_missing(cfg, tiny_geometry: Geometry) -> None:
    save_geometry(cfg.outputs.geometry_mat, tiny_geometry)
    # with_sensors=True but no sensors file present -> should not raise
    out = render_geometry(cfg, with_sensors=True)
    assert out.exists()
