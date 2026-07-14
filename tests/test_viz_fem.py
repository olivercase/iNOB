"""Tests for inob.viz.fem (PyVista FEM render)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from inob.config import load_config
from inob.io.hdf5 import FemMesh, save_fem
from inob.viz.fem import _build_grid, _build_lut, render_fem

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


@pytest.fixture
def cfg(tmp_path: Path):
    return load_config(TINY_CFG, project_root=tmp_path)


def test_build_grid_shapes(tiny_fem: FemMesh) -> None:
    grid = _build_grid(tiny_fem.nodes, tiny_fem.tets, tiny_fem.tissue)
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_points == len(tiny_fem.nodes)
    assert grid.n_cells == len(tiny_fem.tets)
    np.testing.assert_array_equal(grid["tissue"], tiny_fem.tissue.astype(np.float32))


def test_build_lut_covers_tissue_ids() -> None:
    uid = np.array([1, 2], dtype=np.int32)
    id_to_label = {1: "vagus_left", 2: "skin"}
    lut, n = _build_lut(uid, id_to_label)
    assert n == 3  # max id 2 -> 0..2
    assert isinstance(lut, pv.LookupTable)
    assert lut.GetNumberOfTableValues() == n


def test_render_fem_missing_file_raises(cfg) -> None:
    with pytest.raises(FileNotFoundError):
        render_fem(cfg)


def test_render_fem_writes_png(cfg, tiny_fem: FemMesh) -> None:
    save_fem(cfg.outputs.fem_mat, tiny_fem)
    out = render_fem(cfg)
    assert out == cfg.outputs.fem_png
    assert out.exists()
    assert out.stat().st_size > 0
