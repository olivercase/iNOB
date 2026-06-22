"""Tests for the conductivity-vector builder (no DUNEuro required)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vagus_fm.config import load_config
from vagus_fm.forward.duneuro_driver import (
    build_conductivity_vector,
    build_driver_config,
)
from vagus_fm.io.hdf5 import FemMesh

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"


def test_conductivity_vector_full_4_tissue() -> None:
    cfg = load_config(DEFAULT_CFG)
    nodes = np.eye(4, 3) * 50.0
    nodes[0] = 0
    tets = np.array([[0, 1, 2, 3]] * 4, dtype=np.int32)
    tissue = np.array([1, 2, 3, 4], dtype=np.int32)
    fem = FemMesh(nodes, tets, tissue,
                  ("vagus_left", "vagus_right", "bone", "skin"), "mm")
    cond = build_conductivity_vector(cfg, fem)
    expected = np.array([0.30, 0.30, 0.0042, 0.43]) * 1e-3
    np.testing.assert_allclose(cond, expected)


def test_conductivity_missing_label_raises(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CFG)
    nodes = np.eye(4, 3) * 50.0
    nodes[0] = 0
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    tissue = np.array([1], dtype=np.int32)
    fem = FemMesh(nodes, tets, tissue, ("muscle",), "mm")
    with pytest.raises(KeyError, match="muscle"):
        build_conductivity_vector(cfg, fem)


def test_driver_config_shape_and_solver_settings() -> None:
    cfg = load_config(DEFAULT_CFG)
    nodes = np.eye(4, 3) * 50.0
    nodes[0] = 0
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    tissue = np.array([1], dtype=np.int32)
    fem = FemMesh(nodes, tets, tissue, ("vagus_left",), "mm")
    cond = np.array([3e-4])
    d = build_driver_config(cfg, fem, cond)
    assert d["type"] == "fitted"
    assert d["solver_type"] == "cg"
    assert d["solver"]["scheme"] == "sipg"
    assert d["solver"]["reduction"] == "1e-10"
    assert d["meg"]["intorderadd"] == "5"
    np.testing.assert_array_equal(d["volume_conductor"]["grid"]["elements"], tets)
    np.testing.assert_array_equal(d["volume_conductor"]["tensors"]["labels"],
                                   tissue.astype(np.int64) - 1)
