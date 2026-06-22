"""Mesh-quality metrics + validation gates."""
from __future__ import annotations

import numpy as np
import pytest

from vagus_fm.mesh.quality import (
    MeshQualityError,
    assert_mesh_ok,
    assert_units_mm,
    compute_quality,
)


def _regular_tet() -> tuple[np.ndarray, np.ndarray]:
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3) / 2, 0.0],
        [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
    ], dtype=np.float64)
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return nodes, tets


def _degenerate_tet() -> tuple[np.ndarray, np.ndarray]:
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],     # collinear with the first three
        [3.0, 1e-6, 0.0],
    ], dtype=np.float64)
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return nodes, tets


def test_quality_regular_tet_close_to_one() -> None:
    n, t = _regular_tet()
    s = compute_quality(n, t)
    assert s.n_tets == 1
    assert s.min > 0.95


def test_quality_degenerate_tet_close_to_zero() -> None:
    n, t = _degenerate_tet()
    s = compute_quality(n, t)
    assert s.min < 0.01


def test_assert_mesh_ok_passes_on_regular() -> None:
    n, t = _regular_tet()
    assert_mesh_ok(compute_quality(n, t), min_quality=0.5)


def test_assert_mesh_ok_raises_on_degenerate() -> None:
    n, t = _degenerate_tet()
    with pytest.raises(MeshQualityError, match="min mesh quality"):
        assert_mesh_ok(compute_quality(n, t), min_quality=0.05)


def test_assert_units_mm_passes_for_50_mm_box() -> None:
    nodes = np.array([[0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50]], dtype=np.float64)
    assert_units_mm(nodes)


def test_assert_units_mm_rejects_metres() -> None:
    nodes = np.array([[0, 0, 0], [0.1, 0, 0]], dtype=np.float64)
    with pytest.raises(MeshQualityError, match="implausible for mm"):
        assert_units_mm(nodes)


def test_quality_empty_mesh() -> None:
    s = compute_quality(np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64))
    assert s.n_tets == 0
    # assert_mesh_ok must not raise on empty
    assert_mesh_ok(s, min_quality=0.99)
