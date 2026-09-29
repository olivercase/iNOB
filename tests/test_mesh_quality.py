"""Mesh-quality metrics + validation gates."""

from __future__ import annotations

import numpy as np
import pytest

from inob.mesh.quality import (
    MeshQualityError,
    assert_mesh_ok,
    assert_units_mm,
    compute_quality,
    quality_by_tissue,
)


def _regular_tet() -> tuple[np.ndarray, np.ndarray]:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
        ],
        dtype=np.float64,
    )
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return nodes, tets


def _two_tets() -> tuple[np.ndarray, np.ndarray]:
    """One well-shaped tetrahedron and one sliver, sharing no nodes."""
    good, _ = _regular_tet()
    sliver = np.array(
        [
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.5, 0.866, 0.0],
            [10.5, 0.289, 0.002],  # almost coplanar with the other three
        ],
        dtype=np.float64,
    )
    nodes = np.vstack([good, sliver])
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    return nodes, tets


def _degenerate_tet() -> tuple[np.ndarray, np.ndarray]:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # collinear with the first three
            [3.0, 1e-6, 0.0],
        ],
        dtype=np.float64,
    )
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


def test_quality_by_tissue_splits_the_mesh_by_label() -> None:
    nodes, tets = _two_tets()
    tissue = np.array([1, 2])
    stats = quality_by_tissue(nodes, tets, tissue, ("good", "bad"))
    assert set(stats) == {"good", "bad"}
    assert stats["good"].n_tets == 1
    assert stats["bad"].n_tets == 1
    # The sliver is the worse of the two, and lands under the other's label.
    assert stats["bad"].min < stats["good"].min


def test_quality_by_tissue_omits_labels_with_no_elements() -> None:
    nodes, tets = _two_tets()
    tissue = np.array([1, 1])
    stats = quality_by_tissue(nodes, tets, tissue, ("present", "absent"))
    assert set(stats) == {"present"}


def test_min_quality_is_a_gate_and_never_alters_the_mesh() -> None:
    """Nick's question: does changing the threshold change the result?

    No. ``assert_mesh_ok`` only accepts or rejects — it holds no reference to
    the nodes or elements and nothing re-meshes in response to it. Every
    threshold below the mesh's own minimum accepts the identical mesh; every
    threshold above it rejects the build. So the knob moves where the build
    stops, not what it produces.
    """
    nodes, tets = _two_tets()
    stats = compute_quality(nodes, tets)
    before = (nodes.copy(), tets.copy())

    for threshold in (0.0, stats.min * 0.5, stats.min):
        assert_mesh_ok(stats, min_quality=threshold)
    with pytest.raises(MeshQualityError):
        assert_mesh_ok(stats, min_quality=stats.min * 1.01)

    np.testing.assert_array_equal(nodes, before[0])
    np.testing.assert_array_equal(tets, before[1])
