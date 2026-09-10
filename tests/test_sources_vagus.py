"""Vagus dipole-sampling tests."""

from __future__ import annotations

import numpy as np
import pytest

from inob.io.hdf5 import FemMesh
from inob.sources.vagus import vagus_sources


def _build_tube_fem(length_mm: float = 100.0, n_slabs: int = 50) -> FemMesh:
    """Build a synthetic 'vagus tube' FEM with one tet per Z-slab."""
    z_centres = np.linspace(0.0, length_mm, n_slabs)
    nodes_list = []
    tets_list = []
    base_idx = 0
    for z in z_centres:
        nodes_list.append(
            [
                [0.0, 0.0, z],
                [1.0, 0.0, z],
                [0.0, 1.0, z],
                [0.0, 0.0, z + 1.0],
            ]
        )
        tets_list.append([base_idx, base_idx + 1, base_idx + 2, base_idx + 3])
        base_idx += 4
    nodes = np.vstack(nodes_list).astype(np.float64)
    tets = np.array(tets_list, dtype=np.int32)
    tissue = np.ones(len(tets), dtype=np.int32)
    return FemMesh(
        nodes=nodes,
        tets=tets,
        tissue=tissue,
        tissue_labels=("vagus_left",),
        unit="mm",
    )


def test_source_count_matches_spacing() -> None:
    fem = _build_tube_fem(length_mm=100.0, n_slabs=50)
    pos = vagus_sources(fem, "vagus_left", spacing_mm=5.0)
    # 100 mm / 5 mm spacing → ~20 sources (allow ±1 due to inclusive/exclusive bins)
    assert 18 <= len(pos) <= 21
    # Z values monotonic & within tube span
    assert np.all(pos[:, 2] >= 0.0)
    assert np.all(pos[:, 2] <= 100.5)


def test_source_unknown_tissue_raises() -> None:
    fem = _build_tube_fem()
    with pytest.raises(ValueError, match="not in FEM"):
        vagus_sources(fem, "skin", spacing_mm=5.0)


def test_source_empty_tissue_raises() -> None:
    fem = _build_tube_fem()
    fem_empty = FemMesh(
        nodes=fem.nodes,
        tets=fem.tets,
        tissue=np.full(len(fem.tets), 1, dtype=np.int32),
        tissue_labels=("vagus_left", "skin"),  # skin declared but no tets use it
    )
    # tissue_label exists in labels but no tets carry that id → should raise
    with pytest.raises(ValueError, match="no tets"):
        vagus_sources(fem_empty, "skin", spacing_mm=5.0)


def test_source_single_slab() -> None:
    fem = _build_tube_fem(length_mm=10.0, n_slabs=5)
    pos = vagus_sources(fem, "vagus_left", spacing_mm=20.0)
    # Tube only 10 mm long → at most 1 source
    assert len(pos) <= 1


def test_source_set_is_deterministic() -> None:
    """Sampling the same FEM twice must produce byte-identical source positions.

    The forward solve, the sensitivity sweep, the cross-modality coupling and
    the cluster reduce step ALL re-derive the source set from the FEM via
    :func:`vagus_sources`; any non-determinism here would break the
    one-source-set-shared-by-all-modalities invariant of the dual-modality
    pipeline (and would silently corrupt the cluster reduce, where the chunks
    are produced in a different process from the reduce).
    """
    fem = _build_tube_fem(length_mm=100.0, n_slabs=50)
    pos_a = vagus_sources(fem, "vagus_left", spacing_mm=5.0)
    pos_b = vagus_sources(fem, "vagus_left", spacing_mm=5.0)
    np.testing.assert_array_equal(pos_a, pos_b)


def test_source_set_independent_of_tet_order() -> None:
    """Permuting the tet ordering must not change the resulting sources.

    ``vagus_sources`` averages tet centroids per Z-slab — a permutation of
    tet rows changes the order of the input but must not change the per-slab
    means (modulo floating-point summation order). This documents an
    assumption the cluster pipeline relies on: each chunk worker re-derives
    the source set from the FEM, and they must agree across chunks.
    """
    fem = _build_tube_fem(length_mm=80.0, n_slabs=40)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(fem.tets))
    fem_perm = FemMesh(
        nodes=fem.nodes,
        tets=fem.tets[perm],
        tissue=fem.tissue[perm],
        tissue_labels=fem.tissue_labels,
        unit=fem.unit,
    )
    pos_a = vagus_sources(fem, "vagus_left", spacing_mm=5.0)
    pos_b = vagus_sources(fem_perm, "vagus_left", spacing_mm=5.0)
    np.testing.assert_allclose(pos_a, pos_b, rtol=0.0, atol=1e-9)
