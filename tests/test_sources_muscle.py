"""Tests for inob.sources.muscle: volume-fill sampling and per-muscle fibre axes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from inob.io.hdf5 import FemMesh
from inob.sources.muscle import (
    muscle_source_orientations,
    muscle_source_stl_assignment,
    muscle_sources,
)

_TET_OFFSETS = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
])


def _tet_nodes(centre) -> np.ndarray:
    """Nodes of a small tet whose centroid sits at ``centre``."""
    return np.asarray(centre) + _TET_OFFSETS - _TET_OFFSETS.mean(axis=0)


def _fem_from_centroids(centroids: list, *, tissue_id: int = 1) -> FemMesh:
    """One tet per given centroid, all tagged ``tissue_id``, plus a bone/skin filler."""
    nodes: list[np.ndarray] = []
    tets: list[list[int]] = []
    tissue: list[int] = []
    for c in centroids:
        base = len(nodes)
        nodes.extend(_tet_nodes(c))
        tets.append([base, base + 1, base + 2, base + 3])
        tissue.append(tissue_id)
    for filler_id, filler_centre in ((2, [1000.0, 0.0, 0.0]), (3, [2000.0, 0.0, 0.0])):
        base = len(nodes)
        nodes.extend(_tet_nodes(filler_centre))
        tets.append([base, base + 1, base + 2, base + 3])
        tissue.append(filler_id)
    return FemMesh(
        np.asarray(nodes), np.asarray(tets, dtype=np.int32),
        np.asarray(tissue, dtype=np.int32), ("muscle", "bone", "skin"), "mm",
    )


def _write_box_stl(path: Path, extents, centre) -> None:
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(centre)
    m.export(path)


@pytest.fixture
def two_muscle_stls(tmp_path: Path) -> Path:
    """Two muscles at well-separated locations, elongated along different axes."""
    d = tmp_path / "muscle"
    d.mkdir()
    _write_box_stl(d / "a_long_z.stl", (4.0, 4.0, 80.0), (0.0, 0.0, 0.0))
    _write_box_stl(d / "b_long_x.stl", (80.0, 4.0, 4.0), (500.0, 0.0, 0.0))
    return d


def test_muscle_sources_volume_fill_returns_real_muscle_points() -> None:
    centroids = [[0.0, 0.0, z] for z in (-30.0, -10.0, 10.0, 30.0)]
    fem = _fem_from_centroids(centroids)
    pos = muscle_sources(fem, spacing_mm=15.0)
    assert pos.shape[1] == 3
    assert 1 <= len(pos) <= len(centroids)
    # Every returned position must be one of the real muscle tet centroids,
    # never an interpolated/averaged point outside the tissue.
    for p in pos:
        assert any(np.allclose(p, c) for c in centroids)


def test_muscle_sources_raises_without_muscle_tissue() -> None:
    fem = FemMesh(
        np.eye(4, 3), np.array([[0, 1, 2, 3]], dtype=np.int32),
        np.array([1], dtype=np.int32), ("bone",), "mm",
    )
    with pytest.raises(ValueError, match="muscle"):
        muscle_sources(fem, spacing_mm=5.0)


def test_muscle_source_orientations_follow_each_muscles_own_axis(
    two_muscle_stls: Path,
) -> None:
    """A source near the Z-elongated box gets a Z axis; near the X-elongated
    box, an X axis — not one global axis shared across both muscles."""
    fem = _fem_from_centroids(
        [[0.0, 0.0, -30.0], [0.0, 0.0, 30.0],       # near a_long_z
         [470.0, 0.0, 0.0], [530.0, 0.0, 0.0]],     # near b_long_x
    )
    pos = muscle_sources(fem, spacing_mm=200.0)
    orient = muscle_source_orientations(fem, pos, muscle_dir=two_muscle_stls)
    for p, o in zip(pos, orient, strict=True):
        dominant_axis = int(np.argmax(np.abs(o)))
        expected_axis = 2 if p[2] != 0.0 or abs(p[0]) < 100.0 else 0
        # Sources sit either near (0,0,±30) -> Z-axis muscle, or near
        # (470/530, 0, 0) -> X-axis muscle.
        if abs(p[0]) < 100.0:
            assert dominant_axis == 2
        else:
            assert dominant_axis == 0


def test_muscle_source_stl_assignment_matches_orientation_rule(
    two_muscle_stls: Path,
) -> None:
    """The exposed nearest-STL index must agree with the axis
    muscle_source_orientations actually applied to each source."""
    fem = _fem_from_centroids(
        [[0.0, 0.0, -30.0], [470.0, 0.0, 0.0]],
    )
    pos = muscle_sources(fem, spacing_mm=200.0)
    orient = muscle_source_orientations(fem, pos, muscle_dir=two_muscle_stls)
    idx = muscle_source_stl_assignment(pos, muscle_dir=two_muscle_stls)

    from inob.sources.muscle import _muscle_stl_axes
    axes_arr, _centres = _muscle_stl_axes(two_muscle_stls)
    for i, o in zip(idx, orient, strict=True):
        np.testing.assert_allclose(axes_arr[i], o)
