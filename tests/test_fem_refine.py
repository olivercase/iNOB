"""Local TetGen refinement of one tissue (inob.fem.refine)."""

from __future__ import annotations

import numpy as np
import pytest

from inob.fem.refine import find_tetgen, interface_faces, refine_tissue, vertices_interior_to
from inob.io.hdf5 import FemMesh


def _tetgen_or_skip() -> str:
    try:
        exe = find_tetgen()
    except FileNotFoundError:
        pytest.skip("TetGen not available")
    import os

    if not os.path.exists(exe):
        pytest.skip("TetGen not available")
    return exe


def _two_tissue_block() -> FemMesh:
    """A 4 x 4 x 8 block of cubes, each cut into six tets; a 2 x 2 core is tissue 1."""
    nx, ny, nz, h = 4, 4, 8, 6.0
    gx, gy, gz = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]) * h

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    tets, tissue = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [nid(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                # Kuhn subdivision: six tets round the main diagonal c[0]-c[7].
                for a, b in ((1, 3), (3, 2), (2, 6), (6, 4), (4, 5), (5, 1)):
                    tets.append([c[0], c[a], c[b], c[7]])
                    tissue.append(1 if (1 <= i <= 2 and 1 <= j <= 2) else 2)
    return FemMesh(
        nodes=nodes.astype(np.float64),
        tets=np.asarray(tets, dtype=np.int32),
        tissue=np.asarray(tissue, dtype=np.int32),
        tissue_labels=("cord", "body"),
        unit="mm",
    )


def _volumes(fem: FemMesh) -> np.ndarray:
    v = fem.nodes[fem.tets]
    return (
        np.abs(
            np.einsum("ij,ij->i", np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), v[:, 3] - v[:, 0])
        )
        / 6.0
    )


def test_interface_faces_are_the_hull_plus_the_tissue_boundary() -> None:
    fem = _two_tissue_block()
    faces = interface_faces(fem.tets, fem.tissue)
    # Hull: 2 triangles per cube face. Core boundary: four 2 x 8 sides.
    hull = 2 * 2 * (4 * 4 + 4 * 8 + 4 * 8)
    core = 2 * 4 * (2 * 8)
    assert len(faces) == hull + core


def test_refinement_is_local_and_keeps_every_tissue_volume() -> None:
    _tetgen_or_skip()
    fem = _two_tissue_block()
    new = refine_tissue(fem, "cord", max_volume_mm3=2.0)
    before, after = _volumes(fem), _volumes(new)
    for tid in (1, 2):
        assert after[new.tissue == tid].sum() == pytest.approx(
            before[fem.tissue == tid].sum(), rel=1e-9
        )
    # TetGen treats the bound as a target for the elements it was given: the
    # quality pass can leave a few a little over it.
    assert np.percentile(after[new.tissue == 1], 95) <= 2.0
    assert after[new.tissue == 1].max() <= 1.5 * 2.0
    # The target is refined far more than the rest.
    growth_cord = (new.tissue == 1).sum() / (fem.tissue == 1).sum()
    growth_body = (new.tissue == 2).sum() / (fem.tissue == 2).sum()
    assert growth_cord > 2 * growth_body
    # The cord now has vertices no other tissue touches. (It had 7 x ... on its
    # axis before; the count must grow.)
    assert vertices_interior_to(new, "cord").sum() > vertices_interior_to(fem, "cord").sum()


def test_unknown_tissue_and_bad_volume_are_refused() -> None:
    fem = _two_tissue_block()
    with pytest.raises(ValueError):
        refine_tissue(fem, "nerve", max_volume_mm3=1.0)
    with pytest.raises(ValueError):
        refine_tissue(fem, "cord", max_volume_mm3=0.0)
