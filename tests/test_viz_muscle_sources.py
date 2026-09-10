"""Tests for inob.viz.muscle_sources: pair-name parsing + render smoke tests."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import trimesh

from inob.config import load_config
from inob.io.hdf5 import CompartmentMesh, FemMesh, Geometry, save_fem, save_geometry
from inob.viz.muscle_sources import (
    _muscle_pair_name,
    render_muscle_source_orientations,
    render_muscle_source_pairs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"

_TET_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


@pytest.mark.parametrize(
    ("filename", "expected_name", "expected_side"),
    [
        ("FJ1573_BP23424_FMA13409_Left sternocleidomastoid.stl", "sternocleidomastoid", "Left"),
        ("FJ1595_BP23713_FMA13408_Right sternocleidomastoid.stl", "sternocleidomastoid", "Right"),
        (
            "FJ1532M_BP21819_FMA32540_Left levator scapulae (mirrored).stl",
            "levator scapulae",
            "Left",
        ),
        (
            "FJ1557M_BP23075_FMA46288_Inferior oblique part of right longus colli (mirrored).stl",
            "Inferior oblique part of longus colli",
            "Right",
        ),
    ],
)
def test_muscle_pair_name_parses_side_and_canonical_name(
    filename: str,
    expected_name: str,
    expected_side: str,
) -> None:
    name, side = _muscle_pair_name(Path(filename))
    assert name == expected_name
    assert side == expected_side


def test_muscle_pair_name_pairs_left_and_right_the_same() -> None:
    """The whole point of the pairing figure: L and R variants of one muscle
    must resolve to the same canonical name so they share a colour."""
    left, _ = _muscle_pair_name(Path("x_y_z_Left omohyoid.stl"))
    right, _ = _muscle_pair_name(Path("x_y_z_Right omohyoid.stl"))
    assert left == right


def _tet_nodes(centre) -> np.ndarray:
    return np.asarray(centre) + _TET_OFFSETS - _TET_OFFSETS.mean(axis=0)


def _write_box_stl(path: Path, extents, centre) -> None:
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(centre)
    m.export(path)


@pytest.fixture
def muscle_cfg(tmp_path: Path):
    """A tiny on-disk pipeline: FEM with 2 muscle regions + 2 muscle STLs.

    Deliberately builds no geometry.mat, exercising the "no backdrop" path —
    the figures must still render without a built geometry compartment.
    """
    muscle_dir = tmp_path / "muscle"
    muscle_dir.mkdir()
    _write_box_stl(muscle_dir / "a_long_z.stl", (4.0, 4.0, 80.0), (0.0, 0.0, 0.0))
    _write_box_stl(muscle_dir / "b_long_x.stl", (80.0, 4.0, 4.0), (200.0, 0.0, 0.0))

    cfg = load_config(
        TINY_CFG,
        project_root=tmp_path,
        overrides=[f"data.muscle_dir={muscle_dir}"],
    )

    centroids = [[0.0, 0.0, -30.0], [0.0, 0.0, 30.0], [170.0, 0.0, 0.0], [230.0, 0.0, 0.0]]
    nodes, tets, tissue = [], [], []
    for c in centroids:
        base = len(nodes)
        nodes.extend(_tet_nodes(c))
        tets.append([base, base + 1, base + 2, base + 3])
        tissue.append(1)
    for filler_id, filler_centre in ((2, [1000.0, 0.0, 0.0]), (3, [2000.0, 0.0, 0.0])):
        base = len(nodes)
        nodes.extend(_tet_nodes(filler_centre))
        tets.append([base, base + 1, base + 2, base + 3])
        tissue.append(filler_id)
    fem = FemMesh(
        np.asarray(nodes),
        np.asarray(tets, dtype=np.int32),
        np.asarray(tissue, dtype=np.int32),
        ("muscle", "bone", "skin"),
        "mm",
    )
    save_fem(cfg.outputs.fem_mat, fem)
    return cfg


def test_render_muscle_source_pairs_smoke(muscle_cfg, tmp_path: Path) -> None:
    """The pairing figure needs only the raw muscle STLs, not a FEM."""
    out = tmp_path / "pairs.png"
    result = render_muscle_source_pairs(muscle_cfg, out_path=out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0


def test_render_muscle_source_orientations_smoke(muscle_cfg, tmp_path: Path) -> None:
    out = tmp_path / "orientations.png"
    result = render_muscle_source_orientations(muscle_cfg, spacing_mm=200.0, out_path=out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0


def test_render_muscle_source_pairs_with_skin_backdrop(muscle_cfg, tmp_path: Path) -> None:
    """Also exercise the geometry-backdrop path (mesh_skin compartment present)."""
    box = trimesh.creation.box(extents=(300.0, 200.0, 400.0))
    geom = Geometry(
        compartments={
            "mesh_skin": CompartmentMesh(
                name="mesh_skin",
                vertices=np.asarray(box.vertices, dtype=np.float64),
                faces=np.asarray(box.faces, dtype=np.int64),
            ),
        }
    )
    save_geometry(muscle_cfg.outputs.geometry_mat, geom)
    out = tmp_path / "pairs_with_backdrop.png"
    render_muscle_source_pairs(muscle_cfg, out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_render_muscle_source_pairs_raises_with_no_muscle_stls(tmp_path: Path) -> None:
    empty_muscle_dir = tmp_path / "empty_muscle"
    empty_muscle_dir.mkdir()
    cfg = load_config(
        TINY_CFG,
        project_root=tmp_path,
        overrides=[f"data.muscle_dir={empty_muscle_dir}"],
    )
    with pytest.raises(ValueError, match="no muscle STLs"):
        render_muscle_source_pairs(cfg, out_path=tmp_path / "unused.png")


def test_render_muscle_source_orientations_raises_without_muscle_fem(
    tmp_path: Path,
) -> None:
    muscle_dir = tmp_path / "muscle"
    muscle_dir.mkdir()
    _write_box_stl(muscle_dir / "a.stl", (4.0, 4.0, 80.0), (0.0, 0.0, 0.0))
    cfg = load_config(
        TINY_CFG,
        project_root=tmp_path,
        overrides=[f"data.muscle_dir={muscle_dir}"],
    )
    fem = FemMesh(
        np.eye(4, 3),
        np.array([[0, 1, 2, 3]], dtype=np.int32),
        np.array([1], dtype=np.int32),
        ("bone",),
        "mm",
    )
    save_fem(cfg.outputs.fem_mat, fem)
    with pytest.raises(ValueError, match="muscle"):
        render_muscle_source_orientations(cfg, out_path=tmp_path / "unused.png")
