"""STL loader tests: friendly errors on missing files, unit guard."""
from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from inob.io.stl import (
    STLLoadError,
    concat_stls,
    load_first_stl,
    load_stl,
    load_stl_glob,
)


def _write_box(path: Path, extents=(100.0, 100.0, 100.0)) -> None:
    trimesh.creation.box(extents=extents).export(str(path))


def test_load_existing_stl(tmp_path: Path) -> None:
    p = tmp_path / "cube.stl"
    _write_box(p)
    m = load_stl(p)
    assert isinstance(m, trimesh.Trimesh)
    assert m.vertices.size > 0


def test_missing_file_friendly_error(tmp_path: Path) -> None:
    with pytest.raises(STLLoadError, match="not found"):
        load_stl(tmp_path / "nope.stl")


def test_glob_empty_match_raises(tmp_path: Path) -> None:
    with pytest.raises(STLLoadError, match="no STL files"):
        load_stl_glob(str(tmp_path / "*left*.stl"))


def test_glob_returns_sorted(tmp_path: Path) -> None:
    for n in ["c.stl", "a.stl", "b.stl"]:
        _write_box(tmp_path / n)
    paths = load_stl_glob(str(tmp_path / "*.stl"))
    assert [p.name for p in paths] == ["a.stl", "b.stl", "c.stl"]


def test_load_first_stl(tmp_path: Path) -> None:
    _write_box(tmp_path / "a_left.stl")
    m = load_first_stl(str(tmp_path / "*left*.stl"))
    assert isinstance(m, trimesh.Trimesh)


def test_unit_guard_rejects_metres(tmp_path: Path) -> None:
    p = tmp_path / "tiny.stl"
    _write_box(p, extents=(0.05, 0.05, 0.05))   # 0.05 mm — looks like metres input
    with pytest.raises(STLLoadError, match="implausible for mm"):
        load_stl(p)


def test_unit_guard_skipped_on_request(tmp_path: Path) -> None:
    p = tmp_path / "tiny.stl"
    _write_box(p, extents=(0.05, 0.05, 0.05))
    m = load_stl(p, check_units_mm=False)
    assert isinstance(m, trimesh.Trimesh)


def test_concat_multiple(tmp_path: Path) -> None:
    for n in ["a.stl", "b.stl"]:
        _write_box(tmp_path / n)
    m = concat_stls(load_stl_glob(str(tmp_path / "*.stl")))
    assert m.vertices.shape[0] >= 16   # both cubes' verts merged


def test_concat_empty_raises() -> None:
    with pytest.raises(STLLoadError, match="empty path list"):
        concat_stls([])
