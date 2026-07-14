"""OBJ → STL conversion CLI utility."""
from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from inob_obj2stl import _parse_arguments, convert_obj_to_stl, main


@pytest.fixture
def obj_file(tmp_path: Path) -> Path:
    m = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    p = tmp_path / "mesh.obj"
    m.export(p)
    return p


# ── convert_obj_to_stl ──────────────────────────────────────────────────

def test_convert_obj_to_stl_basic(tmp_path: Path, obj_file: Path) -> None:
    stl_path = tmp_path / "mesh.stl"
    out = convert_obj_to_stl(obj_file, stl_path)
    assert out == stl_path
    assert stl_path.is_file()
    loaded = trimesh.load_mesh(stl_path)
    assert len(loaded.vertices) > 0


def test_convert_obj_to_stl_appends_missing_suffix(tmp_path: Path, obj_file: Path) -> None:
    stl_path = tmp_path / "mesh_no_suffix"
    out = convert_obj_to_stl(obj_file, stl_path)
    assert out.suffix == ".stl"
    assert out.is_file()


def test_convert_obj_to_stl_preserves_stl_suffix_case_insensitively(
    tmp_path: Path, obj_file: Path
) -> None:
    stl_path = tmp_path / "mesh.STL"
    out = convert_obj_to_stl(obj_file, stl_path)
    assert out == stl_path
    assert out.is_file()


def test_convert_obj_to_stl_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        convert_obj_to_stl(tmp_path / "nope.obj", tmp_path / "out.stl")


def test_convert_obj_to_stl_same_path_raises(tmp_path: Path) -> None:
    # Both suffix ".stl" so no suffix rewrite happens and the paths collide.
    m = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    same_path = tmp_path / "mesh.stl"
    m.export(same_path)
    with pytest.raises(ValueError):
        convert_obj_to_stl(same_path, same_path)


# ── _parse_arguments ─────────────────────────────────────────────────────

def test_parse_arguments_both_positional() -> None:
    args = _parse_arguments(["in.obj", "out.stl"])
    assert args.obj_path == Path("in.obj")
    assert args.stl_path == Path("out.stl")


def test_parse_arguments_none_given() -> None:
    args = _parse_arguments([])
    assert args.obj_path is None
    assert args.stl_path is None


# ── main ─────────────────────────────────────────────────────────────────

def test_main_success(tmp_path: Path, obj_file: Path, capsys: pytest.CaptureFixture) -> None:
    stl_path = tmp_path / "out.stl"
    rc = main([str(obj_file), str(stl_path)])
    assert rc == 0
    assert stl_path.is_file()
    out = capsys.readouterr().out
    assert "Converted" in out


def test_main_missing_input_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    rc = main([str(tmp_path / "missing.obj"), str(tmp_path / "out.stl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Error" in err


def test_main_prompts_when_args_missing(
    tmp_path: Path, obj_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stl_path = tmp_path / "out.stl"
    inputs = iter([str(obj_file), str(stl_path)])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    rc = main([])
    assert rc == 0
    assert stl_path.is_file()
