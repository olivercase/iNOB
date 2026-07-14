"""Project-root resolution and output-path helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from inob.paths import find_project_root, resolve_path, resolve_project_root


# ── find_project_root ───────────────────────────────────────────────────

def test_find_project_root_finds_pyproject_toml(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    sub = tmp_path / "a" / "b" / "c"
    sub.mkdir(parents=True)
    assert find_project_root(sub) == tmp_path.resolve()


def test_find_project_root_finds_configs_default_yaml(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text("x: 1\n")
    sub = tmp_path / "nested"
    sub.mkdir()
    assert find_project_root(sub) == tmp_path.resolve()


def test_find_project_root_starts_at_start_itself(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    assert find_project_root(tmp_path) == tmp_path.resolve()


def test_find_project_root_falls_back_to_start_when_no_marker(tmp_path: Path) -> None:
    # An isolated directory tree with no markers anywhere above it won't
    # exist in practice (repo root has pyproject.toml), so instead verify
    # the *nearest* marker wins over a further one.
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "pyproject.toml").write_text("[project]\n")
    leaf = inner / "leaf"
    leaf.mkdir()
    assert find_project_root(leaf) == inner.resolve()


def test_find_project_root_defaults_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    monkeypatch.chdir(tmp_path)
    assert find_project_root() == tmp_path.resolve()


# ── resolve_project_root ────────────────────────────────────────────────

def test_resolve_project_root_explicit_argument_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    monkeypatch.setenv("INOB_ROOT", str(tmp_path / "envdir"))
    assert resolve_project_root(explicit) == explicit.resolve()


def test_resolve_project_root_explicit_string(tmp_path: Path) -> None:
    assert resolve_project_root(str(tmp_path)) == tmp_path.resolve()


def test_resolve_project_root_expands_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INOB_ROOT", raising=False)
    out = resolve_project_root("~")
    assert out == Path("~").expanduser().resolve()


def test_resolve_project_root_env_var_used_when_no_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_dir = tmp_path / "from_env"
    env_dir.mkdir()
    monkeypatch.setenv("INOB_ROOT", str(env_dir))
    assert resolve_project_root(None) == env_dir.resolve()


def test_resolve_project_root_falls_back_to_marker_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INOB_ROOT", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    monkeypatch.chdir(tmp_path)
    assert resolve_project_root(None) == tmp_path.resolve()


def test_resolve_project_root_none_explicit_falsy_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Empty string is falsy, so it should fall through to env/marker search,
    # not be treated as an explicit path.
    monkeypatch.setenv("INOB_ROOT", str(tmp_path))
    assert resolve_project_root("") == tmp_path.resolve()


# ── resolve_path ─────────────────────────────────────────────────────────

def test_resolve_path_relative_joins_root(tmp_path: Path) -> None:
    out = resolve_path("data/foo.stl", tmp_path)
    assert out == (tmp_path / "data/foo.stl").resolve()


def test_resolve_path_absolute_passthrough(tmp_path: Path) -> None:
    abs_p = tmp_path / "somewhere" / "file.stl"
    out = resolve_path(abs_p, tmp_path / "unrelated_root")
    assert out == abs_p


def test_resolve_path_accepts_path_object(tmp_path: Path) -> None:
    out = resolve_path(Path("sub/file.txt"), tmp_path)
    assert out == (tmp_path / "sub/file.txt").resolve()


def test_resolve_path_expands_user(tmp_path: Path) -> None:
    out = resolve_path("~/x.txt", tmp_path)
    assert out == Path("~/x.txt").expanduser()
