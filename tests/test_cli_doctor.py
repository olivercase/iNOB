"""CLI: `inob doctor` — environment checks and exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import inob.cli.doctor as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"

LFS_STUB = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 123\n"


def _write_stl(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_json_output_shape(tmp_path, capsys) -> None:
    cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--json",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data["ok"], bool)
    assert data["checks"]
    for check in data["checks"]:
        assert set(check) == {"name", "status", "detail", "hints"}
        assert check["status"] in (cli_mod.OK, cli_mod.WARN, cli_mod.FAIL)


def test_check_python_passes_on_this_interpreter() -> None:
    check = cli_mod.check_python()
    assert check.status == cli_mod.OK


def test_check_meshes_fails_on_lfs_pointer(tmp_path) -> None:
    _write_stl(tmp_path / "data" / "bone" / "c7.stl", LFS_STUB)
    check = cli_mod.check_meshes(tmp_path)
    assert check.status == cli_mod.FAIL
    assert "Git LFS pointers" in check.detail


def test_check_meshes_ok_for_real_looking_stl(tmp_path) -> None:
    _write_stl(tmp_path / "data" / "bone" / "c7.stl", b"solid c7\n" + b"facet normal 0 0 1\n" * 200)
    check = cli_mod.check_meshes(tmp_path)
    assert check.status == cli_mod.OK
    assert "1 STL files present" in check.detail


def test_check_meshes_fails_with_no_stls(tmp_path) -> None:
    check = cli_mod.check_meshes(tmp_path)
    assert check.status == cli_mod.FAIL
    assert "no .stl files" in check.detail


def test_check_outputs_writable_ok(tmp_path) -> None:
    check = cli_mod.check_outputs_writable(tmp_path)
    assert check.status == cli_mod.OK
    assert (tmp_path / "outputs").is_dir()


def test_main_returns_1_when_a_check_fails(tmp_path, capsys) -> None:
    # No data/ directory at all, so the mesh check fails.
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_main_returns_0_when_only_warnings(tmp_path, monkeypatch, capsys) -> None:
    _write_stl(tmp_path / "data" / "bone" / "c7.stl", b"solid c7\n" + b"facet normal 0 0 1\n" * 200)
    monkeypatch.setattr(
        cli_mod,
        "check_core_deps",
        lambda: cli_mod.Check("Core dependencies", cli_mod.OK, "all present"),
    )
    monkeypatch.setattr(
        cli_mod,
        "check_duneuro",
        lambda: cli_mod.Check("DUNEuro (duneuropy)", cli_mod.WARN, "missing"),
    )
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
