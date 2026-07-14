"""CLI: build_geom argparse + wiring."""
from __future__ import annotations

from pathlib import Path

import inob.cli.build_geom as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def test_main_check_only_calls_check_existing(tmp_path, monkeypatch) -> None:
    calls = {}

    def fake_check_existing(cfg):
        calls["cfg"] = cfg
        return True

    monkeypatch.setattr(cli_mod, "check_existing", fake_check_existing)
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path), "--check-only",
    ])
    assert rc == 0
    assert "cfg" in calls


def test_main_check_only_failure_returns_1(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli_mod, "check_existing", lambda cfg: False)
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path), "--check-only",
    ])
    assert rc == 1


def test_main_build_calls_build_geometry_with_flags(tmp_path, monkeypatch) -> None:
    calls = {}

    def fake_build_geometry(cfg, force_shrinkwrap=False, only_compartments=None):
        calls["force_shrinkwrap"] = force_shrinkwrap
        calls["only_compartments"] = only_compartments
        return None

    monkeypatch.setattr(cli_mod, "build_geometry", fake_build_geometry)
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--shrinkwrap-only", "--only", "mesh_skin, mesh_vagus_left",
    ])
    assert rc == 0
    assert calls["force_shrinkwrap"] is True
    assert calls["only_compartments"] == ("mesh_skin", "mesh_vagus_left")


def test_main_build_default_only_is_none(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "build_geometry",
        lambda cfg, force_shrinkwrap=False, only_compartments=None:
            calls.setdefault("only_compartments", only_compartments),
    )
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls["only_compartments"] is None
