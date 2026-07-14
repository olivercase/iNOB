"""CLI: visualise argparse targets + wiring."""
from __future__ import annotations

from pathlib import Path

import inob.cli.visualise as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def _patch(monkeypatch, calls):
    monkeypatch.setattr(
        cli_mod, "render_geometry", lambda cfg, **kw: calls.setdefault("geom", kw),
    )
    monkeypatch.setattr(
        cli_mod, "render_fem", lambda cfg, **kw: calls.setdefault("fem", kw),
    )


def test_main_target_all_calls_both(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert "geom" in calls and "fem" in calls
    assert calls["geom"]["with_sensors"] is True
    assert calls["fem"]["interactive"] is False


def test_main_target_geom_only(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path), "--target", "geom",
    ])
    assert rc == 0
    assert "geom" in calls and "fem" not in calls


def test_main_target_fem_only(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path), "--target", "fem",
    ])
    assert rc == 0
    assert "fem" in calls and "geom" not in calls


def test_main_no_sensors_flag(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--target", "geom", "--no-sensors", "--dpi", "72",
    ])
    assert rc == 0
    assert calls["geom"]["with_sensors"] is False
    assert calls["geom"]["dpi"] == 72
