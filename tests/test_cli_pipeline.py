"""CLI: pipeline main() argparse wiring (run_pipeline itself is covered by
test_pipeline_orchestrator.py)."""
from __future__ import annotations

from pathlib import Path

import inob.cli.pipeline as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def test_main_default_runs_all_stages(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "run_pipeline",
        lambda cfg, *, stages, force=False: (calls.update(stages=stages, force=force), {})[1],
    )
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls["stages"] == list(cli_mod.ALL_STAGES)
    assert calls["force"] is False


def test_main_stages_subset(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "run_pipeline",
        lambda cfg, *, stages, force=False: (calls.update(stages=stages), {})[1],
    )
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--stages", "geom,fem",
    ])
    assert rc == 0
    assert calls["stages"] == ["geom", "fem"]


def test_main_force_flag(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "run_pipeline",
        lambda cfg, *, stages, force=False: (calls.update(force=force), {})[1],
    )
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path), "--force",
    ])
    assert rc == 0
    assert calls["force"] is True


def test_main_skip_viz_removes_viz_stage(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "run_pipeline",
        lambda cfg, *, stages, force=False: (calls.update(stages=stages), {})[1],
    )
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path), "--skip-viz",
    ])
    assert rc == 0
    assert "viz" not in calls["stages"]


def test_main_invalid_stage_returns_2(tmp_path) -> None:
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--stages", "nope",
    ])
    assert rc == 2


def test_stage_helpers_delegate_to_library_functions(tmp_path, monkeypatch) -> None:
    import inob.geometry.builder as geom_mod
    import inob.fem.cgal_builder as fem_mod
    import inob.sensors.triaxial as sensors_mod
    import inob.forward.solve as solve_mod

    calls = {}
    monkeypatch.setattr(geom_mod, "build_geometry", lambda cfg: calls.setdefault("geom", cfg))
    monkeypatch.setattr(fem_mod, "build_fem", lambda cfg: calls.setdefault("fem", cfg))
    monkeypatch.setattr(sensors_mod, "generate_sensor_array", lambda cfg: calls.setdefault("sensors", cfg))
    monkeypatch.setattr(solve_mod, "run_forward", lambda cfg: calls.setdefault("forward", cfg))

    cfg = object()
    cli_mod._stage_geom(cfg)
    cli_mod._stage_fem(cfg)
    cli_mod._stage_sensors(cfg)
    cli_mod._stage_forward(cfg)
    assert calls == {"geom": cfg, "fem": cfg, "sensors": cfg, "forward": cfg}
