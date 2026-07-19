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


def test_parse_stages_empty_string_is_rejected() -> None:
    """`--stages ""` (e.g. an unset shell var) must NOT silently run everything."""
    import pytest

    from inob.cli.pipeline import _parse_stages
    with pytest.raises(ValueError, match="no stages selected"):
        _parse_stages("")
    with pytest.raises(ValueError, match="no stages selected"):
        _parse_stages(" , ,")


def test_parse_stages_none_and_all_run_everything() -> None:
    from inob.cli.pipeline import ALL_STAGES, _parse_stages
    assert _parse_stages(None) == list(ALL_STAGES)
    assert _parse_stages("all") == list(ALL_STAGES)


def test_main_empty_stages_returns_2_not_full_run(tmp_path) -> None:
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--stages", "",
    ])
    assert rc == 2


def test_stage_helpers_delegate_to_library_functions(tmp_path, monkeypatch) -> None:
    import inob.fem.cgal_builder as fem_mod
    import inob.forward.local as local_mod
    import inob.geometry.builder as geom_mod
    import inob.sensors.triaxial as sensors_mod

    calls = {}
    monkeypatch.setattr(geom_mod, "build_geometry", lambda cfg: calls.setdefault("geom", cfg))
    monkeypatch.setattr(fem_mod, "build_fem", lambda cfg: calls.setdefault("fem", cfg))
    monkeypatch.setattr(sensors_mod, "generate_sensor_array", lambda cfg: calls.setdefault("sensors", cfg))
    # Forward now defaults to the local multi-core orchestrator.
    monkeypatch.setattr(local_mod, "run_forward_local", lambda cfg: calls.setdefault("forward", cfg))

    cfg = object()
    cli_mod._stage_geom(cfg)
    cli_mod._stage_fem(cfg)
    cli_mod._stage_sensors(cfg)
    cli_mod._stage_forward(cfg)
    assert calls == {"geom": cfg, "fem": cfg, "sensors": cfg, "forward": cfg}


# --- prerequisite checking -------------------------------------------------

class _Recorder:
    """Stand-in for the module logger; keeps formatted messages."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def _record(self, msg, *args):
        self.messages.append(msg % args if args else msg)

    info = error = warning = debug = _record


def _cfg(tmp_path):
    from inob.config import load_config
    return load_config(TINY_CFG, project_root=tmp_path)


def test_missing_prerequisites_forward_reports_fem_and_sensors(tmp_path) -> None:
    unmet = cli_mod.missing_prerequisites(_cfg(tmp_path), ["forward"])
    assert unmet == [("forward", "fem"), ("forward", "sensors")]


def test_missing_prerequisites_satisfied_within_same_run(tmp_path) -> None:
    stages = ["geom", "fem", "sensors", "forward"]
    assert cli_mod.missing_prerequisites(_cfg(tmp_path), stages) == []


def test_main_returns_2_when_prerequisites_unmet(tmp_path) -> None:
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--stages", "forward",
    ])
    assert rc == 2


def test_main_suggests_transitively_complete_stage_list(tmp_path, monkeypatch) -> None:
    rec = _Recorder()
    monkeypatch.setattr(cli_mod, "logger", rec)
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--stages", "forward",
    ])
    assert rc == 2
    assert "build it first: inob run --stages geom,fem,sensors" in rec.messages
