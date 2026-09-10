"""Pipeline orchestrator: stage parsing, skip logic, failed markers."""

from __future__ import annotations

from pathlib import Path

import pytest

from inob.cli.pipeline import (
    ALL_STAGES,
    DEFAULT_STAGES,
    STAGES,
    Stage,
    _parse_stages,
    run_pipeline,
)
from inob.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_parse_stages_all() -> None:
    assert _parse_stages("all") == list(ALL_STAGES)


def test_parse_stages_omitted_leaves_figures_out() -> None:
    assert _parse_stages(None) == list(DEFAULT_STAGES)


def test_parse_stages_subset() -> None:
    assert _parse_stages("geom,fem") == ["geom", "fem"]


def test_parse_stages_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        _parse_stages("geom,nope")


def test_run_pipeline_skips_when_outputs_exist(tmp_path: Path, monkeypatch) -> None:
    """If primary outputs exist, the stage's run_fn must not be called."""
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml", project_root=tmp_path)
    # Make every output exist by touching its primary path.
    for stage in STAGES.values():
        for f in stage.output_paths:
            p = getattr(cfg.outputs, f)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

    calls: list[str] = []
    fake_stages = {
        name: Stage(s.name, s.description, s.output_paths, lambda c, n=name: calls.append(n))
        for name, s in STAGES.items()
    }
    monkeypatch.setattr("inob.cli.pipeline.STAGES", fake_stages)

    statuses = run_pipeline(cfg, stages=list(ALL_STAGES))
    assert calls == [], f"unexpected stage runs: {calls}"
    assert all(v == "skipped" for v in statuses.values())


def test_run_pipeline_force_runs_everything(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml", project_root=tmp_path)
    for stage in STAGES.values():
        for f in stage.output_paths:
            p = getattr(cfg.outputs, f)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

    calls: list[str] = []
    fake_stages = {
        name: Stage(s.name, s.description, s.output_paths, lambda c, n=name: calls.append(n))
        for name, s in STAGES.items()
    }
    monkeypatch.setattr("inob.cli.pipeline.STAGES", fake_stages)

    run_pipeline(cfg, stages=list(ALL_STAGES), force=True)
    assert calls == list(ALL_STAGES)


def test_run_pipeline_failed_marker(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml", project_root=tmp_path)

    def raises(_cfg):
        raise RuntimeError("boom")

    fake_stage = Stage("geom", "test", ("geometry_mat",), raises)
    monkeypatch.setattr("inob.cli.pipeline.STAGES", {"geom": fake_stage})
    with pytest.raises(RuntimeError):
        run_pipeline(cfg, stages=["geom"])
    marker = cfg.outputs.geometry_mat.parent / ".geom.FAILED"
    assert marker.exists()
    assert "boom" in marker.read_text()
