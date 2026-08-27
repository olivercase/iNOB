"""CLI: `inob status` — on-disk state reporting and the next-step hint."""
from __future__ import annotations

import json
from pathlib import Path

import inob.cli.status as cli_mod
from inob.cli.pipeline import DEFAULT_STAGES
from inob.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"

# Relative to project_root, per configs/tiny_test.yaml. Geometry/FEM/sensors are
# shared across every source-target, so they stay untagged; only the forward
# leadfield carries the target slug.
STAGE_FILES: dict[str, tuple[str, ...]] = {
    "geom": ("outputs/geometry/geometry.mat",),
    "fem": ("outputs/fem/fem.mat",),
    "sensors": ("outputs/sensors/sensor_array.mat",),
    "forward": ("outputs/forward/duneuro_leadfield_vagus.npz",),
    "viz": ("outputs/geometry/geometry.png", "outputs/fem/fem.png"),
}


def _build(root: Path, *stages: str) -> None:
    for stage in stages:
        for rel in STAGE_FILES[stage]:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")


def _cfg(tmp_path: Path):
    return load_config(TINY_CFG, project_root=tmp_path)


def test_main_returns_zero(tmp_path, capsys) -> None:
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert "Pipeline" in capsys.readouterr().out


def test_json_output_shape(tmp_path, capsys) -> None:
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path), "--json",
    ])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data) >= {"project_root", "config", "stages", "optional", "next"}
    assert data["project_root"] == str(tmp_path)
    assert [s["name"] for s in data["stages"]] == list(DEFAULT_STAGES)
    # Figures are an optional artefact, not a pipeline stage: a run that
    # drew nothing is complete, not four-fifths done.
    assert "viz" not in [s["name"] for s in data["stages"]]
    assert "figures" in [e["name"] for e in data["optional"]]


def test_nothing_built_suggests_inob_run(tmp_path) -> None:
    stages, _ = cli_mod.collect(_cfg(tmp_path))
    assert all(s.state == "missing" for s in stages)
    assert cli_mod.next_step(stages).startswith("inob run")


def test_only_last_stage_missing_suggests_that_stage(tmp_path) -> None:
    _build(tmp_path, "geom", "fem", "sensors")
    stages, _ = cli_mod.collect(_cfg(tmp_path))
    assert cli_mod.next_step(stages) == "inob forward"


def test_undrawn_figures_do_not_make_a_finished_run_look_unfinished(tmp_path) -> None:
    """The whole point of opt-in figures: not drawing them is not a missing step."""
    _build(tmp_path, "geom", "fem", "sensors", "forward")
    stages, extras = cli_mod.collect(_cfg(tmp_path))
    assert cli_mod.next_step(stages) is None
    figures = next(e for e in extras if e.name == "figures")
    assert figures.state == "missing"
    assert figures.command == "inob run --with-viz"


def test_failed_marker_reports_failed_and_suggests_force(tmp_path) -> None:
    _build(tmp_path, "geom")
    (tmp_path / "outputs" / "geometry" / ".geom.FAILED").write_text("boom\n")
    stages, _ = cli_mod.collect(_cfg(tmp_path))
    geom = next(s for s in stages if s.name == "geom")
    assert geom.state == "failed"
    assert cli_mod.next_step(stages).startswith("inob build-geom --force")
