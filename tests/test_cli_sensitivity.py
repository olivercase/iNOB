"""CLI: sensitivity sweep argparse + wiring (_run_modality, main)."""

from __future__ import annotations

from pathlib import Path

import pytest

import inob.cli.sensitivity as cli_mod
from inob.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def _cfg(tmp_path: Path):
    return load_config(TINY_CFG, project_root=tmp_path)


def test_run_modality_raises_if_baseline_missing(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(FileNotFoundError, match="baseline leadfield missing"):
        cli_mod._run_modality(cfg, "meg", skip_unit=True)


def test_run_modality_skips_unit_perturbation(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    baseline = cfg.outputs.forward_npz
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.touch()

    from dataclasses import replace

    cfg = replace(
        cfg,
        sensitivity=replace(
            cfg.sensitivity,
            perturbations=(0.5, 1.0, 1.5),
        ),
    )

    seen = {}

    def fake_sweep(cfg_run, *, baseline_path, forward_fn, out_path, modality):
        seen["perturbations"] = cfg_run.sensitivity.perturbations
        seen["modality"] = modality
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}")

    monkeypatch.setattr(cli_mod, "sweep", fake_sweep)
    out = cli_mod._run_modality(cfg, "meg", skip_unit=True)
    assert 1.0 not in seen["perturbations"]
    assert seen["modality"] == "meg"
    assert out == cfg.outputs.sensitivity_dir / "sensitivity_meg.json"


def test_run_modality_keeps_unit_when_not_skipped(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    baseline = cfg.outputs.forward_eeg_npz
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.touch()

    seen = {}

    def fake_sweep(cfg_run, *, baseline_path, forward_fn, out_path, modality):
        seen["perturbations"] = cfg_run.sensitivity.perturbations

    monkeypatch.setattr(cli_mod, "sweep", fake_sweep)
    cli_mod._run_modality(cfg, "eeg", skip_unit=False)
    assert 1.0 in seen["perturbations"]


def test_main_runs_both_modalities_by_default(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        cli_mod,
        "_run_modality",
        lambda cfg, m, *, skip_unit: (calls.append(m), Path("x"))[1],
    )
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls == ["meg", "eeg"]


def test_main_single_modality(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        cli_mod,
        "_run_modality",
        lambda cfg, m, *, skip_unit: (calls.append(m), Path("x"))[1],
    )
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--modality",
            "eeg",
        ]
    )
    assert rc == 0
    assert calls == ["eeg"]


def test_main_plot_only_skips_solves(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        cli_mod,
        "_run_modality",
        lambda cfg, m, *, skip_unit: calls.append(m),
    )
    render_calls = {}
    import inob.viz.sensitivity_plot as sp_mod

    monkeypatch.setattr(sp_mod, "render_sensitivity", lambda cfg, **kw: render_calls.update(kw))
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--plot-only",
        ]
    )
    assert rc == 0
    assert calls == []
    assert render_calls


def test_main_plot_renders_figure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "_run_modality",
        lambda cfg, m, *, skip_unit: Path("x"),
    )
    render_calls = {}
    import inob.viz.sensitivity_plot as sp_mod

    monkeypatch.setattr(sp_mod, "render_sensitivity", lambda cfg, **kw: render_calls.update(kw))
    out = tmp_path / "cmp.png"
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--plot",
            "--out",
            str(out),
            "--dpi",
            "72",
        ]
    )
    assert rc == 0
    assert render_calls["out_path"] == out
    assert render_calls["dpi"] == 72
