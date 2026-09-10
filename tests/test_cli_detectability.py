"""CLI: detectability argparse targets + wiring."""

from __future__ import annotations

from pathlib import Path

import inob.cli.detectability as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def _patch_all(monkeypatch, calls):
    monkeypatch.setattr(
        cli_mod,
        "render_surface_topoplots",
        lambda cfg, **kw: calls.setdefault("surface", kw),
    )
    monkeypatch.setattr(
        cli_mod,
        "render_detectability",
        lambda cfg, **kw: calls.setdefault("detect", kw),
    )
    monkeypatch.setattr(
        cli_mod,
        "detectability_summary",
        lambda cfg, **kw: {"summary": True, **kw},
    )


def test_main_target_all_calls_both(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch_all(monkeypatch, calls)
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert "surface" in calls and "detect" in calls


def test_main_target_surface_only(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch_all(monkeypatch, calls)
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--target",
            "surface",
        ]
    )
    assert rc == 0
    assert "surface" in calls
    assert "detect" not in calls


def test_main_target_detect_only(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch_all(monkeypatch, calls)
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--target",
            "detect",
        ]
    )
    assert rc == 0
    assert "detect" in calls
    assert "surface" not in calls


def test_main_print_summary_writes_json(tmp_path, monkeypatch, capsys) -> None:
    calls = {}
    _patch_all(monkeypatch, calls)
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--target",
            "surface",
            "--print-summary",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert '"summary": true' in out


def test_main_forwards_thresholds(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch_all(monkeypatch, calls)
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--target",
            "detect",
            "--snr-threshold",
            "5",
            "--max-trials",
            "100",
            "--source-idx",
            "2",
        ]
    )
    assert rc == 0
    assert calls["detect"]["snr_threshold"] == 5.0
    assert calls["detect"]["max_trials"] == 100
    assert calls["detect"]["source_idx"] == 2
