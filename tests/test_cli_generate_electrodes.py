"""CLI: generate_electrodes argparse + wiring."""
from __future__ import annotations

from pathlib import Path

import pytest

import inob.cli.generate_electrodes as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def test_main_defaults_pass_none_overrides(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "generate_electrode_array",
        lambda cfg, **kw: calls.update(kw),
    )
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls == {"rows": None, "cols": None, "contact_pitch_mm": None,
                     "target_level": None}


def test_main_overrides_forwarded(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "generate_electrode_array",
        lambda cfg, **kw: calls.update(kw),
    )
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--rows", "4", "--cols", "8", "--pitch", "2.5", "--level", "c7",
    ])
    assert rc == 0
    assert calls == {"rows": 4, "cols": 8, "contact_pitch_mm": 2.5,
                     "target_level": "c7"}


def test_main_full_spine_forces_slab(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "generate_electrode_array",
        lambda cfg, **kw: calls.update(kw),
    )
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path), "--full-spine",
    ])
    assert rc == 0
    # "" is the explicit slab opt-out (overrides a source-target level default).
    assert calls["target_level"] == ""


def test_main_full_spine_and_level_conflict(tmp_path) -> None:
    with pytest.raises(SystemExit):
        cli_mod.main([
            "--config", str(TINY_CFG), "--project-root", str(tmp_path),
            "--full-spine", "--level", "c7",
        ])
