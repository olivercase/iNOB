"""CLI: generate_electrodes argparse + wiring."""
from __future__ import annotations

from pathlib import Path

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
    assert calls == {"rows": None, "cols": None, "contact_pitch_mm": None}


def test_main_overrides_forwarded(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "generate_electrode_array",
        lambda cfg, **kw: calls.update(kw),
    )
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--rows", "4", "--cols", "8", "--pitch", "2.5",
    ])
    assert rc == 0
    assert calls == {"rows": 4, "cols": 8, "contact_pitch_mm": 2.5}
