"""CLI: build_fem argparse + wiring."""
from __future__ import annotations

from pathlib import Path

import inob.cli.build_fem as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def test_main_calls_build_fem(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(cli_mod, "build_fem", lambda cfg: calls.setdefault("cfg", cfg))
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert "cfg" in calls
    assert calls["cfg"].project_root == tmp_path
