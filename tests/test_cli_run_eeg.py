"""CLI: run_eeg argparse + wiring."""
from __future__ import annotations

from pathlib import Path

import inob.cli.run_eeg as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def test_main_calls_run_eeg_forward(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "run_eeg_forward", lambda cfg: calls.setdefault("cfg", cfg),
    )
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls["cfg"].project_root == tmp_path
