"""CLI: run_forward argparse + wiring."""

from __future__ import annotations

from pathlib import Path

import inob.cli.run_forward as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def test_main_uses_the_multicore_path(tmp_path, monkeypatch) -> None:
    # Not the serial solve: the transfer matrix costs one solve per coil, so
    # running `inob forward` on one core wastes every other core on the machine.
    calls = {}
    monkeypatch.setattr(
        cli_mod,
        "run_forward_local",
        lambda cfg: calls.setdefault("cfg", cfg),
    )
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls["cfg"].project_root == tmp_path


def test_workers_flag_reaches_the_config(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod,
        "run_forward_local",
        lambda cfg: calls.setdefault("cfg", cfg),
    )
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--workers",
            "4",
        ]
    )
    assert rc == 0
    assert calls["cfg"].forward.local_workers == 4
