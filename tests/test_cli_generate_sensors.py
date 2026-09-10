"""CLI: generate_sensors argparse + wiring."""

from __future__ import annotations

from pathlib import Path

import inob.cli.generate_sensors as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def test_main_defaults(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod,
        "generate_sensor_array",
        lambda cfg, **kw: calls.update(kw),
    )
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls == {"z_min": None, "z_max": None}


def test_main_zmin_zmax_forwarded(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod,
        "generate_sensor_array",
        lambda cfg, **kw: calls.update(kw),
    )
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--zmin",
            "-10",
            "--zmax",
            "50",
        ]
    )
    assert rc == 0
    assert calls == {"z_min": -10.0, "z_max": 50.0}
