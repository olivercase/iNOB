"""CLI: cross_modality argparse defaults + wiring."""
from __future__ import annotations

from pathlib import Path

import inob.cli.cross_modality as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def test_main_defaults(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "render_cross_modality",
        lambda cfg, **kwargs: calls.update(kwargs),
    )
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls["source_idx"] == -1
    assert calls["out_path"] is None
    assert calls["dpi"] == 300
    assert calls["noise_uV"] is None
    assert calls["noise_seed"] == 0


def test_main_custom_args(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "render_cross_modality",
        lambda cfg, **kwargs: calls.update(kwargs),
    )
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--source-idx", "3", "--noise-uV", "1.5", "--noise-seed", "7",
    ])
    assert rc == 0
    assert calls["source_idx"] == 3
    assert calls["noise_uV"] == 1.5
    assert calls["noise_seed"] == 7
