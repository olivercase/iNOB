"""CLI: cap_compare argparse defaults + wiring."""
from __future__ import annotations

from pathlib import Path

import inob.cli.cap_compare as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def test_main_defaults(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "render_cap_compare",
        lambda cfg, **kwargs: calls.update(kwargs),
    )
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    # ap_width_ms / duration_ms default to None so the physiology profile
    # supplies them (profile-driven cap_compare); only fs_hz/dpi are fixed here.
    assert calls["ap_width_ms"] is None
    assert calls["fs_hz"] == 30_000.0
    assert calls["duration_ms"] is None
    assert calls["out_path"] is None
    assert calls["dpi"] == 300


def test_main_custom_args(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "render_cap_compare",
        lambda cfg, **kwargs: calls.update(kwargs),
    )
    out = tmp_path / "fig.png"
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--ap-width-ms", "1.0", "--fs-hz", "1000", "--duration-ms", "10",
        "--out", str(out), "--dpi", "72",
    ])
    assert rc == 0
    assert calls["ap_width_ms"] == 1.0
    assert calls["fs_hz"] == 1000.0
    assert calls["duration_ms"] == 10.0
    assert calls["out_path"] == out
    assert calls["dpi"] == 72
