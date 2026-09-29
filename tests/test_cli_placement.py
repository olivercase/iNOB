"""CLI: placement argparse defaults + wiring."""

from __future__ import annotations

from pathlib import Path

import inob.cli.placement as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def test_main_defaults(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod,
        "render_placement_comparison",
        lambda cfg, **kw: (calls.update(kw), (Path("fig.png"), []))[1],
    )
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls["forward_npz"] is None  # resolved from the config downstream
    assert calls["source_idx"] == -1  # the best-coupled source
    assert calls["n_sensors"] == 20
    assert calls["min_spacing_mm"] == 30.0
    assert calls["out_path"] is None
    assert calls["dpi"] == 300


def test_main_custom_arguments(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod,
        "render_placement_comparison",
        lambda cfg, **kw: (calls.update(kw), (Path("fig.png"), []))[1],
    )
    npz = tmp_path / "lf.npz"
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--forward-npz",
            str(npz),
            "--source-idx",
            "7",
            "--n-sensors",
            "64",
            "--min-spacing-mm",
            "20",
        ]
    )
    assert rc == 0
    assert calls["forward_npz"] == npz
    assert calls["source_idx"] == 7
    assert calls["n_sensors"] == 64
    assert calls["min_spacing_mm"] == 20.0
