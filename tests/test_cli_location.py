"""CLI: location argparse defaults + wiring."""
from __future__ import annotations

from pathlib import Path

import inob.cli.location as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def test_main_defaults(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "render_location_optimisation",
        lambda cfg, **kw: calls.update(kw),
    )
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls["paddle_mat"] == Path("outputs/sensors/electrode_array.mat")
    assert calls["paddle_npz"] == Path("outputs/forward/duneuro_eeg_leadfield_vagus.npz")
    assert calls["wholebody_mat"] == Path("outputs/sensors/electrode_array_wholebody.mat")
    assert calls["wholebody_npz"] == Path("outputs/forward/duneuro_eeg_leadfield_wholebody.npz")
    assert calls["source_idx"] == -1
    assert calls["out_path"] is None
    assert calls["dpi"] == 300


def test_main_custom_paths(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "render_location_optimisation",
        lambda cfg, **kw: calls.update(kw),
    )
    paddle_mat = tmp_path / "p.mat"
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--paddle-mat", str(paddle_mat), "--source-idx", "1",
    ])
    assert rc == 0
    assert calls["paddle_mat"] == paddle_mat
    assert calls["source_idx"] == 1
