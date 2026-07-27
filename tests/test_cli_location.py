"""CLI: location argparse defaults + wiring."""
from __future__ import annotations

from pathlib import Path

import inob.cli.location as cli_mod
from inob.config import load_config, source_target_tag, tag_path

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
    # Paddle inputs now follow the source-target-aware config (no --source-target
    # here → the untagged defaults), not hardcoded vagus paths.
    cfg = load_config(TINY_CFG, overrides=[], project_root=tmp_path)
    tag = source_target_tag(cfg)
    assert calls["paddle_mat"] == cfg.outputs.electrodes_mat
    assert calls["paddle_npz"] == cfg.outputs.forward_eeg_npz
    assert calls["wholebody_mat"] == Path("outputs/sensors/electrode_array_wholebody.mat")
    assert calls["wholebody_npz"] == tag_path(
        Path("outputs/forward/duneuro_eeg_leadfield_wholebody.npz"), tag
    )
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
