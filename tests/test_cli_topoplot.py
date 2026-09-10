"""CLI: topoplot argparse targets + wiring."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import inob.cli.topoplot as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def _fig():
    return plt.figure()


def test_main_target_dual(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod,
        "render_dual_topoplot",
        lambda cfg, **kw: calls.update(kw),
    )
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls["source_idx"] == -1
    assert calls["dpi"] == 180


def test_main_target_meg_saves_png(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "render_meg_topoplot",
        lambda cfg, **kw: (_fig(), None),
    )
    out = tmp_path / "meg.png"
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--target",
            "meg",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()


def test_main_target_eeg_saves_png(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli_mod, "render_eeg_topoplot", lambda cfg, **kw: _fig())
    out = tmp_path / "eeg.png"
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--target",
            "eeg",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()


def test_main_target_montage(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod,
        "render_meg_montage",
        lambda cfg, **kw: calls.update(kw),
    )
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--target",
            "montage",
            "--n-sources",
            "3",
        ]
    )
    assert rc == 0
    assert calls["n_sources"] == 3
