"""CLI: source-models argparse targets + wiring."""
from __future__ import annotations

import json
from pathlib import Path

import inob.cli.source_models as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def _patch(monkeypatch, calls):
    def fake_summary(cfg, **kw):
        calls["summary"] = kw
        return {"models": {}, "args": {k: str(v) for k, v in kw.items()}}

    def fake_render(cfg, **kw):
        calls["render"] = kw
        return Path("fig.png")

    monkeypatch.setattr(cli_mod, "source_model_summary", fake_summary)
    monkeypatch.setattr(cli_mod, "render_source_models", fake_render)


def test_main_renders_and_prints(tmp_path, monkeypatch, capsys) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert "models" in json.loads(capsys.readouterr().out)
    assert "render" in calls
    # Both paths default to "unspecified" so the analysis module stays the one
    # place that decides the source index and Q.
    assert calls["summary"] == {"Q_nAm": None, "source_idx": None}


def test_no_figure_skips_the_render(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path), "--no-figure",
    ])
    assert rc == 0
    assert "render" not in calls
    assert "summary" in calls


def test_q_and_source_idx_reach_both_the_figure_and_the_numbers(
    tmp_path, monkeypatch,
) -> None:
    """They must agree, or the figure and the printed table describe different runs."""
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--Q-nAm", "12.5", "--source-idx", "70",
    ])
    assert rc == 0
    assert calls["summary"]["Q_nAm"] == 12.5
    assert calls["summary"]["source_idx"] == 70
    assert calls["render"]["Q_nAm"] == 12.5
    assert calls["render"]["source_idx"] == 70


def test_json_out_is_written(tmp_path, monkeypatch) -> None:
    _patch(monkeypatch, {})
    out = tmp_path / "nested" / "cmp.json"
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--json-out", str(out), "--no-figure",
    ])
    assert rc == 0
    assert "models" in json.loads(out.read_text())
