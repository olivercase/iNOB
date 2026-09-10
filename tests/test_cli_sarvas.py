"""CLI: sarvas argparse defaults + wiring."""

from __future__ import annotations

from pathlib import Path

import inob.cli.sarvas as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


class _FakeResult:
    pass


def _patch(monkeypatch, calls):
    def fake_compare(cfg, **kw):
        calls["compare"] = kw
        return _FakeResult()

    def fake_render(result, **kw):
        calls["render"] = kw

    def fake_save(result, path):
        calls["save_path"] = path

    monkeypatch.setattr(cli_mod, "compare_sarvas_vs_fem", fake_compare)
    monkeypatch.setattr(cli_mod, "render_sarvas_vs_fem", fake_render)
    monkeypatch.setattr(cli_mod, "save_comparison_summary", fake_save)


def test_main_defaults(tmp_path, monkeypatch, capsys) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls["compare"]["Q_nAm"] == 1.0
    assert calls["render"]["source_idx"] == -1
    assert calls["render"]["dpi"] == 300
    out = capsys.readouterr().out
    assert "Hämäläinen" not in out
    assert "figure:" in out and "summary:" in out


def test_main_show_hamalainen_prints(tmp_path, monkeypatch, capsys) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--show-hamalainen",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "A-fibre" in out and "C-fibre" in out


def test_main_custom_out_path(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    out = tmp_path / "custom.png"
    rc = cli_mod.main(
        [
            "--config",
            str(TINY_CFG),
            "--project-root",
            str(tmp_path),
            "--Q-nAm",
            "70",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert calls["compare"]["Q_nAm"] == 70.0
    assert calls["save_path"] == out.with_suffix(".json")
