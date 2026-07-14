"""CLI: physiology argparse defaults + wiring."""
from __future__ import annotations

from pathlib import Path

import inob.cli.physiology as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def _patch(monkeypatch, calls):
    def fake_baro(**kw):
        calls["baro"] = kw
        return "baro-scenario"

    def fake_resp(**kw):
        calls["resp"] = kw
        return "resp-scenario"

    def fake_render(cfg, **kw):
        calls["render"] = kw

    monkeypatch.setattr(cli_mod, "baroreceptor_scenario", fake_baro)
    monkeypatch.setattr(cli_mod, "respiratory_scenario", fake_resp)
    monkeypatch.setattr(cli_mod, "render_physiology", fake_render)


def test_main_defaults(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls["baro"] == {
        "duration_s": 6.0, "hr_bpm": 70.0, "n_fibres_per_burst": 200,
    }
    assert calls["resp"] == {
        "duration_s": 12.0, "breath_bpm": 6.0,
        "n_phasic_fibres_RAR": 600, "n_tonic_fibres_SAR": 80,
    }
    assert calls["render"]["scenarios"] == ("baro-scenario", "resp-scenario")
    assert calls["render"]["fs_hz"] == 30_000.0
    assert calls["render"]["out_path"] is None
    assert calls["render"]["dpi"] == 300


def test_main_custom_args(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--hr-bpm", "80", "--breath-bpm", "10", "--n-fibres-baro", "50",
        "--n-fibres-rar", "20", "--n-fibres-sar", "5", "--fs-hz", "1000",
    ])
    assert rc == 0
    assert calls["baro"]["hr_bpm"] == 80.0
    assert calls["baro"]["n_fibres_per_burst"] == 50
    assert calls["resp"]["breath_bpm"] == 10.0
    assert calls["resp"]["n_phasic_fibres_RAR"] == 20
    assert calls["resp"]["n_tonic_fibres_SAR"] == 5
    assert calls["render"]["fs_hz"] == 1000.0
