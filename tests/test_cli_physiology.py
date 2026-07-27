"""CLI: physiology argparse defaults + wiring.

The CLI is profile-driven: it selects the target's physiology profile and
forwards only the per-scenario knobs the user actually set into
``profile.scenarios(**kw)``, so each scenario keeps its own documented defaults.
"""
from __future__ import annotations

from pathlib import Path

import inob.cli.physiology as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


class _FakeProfile:
    """Stand-in profile that records the kwargs the CLI forwards."""

    label = "vagus"
    validated = True

    def __init__(self, calls: dict) -> None:
        self._calls = calls

    def describe(self) -> str:
        return "fake profile"

    def scenarios(self, **kw):
        self._calls["scenario_kw"] = kw
        return ("baro-scenario", "resp-scenario")


def _patch(monkeypatch, calls):
    monkeypatch.setattr(cli_mod, "profile_for_tag", lambda tag: _FakeProfile(calls))
    monkeypatch.setattr(
        cli_mod, "render_physiology",
        lambda cfg, **kw: calls.update({"render": kw}),
    )


def test_main_defaults(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    # No per-scenario knobs set → nothing forwarded; scenarios use their defaults.
    assert calls["scenario_kw"] == {}
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
    # Only the knobs the user set are forwarded, each prefixed by its scenario.
    assert calls["scenario_kw"] == {
        "baro_hr_bpm": 80.0,
        "baro_n_fibres_per_burst": 50,
        "resp_breath_bpm": 10.0,
        "resp_n_phasic_fibres_RAR": 20,
        "resp_n_tonic_fibres_SAR": 5,
    }
    assert calls["render"]["fs_hz"] == 1000.0
