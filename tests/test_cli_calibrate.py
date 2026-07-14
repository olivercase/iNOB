"""CLI: calibrate argparse defaults + wiring."""
from __future__ import annotations

from pathlib import Path

import inob.cli.calibrate as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


class _FakeSummary:
    factor_peak = 1.0
    factor_median = 2.0
    factor_geomean = 3.0
    n_electrodes = 200
    radius_mm = 100.0
    source_radius_mm = 50.0
    sigma_S_per_m = 0.43
    n_tets = 42


def test_main_defaults_and_wiring(tmp_path, monkeypatch, capsys) -> None:
    calls = {}

    def fake_calibrate(**kwargs):
        calls.update(kwargs)
        return _FakeSummary()

    monkeypatch.setattr(cli_mod, "calibrate_eeg_factor", fake_calibrate)
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls["radius_mm"] == 100.0
    assert calls["pitch_mm"] == 3.0
    assert calls["sigma_S_per_m"] == 0.43
    assert calls["source_radius_mm"] == 50.0
    assert calls["n_electrodes"] == 200
    assert calls["out_dir"] == Path("outputs/calibration")
    out = capsys.readouterr().out
    assert "empirical EEG factor" in out


def test_main_overrides_are_forwarded(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        cli_mod, "calibrate_eeg_factor",
        lambda **kwargs: (calls.update(kwargs), _FakeSummary())[1],
    )
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--radius-mm", "80", "--pitch-mm", "1.5", "--sigma", "0.3",
        "--source-radius-mm", "40", "--n-electrodes", "50",
        "--out-dir", str(tmp_path / "calib"),
    ])
    assert rc == 0
    assert calls["radius_mm"] == 80.0
    assert calls["pitch_mm"] == 1.5
    assert calls["sigma_S_per_m"] == 0.3
    assert calls["source_radius_mm"] == 40.0
    assert calls["n_electrodes"] == 50
    assert calls["out_dir"] == tmp_path / "calib"
