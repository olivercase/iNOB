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


class _FakeMeg:
    """Stand-in for MegSphereValidation — only the reported fields matter."""

    def __init__(self, rdm: float = 0.01, mag: float = 1.0) -> None:
        self.rdm = rdm
        self.mag = mag
        self.fem_peak_fT_per_nAm = 1.23
        self.sarvas_peak_fT_per_nAm = 1.24
        self.n_coils = 60
        self.n_tets = 42
        self.radius_mm = 100.0


def test_meg_flag_validates_against_sarvas(tmp_path, monkeypatch, capsys) -> None:
    calls = {}

    def fake_validate(**kwargs):
        calls.update(kwargs)
        return _FakeMeg()

    monkeypatch.setattr(cli_mod, "validate_meg_sphere", fake_validate)
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path),
                       "--meg"])
    assert rc == 0
    # The source model must reach DUNEuro, not be silently dropped.
    assert calls["source_model"] == {"type": "partial_integration"}
    out = capsys.readouterr().out
    assert "MEG sphere" in out and "RDM" in out


def test_compare_source_models_runs_one_per_model(tmp_path, monkeypatch, capsys) -> None:
    seen: list[str] = []

    def fake_validate(**kwargs):
        model = kwargs["source_model"]["type"]
        seen.append(model)
        # Make Venant the winner so the "lowest RDM" line is checkable.
        return _FakeMeg(rdm=0.005 if model == "venant" else 0.02)

    monkeypatch.setattr(cli_mod, "validate_meg_sphere", fake_validate)
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path),
                       "--compare-source-models"])
    assert rc == 0
    assert seen == ["partial_integration", "venant", "multipolar_venant"]
    out = capsys.readouterr().out
    assert "Lowest RDM: venant" in out


def test_compare_source_models_accepts_an_explicit_subset(
    tmp_path, monkeypatch,
) -> None:
    seen: list[dict] = []

    def fake_validate(**kwargs):
        seen.append(kwargs["source_model"])
        return _FakeMeg()

    monkeypatch.setattr(cli_mod, "validate_meg_sphere", fake_validate)
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--compare-source-models", "partial_integration", "venant",
    ])
    assert rc == 0
    assert [s["type"] for s in seen] == ["partial_integration", "venant"]
    # The Venant row must carry the parameters DUNEuro has no defaults for —
    # sweeping the type alone would throw inside C++.
    assert "numberOfMoments" in seen[1] and "restrict" in seen[1]
    assert "numberOfMoments" not in seen[0]
