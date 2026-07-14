"""CLI: snr argparse defaults + wiring."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import inob.cli.snr as cli_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


class _FakeLF:
    # shape (channels=2, 3*sources=3) so per_source_amplitude's reshape works.
    L_fT_per_nAm = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


class _FakeFloors:
    meg_per_channel_fT = 15.0
    eeg_per_channel_uV = 0.5


def _patch(monkeypatch, calls):
    monkeypatch.setattr(cli_mod, "load_leadfield", lambda path: (calls.setdefault("lf_path", path), _FakeLF())[1])
    monkeypatch.setattr(cli_mod, "compute_noise_floors", lambda cfg: _FakeFloors())


def test_main_defaults_uses_meg_forward_npz(tmp_path, monkeypatch, capsys) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    cfg_forward_npz = tmp_path / "outputs" / "forward" / "duneuro_leadfield_vagus.npz"
    rc = cli_mod.main(["--config", str(TINY_CFG), "--project-root", str(tmp_path)])
    assert rc == 0
    assert calls["lf_path"] == cfg_forward_npz
    out = json.loads(capsys.readouterr().out)
    assert out["modality"] == "meg"
    assert out["moment"] == "rms"
    assert out["n_averages"] == 1
    assert out["sigma_per_channel"] == 15.0


def test_main_eeg_modality_uses_eeg_npz(tmp_path, monkeypatch, capsys) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    cfg_eeg_npz = tmp_path / "outputs" / "forward" / "duneuro_eeg_leadfield_vagus.npz"
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path), "--modality", "eeg",
    ])
    assert rc == 0
    assert calls["lf_path"] == cfg_eeg_npz
    out = json.loads(capsys.readouterr().out)
    assert out["sigma_per_channel"] == 0.5


def test_main_explicit_leadfield_overrides(tmp_path, monkeypatch, capsys) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    custom = tmp_path / "custom.npz"
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--leadfield", str(custom),
    ])
    assert rc == 0
    assert calls["lf_path"] == custom


def test_main_writes_json_when_out_given(tmp_path, monkeypatch) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    out_file = tmp_path / "snr_out" / "result.json"
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--out", str(out_file),
    ])
    assert rc == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["modality"] == "meg"


def test_main_n_averages_forwarded(tmp_path, monkeypatch, capsys) -> None:
    calls = {}
    _patch(monkeypatch, calls)
    rc = cli_mod.main([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--n-averages", "4",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_averages"] == 4
