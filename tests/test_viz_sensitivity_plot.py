"""Tests for inob.viz.sensitivity_plot (MEG-vs-EEG sensitivity bar chart)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pytest

from inob.config import load_config
from inob.viz.sensitivity_plot import _key, _load_results, render_sensitivity

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


@pytest.fixture
def cfg(tmp_path: Path):
    return load_config(TINY_CFG, project_root=tmp_path)


def _write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": rows}))


def test_key_format() -> None:
    assert _key({"tissue": "bone", "factor": 1.25}) == "bone ×1.25"


def test_load_results_missing_file(tmp_path: Path) -> None:
    assert _load_results(tmp_path / "nope.json") == []


def test_load_results_reads_json(tmp_path: Path) -> None:
    p = tmp_path / "sensitivity_meg.json"
    rows = [{"tissue": "bone", "factor": 0.8, "p95_rel_change": 0.01}]
    _write_results(p, rows)
    assert _load_results(p) == rows


def test_render_sensitivity_raises_if_no_data(cfg) -> None:
    with pytest.raises(FileNotFoundError):
        render_sensitivity(cfg)


def test_render_sensitivity_meg_only(cfg) -> None:
    rows = [
        {"tissue": "bone", "factor": 0.8, "p95_rel_change": 0.02, "rms": 0.01, "p50": 0.015},
        {"tissue": "skin", "factor": 1.25, "p95_rel_change": 0.05, "rms": 0.03, "p50": 0.04},
    ]
    _write_results(cfg.outputs.sensitivity_dir / "sensitivity_meg.json", rows)
    out = render_sensitivity(cfg)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_sensitivity_meg_and_eeg(cfg) -> None:
    meg_rows = [{"tissue": "bone", "factor": 0.8, "p95_rel_change": 0.02}]
    eeg_rows = [{"tissue": "bone", "factor": 0.8, "p95_rel_change": 0.4}]
    _write_results(cfg.outputs.sensitivity_dir / "sensitivity_meg.json", meg_rows)
    _write_results(cfg.outputs.sensitivity_dir / "sensitivity_eeg.json", eeg_rows)
    out = render_sensitivity(cfg)
    assert out.exists()


def test_render_sensitivity_custom_out_path(cfg, tmp_path: Path) -> None:
    rows = [{"tissue": "bone", "factor": 0.8, "p95_rel_change": 0.02}]
    _write_results(cfg.outputs.sensitivity_dir / "sensitivity_meg.json", rows)
    out_path = tmp_path / "custom.png"
    out = render_sensitivity(cfg, out_path=out_path)
    assert out == out_path
    assert out.exists()
