"""Tests for inob.viz.physiology_plot: unit-conversion helper + render smoke."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from inob.physiology.scenarios import CapEvent, Scenario, a_fibre_population
from inob.viz.physiology_plot import _to_human_units, render_physiology
from tests.viz_pipeline_helpers import build_pipeline_cfg


def test_to_human_units_meg() -> None:
    vals, unit = _to_human_units(np.array([1e-15, 2e-15]), modality="meg")
    np.testing.assert_allclose(vals, [1.0, 2.0])
    assert unit == "fT"


def test_to_human_units_eeg() -> None:
    vals, unit = _to_human_units(np.array([1e-6]), modality="eeg")
    np.testing.assert_allclose(vals, [1.0])
    assert unit == "µV"


def test_to_human_units_unknown_modality_raises() -> None:
    with pytest.raises(ValueError, match="unknown modality"):
        _to_human_units(np.array([1.0]), modality="bogus")


def _tiny_scenario(duration_s: float = 0.2, n_events: int = 2) -> Scenario:
    fibres = a_fibre_population(n_bins=4)
    events = [
        CapEvent(t_start_s=0.02 + 0.05 * k, n_fibres=10, fibres=fibres)
        for k in range(n_events)
    ]
    return Scenario(
        name="tiny", description="unit-test scenario", duration_s=duration_s,
        events=events, rate_hz=5.0,
        physiology_trace_label="trace (a.u.)",
        physiology_trace=np.zeros(50),
    )


def test_render_physiology_smoke(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    out = render_physiology(
        cfg, scenarios=(_tiny_scenario(),), fs_hz=2000.0, out_path=tmp_path / "phys.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_physiology_multiple_scenarios(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    out = render_physiology(
        cfg,
        scenarios=(_tiny_scenario(), _tiny_scenario(duration_s=0.15, n_events=1)),
        fs_hz=1500.0,
        out_path=tmp_path / "phys2.png",
    )
    assert out.exists()


def test_render_physiology_default_out_path(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    out = render_physiology(cfg, scenarios=(_tiny_scenario(),), fs_hz=2000.0)
    assert out == cfg.outputs.base / "physiology_vagus.png"
    assert out.exists()
