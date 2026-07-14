"""Tests for the conductivity sensitivity sweep helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inob.analysis.sensitivity import (
    PerturbationResult,
    _channel_amplitude,
    relative_change,
    summarise_perturbation,
    sweep,
)
from inob.config import load_config
from inob.io.npz import Leadfield, save_leadfield

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"


def test_channel_amplitude_is_rms_across_moments() -> None:
    L = np.array([[3.0, 4.0], [0.0, 0.0]])
    amp = _channel_amplitude(L)
    np.testing.assert_allclose(amp, [np.sqrt(12.5), 0.0])


def test_relative_change_identical_is_zero() -> None:
    L = np.ones((4, 3))
    rel = relative_change(L, L)
    np.testing.assert_allclose(rel, 0.0)


def test_relative_change_doubled_amplitude() -> None:
    L0 = np.ones((2, 3))
    L1 = np.full((2, 3), 2.0)
    rel = relative_change(L1, L0)
    np.testing.assert_allclose(rel, 1.0)


def test_relative_change_handles_zero_baseline_without_div_by_zero() -> None:
    L0 = np.zeros((2, 3))
    L1 = np.ones((2, 3))
    rel = relative_change(L1, L0)
    assert np.isfinite(rel).all()


def test_summarise_perturbation_statistics() -> None:
    rel = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    res = summarise_perturbation(rel, tissue="bone", factor=1.25)
    assert isinstance(res, PerturbationResult)
    assert res.tissue == "bone"
    assert res.factor == 1.25
    assert res.rms_rel_change == pytest.approx(float(np.sqrt(np.mean(rel ** 2))))
    assert res.p50_rel_change == pytest.approx(float(np.percentile(rel, 50)))
    assert res.p95_rel_change == pytest.approx(float(np.percentile(rel, 95)))


def _leadfield(L: np.ndarray, n_src: int) -> Leadfield:
    C = L.shape[0]
    return Leadfield(
        L=L, L_fT_per_nAm=L * 1e6,
        source_pos=np.zeros((n_src, 3)),
        coil_pos=np.zeros((C, 3)),
        coil_orient=np.tile([1.0, 0.0, 0.0], (C, 1)),
        channel_names=tuple(f"mag-{i:04d}-R" for i in range(C)),
        conductivities=np.array([1e-4]),
        tissue_labels=("vagus_left",),
        seed=0,
    )


def test_sweep_writes_summary_and_calls_forward_fn(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CFG, project_root=tmp_path)
    n_src = 2
    baseline_L = np.ones((5, 3 * n_src))
    baseline_path = tmp_path / "baseline.npz"
    save_leadfield(baseline_path, _leadfield(baseline_L, n_src))

    calls: list[tuple[float, Path]] = []

    def fake_forward_fn(cfg_p, out_npz: Path) -> np.ndarray:
        sigma = cfg_p.forward.conductivities_sm["bone"]
        calls.append((sigma, out_npz))
        # perturbed leadfield scaled relative to a nominal sigma of 0.0042
        scale = sigma / 0.0042
        return baseline_L * scale

    out_path = tmp_path / "sensitivity_meg.json"
    summary = sweep(
        cfg, baseline_path=baseline_path, forward_fn=fake_forward_fn,
        out_path=out_path, modality="meg",
    )

    n_tissues = len(cfg.sensitivity.tissues)
    n_pert = len(cfg.sensitivity.perturbations)
    assert len(calls) == n_tissues * n_pert
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk == summary
    assert summary["modality"] == "meg"
    assert len(summary["results"]) == n_tissues * n_pert
    # the factor=1.0 perturbation should show ~zero relative change
    unity = [r for r in summary["results"] if r["factor"] == 1.0]
    assert unity
    for r in unity:
        assert r["rms_rel_change"] == pytest.approx(0.0, abs=1e-9)


def test_sweep_skips_tissue_missing_from_conductivities(tmp_path: Path) -> None:
    from dataclasses import replace

    cfg = load_config(DEFAULT_CFG, project_root=tmp_path)
    cfg = replace(cfg, sensitivity=replace(cfg.sensitivity, tissues=("not_a_tissue",)))

    n_src = 1
    baseline_L = np.ones((2, 3 * n_src))
    baseline_path = tmp_path / "baseline.npz"
    save_leadfield(baseline_path, _leadfield(baseline_L, n_src))

    calls: list[str] = []

    def fake_forward_fn(cfg_p, out_npz: Path) -> np.ndarray:
        calls.append(out_npz.name)
        return baseline_L

    out_path = tmp_path / "sensitivity_meg.json"
    summary = sweep(
        cfg, baseline_path=baseline_path, forward_fn=fake_forward_fn,
        out_path=out_path,
    )
    assert calls == []
    assert summary["results"] == []
