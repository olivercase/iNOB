"""Tests for the GUI detectability path: explicit point sources + trials math."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from inob.analysis.detect import compute_detectability
from inob.analysis.snr import compute_noise_floors, per_source_peak
from inob.config import load_config
from inob.io.npz import Leadfield, save_leadfield

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"


def _synthetic_leadfield(path: Path, peaks: list[float]) -> None:
    """Write a minimal valid leadfield NPZ whose per-source best-channel peak
    |L| equals ``peaks`` (one value per source)."""
    S = len(peaks)
    C = 3
    L3 = np.zeros((C, S, 3), dtype=np.float64)
    for s, p in enumerate(peaks):
        L3[0, s, 0] = p  # one channel/orientation carries the peak
    L = L3.reshape(C, 3 * S)
    lf = Leadfield(
        L=L,
        L_fT_per_nAm=L,
        source_pos=np.column_stack([np.zeros(S), np.zeros(S), np.arange(S) * 10.0]),
        coil_pos=np.zeros((C, 3)),
        coil_orient=np.tile([0.0, 0.0, 1.0], (C, 1)),
        channel_names=tuple(f"mag-{i}" for i in range(C)),
        conductivities=np.array([0.3, 0.43]),
        tissue_labels=("vagus_left", "skin"),
    )
    save_leadfield(path, lf)


def test_per_source_peak_definition() -> None:
    L = np.zeros((3, 6))  # C=3, S=2
    L[0, 0] = 10.0  # source 0, x-moment
    L[1, 4] = -25.0  # source 1, y-moment (abs peak)
    peak = per_source_peak(L)
    assert peak.tolist() == [10.0, 25.0]


def test_compute_detectability_meg(tmp_path: Path) -> None:
    npz = tmp_path / "meg.npz"
    _synthetic_leadfield(npz, peaks=[10.0, 20.0])
    cfg = load_config(DEFAULT_CFG, overrides=[f"outputs.forward_npz={npz}"])
    sigma = compute_noise_floors(cfg).meg_per_channel_fT

    res = compute_detectability(
        cfg, strengths_nAm=[1.0, 1.0], threshold_snr=3.0, modality="meg",
    )
    assert res["modality"] == "meg"
    assert len(res["per_source"]) == 2

    s0, s1 = res["per_source"]
    # SNR = peak * Q / sigma; trials = ceil((threshold / SNR)^2)
    assert s0["snr"] == pytest.approx(10.0 / sigma, rel=1e-3)
    assert s1["snr"] == pytest.approx(20.0 / sigma, rel=1e-3)
    assert s0["trials_needed"] == math.ceil((3.0 / (10.0 / sigma)) ** 2)
    assert s1["trials_needed"] == math.ceil((3.0 / (20.0 / sigma)) ** 2)
    # twice the signal → a quarter of the trials
    assert s0["trials_needed"] == pytest.approx(s1["trials_needed"] * 4, rel=0.02)


def test_detect_strength_scales_snr_linearly(tmp_path: Path) -> None:
    npz = tmp_path / "meg.npz"
    _synthetic_leadfield(npz, peaks=[10.0])
    cfg = load_config(DEFAULT_CFG, overrides=[f"outputs.forward_npz={npz}"])
    weak = compute_detectability(cfg, strengths_nAm=[1.0], modality="meg")
    strong = compute_detectability(cfg, strengths_nAm=[2.0], modality="meg")
    assert strong["per_source"][0]["snr"] == pytest.approx(
        2.0 * weak["per_source"][0]["snr"], rel=1e-3,
    )


def test_detect_zero_signal_never_detectable(tmp_path: Path) -> None:
    npz = tmp_path / "meg.npz"
    _synthetic_leadfield(npz, peaks=[0.0])
    cfg = load_config(DEFAULT_CFG, overrides=[f"outputs.forward_npz={npz}"])
    res = compute_detectability(cfg, strengths_nAm=[70.0], modality="meg")
    assert res["per_source"][0]["snr"] == 0.0
    assert res["per_source"][0]["trials_needed"] == -1  # sentinel for "never"


def test_detect_bad_modality(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CFG)
    with pytest.raises(ValueError, match="modality"):
        compute_detectability(cfg, modality="xray")


def test_point_sources_override_config() -> None:
    cfg = load_config(
        DEFAULT_CFG,
        overrides=["forward.point_sources=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"],
    )
    assert cfg.forward.point_sources == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))


def test_point_sources_default_empty() -> None:
    cfg = load_config(DEFAULT_CFG)
    assert cfg.forward.point_sources == ()


def test_detectability_scenarios_follow_source_target() -> None:
    """Each target plans against the Q range its own literature supports."""
    from inob.physiology.profiles import SPINE_PROFILE
    from inob.viz.detectability import (
        DEFAULT_SCENARIOS,
        MUSCLE_SCENARIOS,
        scenarios_for_target,
    )
    base = "outputs/forward/duneuro_leadfield_{}.npz"

    def cfg_for(target: str):
        return load_config(DEFAULT_CFG,
                           overrides=[f"outputs.forward_npz={base.format(target)}"])

    assert scenarios_for_target(cfg_for("vagus")) is DEFAULT_SCENARIOS
    assert scenarios_for_target(cfg_for("muscle")) is MUSCLE_SCENARIOS

    # The spine plans against magnetospinography, not the vagus's 70 nA·m
    # full-summation figure — which is an order of magnitude above anything
    # reported for the cord. Its ladder must contain the profile's anchor
    # exactly, so the detectability and time-domain figures cannot disagree.
    for target in ("spine", "spine_vagus", "spine_muscle"):
        spine = scenarios_for_target(cfg_for(target))
        assert spine is not DEFAULT_SCENARIOS
        assert any(s.Q_nAm == SPINE_PROFILE.default_strength_nAm for s in spine)
        assert max(s.Q_nAm for s in spine) < max(s.Q_nAm for s in DEFAULT_SCENARIOS)
    # Muscle sources are far stronger than vagal CAPs. The ranges overlap at the
    # bottom (a single MUAP is comparable to a modest CAP), but muscle is shifted
    # up throughout and tops out an order of magnitude higher.
    assert min(s.Q_nAm for s in MUSCLE_SCENARIOS) > min(s.Q_nAm for s in DEFAULT_SCENARIOS)
    assert max(s.Q_nAm for s in MUSCLE_SCENARIOS) > 10 * max(s.Q_nAm for s in DEFAULT_SCENARIOS)
