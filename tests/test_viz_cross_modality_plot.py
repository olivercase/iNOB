"""Tests for inob.viz.cross_modality_plot: render smoke tests."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from inob.viz.cross_modality_plot import render_cross_modality
from tests.viz_pipeline_helpers import build_pipeline_cfg


def test_render_cross_modality_smoke(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    out = render_cross_modality(cfg, out_path=tmp_path / "cm.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_cross_modality_default_out_path(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    out = render_cross_modality(cfg)
    assert out == cfg.outputs.base / "cross_modality_vagus.png"
    assert out.exists()


def test_render_cross_modality_explicit_noise(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    out = render_cross_modality(
        cfg, out_path=tmp_path / "cm_noisy.png", noise_uV=2.0, noise_seed=7, source_idx=2,
    )
    assert out.exists()


def test_render_cross_modality_zero_noise_is_near_exact_recovery(tmp_path: Path) -> None:
    """With noise_uV pinned to a tiny value the lstsq round trip should be
    almost exact, per the module's own docstring."""
    import numpy as np

    from inob.analysis.cross_modality import predict_meg_from_eeg
    from inob.io.npz import load_leadfield

    cfg = build_pipeline_cfg(tmp_path)
    meg_lf = load_leadfield(cfg.outputs.forward_npz)
    eeg_lf = load_leadfield(cfg.outputs.forward_eeg_npz)
    source_idx = meg_lf.source_pos.shape[0] // 2
    L_eeg_src = eeg_lf.L_fT_per_nAm[:, 3 * source_idx: 3 * source_idx + 3]
    L_meg_src = meg_lf.L_fT_per_nAm[:, 3 * source_idx: 3 * source_idx + 3]
    V_obs = L_eeg_src[:, 2].copy()
    B_pred = predict_meg_from_eeg(L_meg_src, L_eeg_src, V_obs)
    B_true = L_meg_src[:, 2]
    np.testing.assert_allclose(B_pred, B_true, atol=1e-8)
