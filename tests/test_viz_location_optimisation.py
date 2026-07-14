"""Tests for inob.viz.location_optimisation: pure helpers + render smoke test."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from inob.io.hdf5 import save_sensors
from inob.io.npz import save_leadfield
from inob.viz.location_optimisation import (
    _column_for_source_norm,
    _per_source_peak,
    render_location_optimisation,
)

from tests.viz_pipeline_helpers import (
    _build_electrodes,
    _build_leadfield,
    build_pipeline_cfg,
)


def test_per_source_peak_shape_and_sign() -> None:
    rng = np.random.default_rng(0)
    L = rng.standard_normal((6, 9))
    peak = _per_source_peak(L)
    assert peak.shape == (3,)
    assert (peak >= 0).all()


def test_per_source_peak_matches_manual_computation() -> None:
    L = np.arange(18, dtype=float).reshape(2, 9) - 9.0
    peak = _per_source_peak(L)
    L3 = L.reshape(2, 3, 3)
    expected = np.abs(L3).max(axis=(0, 2))
    np.testing.assert_allclose(peak, expected)


def test_column_for_source_norm_shape() -> None:
    rng = np.random.default_rng(1)
    L = rng.standard_normal((5, 12))
    col = _column_for_source_norm(L, source_idx=1)
    assert col.shape == (5,)
    expected = np.linalg.norm(L[:, 3:6], axis=1)
    np.testing.assert_allclose(col, expected)


def test_column_for_source_norm_always_nonnegative() -> None:
    rng = np.random.default_rng(2)
    L = rng.standard_normal((4, 6)) * -5.0
    col = _column_for_source_norm(L, source_idx=0)
    assert (col >= 0).all()


def _build_wholebody(tmp_path: Path):
    wb_sensors = _build_electrodes(rows=8, cols=8, pitch=20.0)
    wb_mat = tmp_path / "wb_sensors.mat"
    save_sensors(wb_mat, wb_sensors)
    wb_lf = _build_leadfield(wb_sensors, seed=9, scale=8.0)
    wb_npz = tmp_path / "wb_leadfield.npz"
    save_leadfield(wb_npz, wb_lf)
    return wb_npz, wb_mat


def test_render_location_optimisation_smoke(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    wb_npz, wb_mat = _build_wholebody(tmp_path)

    out = render_location_optimisation(
        cfg,
        paddle_npz=cfg.outputs.forward_eeg_npz,
        wholebody_npz=wb_npz,
        paddle_mat=cfg.outputs.electrodes_mat,
        wholebody_mat=wb_mat,
        out_path=tmp_path / "loc.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_location_optimisation_default_out_path(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    wb_npz, wb_mat = _build_wholebody(tmp_path)

    out = render_location_optimisation(
        cfg,
        paddle_npz=cfg.outputs.forward_eeg_npz,
        wholebody_npz=wb_npz,
        paddle_mat=cfg.outputs.electrodes_mat,
        wholebody_mat=wb_mat,
        source_idx=3,
    )
    assert out == cfg.outputs.base / "location_optimisation.png"
    assert out.exists()
