"""Tests for inob.viz.topoplot: pure helpers + render smoke tests."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from inob.viz.topoplot import (
    TopoFrame,
    _column_for_source,
    _grid_shape_from_labels,
    _paddle_uv_from_labels,
    _radial_channel_mask,
    render_dual_topoplot,
    render_eeg_topoplot,
    render_meg_montage,
    render_meg_topoplot,
)
from tests.viz_pipeline_helpers import build_pipeline_cfg

# ── _column_for_source ──────────────────────────────────────────────────────


def test_column_for_source_moments() -> None:
    L = np.arange(2 * 6, dtype=float).reshape(2, 6)
    x = _column_for_source(L, 0, "x")
    y = _column_for_source(L, 0, "y")
    z = _column_for_source(L, 0, "z")
    np.testing.assert_allclose(x, L[:, 0])
    np.testing.assert_allclose(y, L[:, 1])
    np.testing.assert_allclose(z, L[:, 2])


def test_column_for_source_rms_and_norm() -> None:
    L = np.array([[3.0, 4.0, 0.0]])
    rms = _column_for_source(L, 0, "rms")
    norm = _column_for_source(L, 0, "norm")
    assert rms[0] == pytest.approx(np.sqrt((9 + 16 + 0) / 3))
    assert norm[0] == pytest.approx(5.0)


def test_column_for_source_out_of_range_raises() -> None:
    L = np.zeros((2, 6))
    with pytest.raises(IndexError):
        _column_for_source(L, 5, "z")


def test_column_for_source_unknown_moment_raises() -> None:
    L = np.zeros((2, 3))
    with pytest.raises(ValueError, match="unknown moment"):
        _column_for_source(L, 0, "bogus")


# ── _radial_channel_mask ────────────────────────────────────────────────────


def test_radial_channel_mask_triaxial_block_layout() -> None:
    labels = [f"mag-{i:04d}-{m}" for i in range(3) for m in ("R", "T1", "T2")]
    mask = _radial_channel_mask(labels)
    assert mask.sum() == 3
    assert all(lab.endswith("-R") for lab in np.array(labels)[mask])


def test_radial_channel_mask_fallback_suffix_scan() -> None:
    labels = ["a-R", "a-T1", "b-R", "b-T2"]
    mask = _radial_channel_mask(labels)
    np.testing.assert_array_equal(mask, [True, False, True, False])


# ── _grid_shape_from_labels ──────────────────────────────────────────────────


def test_grid_shape_from_labels_rectangular() -> None:
    labels = tuple(f"elec-{r:02d}-{c:02d}" for r in range(2) for c in range(4))
    assert _grid_shape_from_labels(labels) == (2, 4)


def test_grid_shape_from_labels_non_rectangular_returns_none() -> None:
    labels = ("elec-head-00", "elec-00-00", "elec-foot-00")
    assert _grid_shape_from_labels(labels) is None


# ── _paddle_uv_from_labels ───────────────────────────────────────────────────


def test_paddle_uv_from_labels_none_when_no_head_foot() -> None:
    labels = tuple(f"elec-{r:02d}-{c:02d}" for r in range(2) for c in range(2))
    assert (
        _paddle_uv_from_labels(labels, pitch_mm=5.0, head_offset_mm=15.0, foot_offset_mm=15.0)
        is None
    )


def test_paddle_uv_from_labels_shape_and_head_foot_placement() -> None:
    labels = ("elec-head-00", "elec-00-00", "elec-00-01", "elec-foot-00")
    uv = _paddle_uv_from_labels(labels, pitch_mm=5.0, head_offset_mm=10.0, foot_offset_mm=10.0)
    assert uv is not None
    assert uv.shape == (4, 2)
    # head is above the body rows, foot below
    assert uv[0, 1] > uv[1, 1]
    assert uv[3, 1] < uv[1, 1]


# ── TopoFrame ────────────────────────────────────────────────────────────────


def test_topoframe_defaults() -> None:
    tf = TopoFrame(source_idx=2)
    assert tf.moment == "z"


# ── render smoke tests ───────────────────────────────────────────────────────


def test_render_meg_topoplot_returns_fig_and_axes(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    fig, ax = render_meg_topoplot(cfg)
    assert isinstance(fig, plt.Figure)
    assert ax in fig.axes
    plt.close(fig)


def test_render_meg_topoplot_reuses_given_axis(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig2, ax2 = render_meg_topoplot(cfg, ax=ax)
    assert fig2 is fig
    assert ax2 is ax
    plt.close(fig)


def test_render_eeg_topoplot_with_and_without_3d_context(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    fig1 = render_eeg_topoplot(cfg, with_3d_context=True)
    # 2-D panel + 3-D panel + colorbar axis
    assert len(fig1.axes) >= 3
    assert any(ax.name == "3d" for ax in fig1.axes)
    plt.close(fig1)

    fig2 = render_eeg_topoplot(cfg, with_3d_context=False)
    assert len(fig2.axes) >= 1
    plt.close(fig2)


def test_render_dual_topoplot_smoke(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    out = render_dual_topoplot(cfg, out_path=tmp_path / "dual.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_meg_montage_smoke(tmp_path: Path) -> None:
    cfg = build_pipeline_cfg(tmp_path)
    out = render_meg_montage(cfg, n_sources=3, out_path=tmp_path / "montage.png")
    assert out.exists()
    assert out.stat().st_size > 0
