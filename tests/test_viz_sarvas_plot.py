"""Tests for inob.viz.sarvas_plot (Sarvas-vs-FEM comparison figure)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pytest

from inob.analysis.sarvas_compare import SarvasGeometry, SarvasVsFemResult
from inob.viz.sarvas_plot import render_sarvas_vs_fem


def _make_result(*, C: int = 12, S: int = 5, Q_nAm: float = 1.0) -> SarvasVsFemResult:
    rng = np.random.default_rng(0)
    z = np.linspace(1400.0, 1500.0, S)
    source_pos = np.stack([np.zeros(S), np.zeros(S), z], axis=1)
    sphere_centres = np.stack([np.zeros(S), np.zeros(S), z], axis=1)
    geometry = SarvasGeometry(
        sphere_centres_mm=sphere_centres, axis_xy_mm=np.array([0.0, 0.0]),
    )
    sarvas_T = rng.standard_normal((C, S)) * 1e-12
    fem_T = sarvas_T * 1.1 + rng.standard_normal((C, S)) * 1e-14
    coil_pos = rng.standard_normal((C, 3)) * 60.0
    coil_orient = np.tile(np.array([1.0, 0.0, 0.0]), (C, 1))
    distance = np.abs(rng.standard_normal((C, S))) * 58.5
    return SarvasVsFemResult(
        geometry=geometry, Q_nAm=Q_nAm, source_pos_mm=source_pos,
        sarvas_T=sarvas_T, fem_T=fem_T, coil_pos_mm=coil_pos,
        coil_orient=coil_orient, distance_to_axis_mm=distance,
    )


def test_render_sarvas_vs_fem_writes_png(tmp_path: Path) -> None:
    result = _make_result()
    out_path = tmp_path / "sarvas_vs_fem.png"
    out = render_sarvas_vs_fem(result, out_path=out_path)
    assert out == out_path
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_sarvas_vs_fem_default_source_idx(tmp_path: Path) -> None:
    result = _make_result(S=7)
    out = render_sarvas_vs_fem(result, out_path=tmp_path / "a.png")
    assert out.exists()


def test_render_sarvas_vs_fem_explicit_source_idx(tmp_path: Path) -> None:
    result = _make_result(S=7)
    out = render_sarvas_vs_fem(result, source_idx=2, out_path=tmp_path / "b.png")
    assert out.exists()


def test_render_sarvas_vs_fem_nondefault_Q(tmp_path: Path) -> None:
    result = _make_result(Q_nAm=70.0)
    out = render_sarvas_vs_fem(result, out_path=tmp_path / "c.png")
    assert out.exists()


def test_render_sarvas_vs_fem_creates_parent_dirs(tmp_path: Path) -> None:
    result = _make_result()
    out_path = tmp_path / "nested" / "dir" / "fig.png"
    out = render_sarvas_vs_fem(result, out_path=out_path)
    assert out.exists()
