"""Smoke tests for the placement-comparison figure.

The array is built here rather than taken from ``viz_pipeline_helpers``
because that helper interleaves the R/T1/T2 channels, while
:mod:`inob.sensors.triaxial` writes them in blocks and
:mod:`inob.analysis.placement` reads the blocked layout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from inob.config import load_config
from inob.io.npz import Leadfield, save_leadfield
from inob.sensors.triaxial import build_triaxial
from inob.viz.placement_comparison import render_placement_comparison

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"

N_SOURCES = 6


def _blocked_leadfield(n_pos: int = 40) -> Leadfield:
    """A triaxial array on a ring, in the block layout production writes."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_pos, endpoint=False)
    pos = np.column_stack([100.0 * np.cos(theta), 100.0 * np.sin(theta), np.zeros(n_pos)])
    normals = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    array = build_triaxial(pos, normals)

    source_pos = np.column_stack(
        [np.zeros(N_SOURCES), np.zeros(N_SOURCES), np.linspace(-40.0, 40.0, N_SOURCES)]
    )
    dist = np.linalg.norm(pos[:, None, :] - source_pos[None, :, :], axis=2)
    base = 1.0e4 / dist**2
    per_position = np.repeat(base, 3, axis=1)  # three moments per source
    lf = np.vstack([per_position, 0.5 * per_position, 0.25 * per_position])
    return Leadfield(
        L=lf * 1e-6,
        L_fT_per_nAm=lf,
        source_pos=source_pos,
        coil_pos=array.coilpos,
        coil_orient=array.coilori,
        channel_names=array.labels,
        conductivities=np.array([0.3, 0.43]),
        tissue_labels=("vagus_left", "skin"),
        seed=0,
    )


def test_render_placement_comparison_writes_a_png(tmp_path) -> None:
    cfg = load_config(TINY_CFG, project_root=tmp_path)
    npz = tmp_path / "lf.npz"
    save_leadfield(npz, _blocked_leadfield())
    out, sweep = render_placement_comparison(cfg, forward_npz=npz, n_sensors=6, dpi=60)
    assert out.exists() and out.stat().st_size > 0
    assert [c.n_sensors for c in sweep] == [4, 8, 16, 32]  # capped by the array


def test_render_placement_comparison_honours_an_explicit_out_path(tmp_path) -> None:
    cfg = load_config(TINY_CFG, project_root=tmp_path)
    npz = tmp_path / "lf.npz"
    save_leadfield(npz, _blocked_leadfield())
    out = tmp_path / "custom.png"
    got, _ = render_placement_comparison(cfg, forward_npz=npz, n_sensors=6, out_path=out, dpi=60)
    assert got == out and out.exists()
