"""Tests for inob.viz.cap_compare (propagating vs stationary CAP figure)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pytest

from inob.config import load_config
from inob.io.npz import Leadfield, save_leadfield
from inob.viz.cap_compare import _fwhm_ms, render_cap_compare

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


@pytest.fixture
def cfg(tmp_path: Path):
    return load_config(TINY_CFG, project_root=tmp_path)


@pytest.fixture
def cervical_leadfield(cfg) -> Leadfield:
    """A synthetic leadfield along a 1500 mm polyline (mimics the vagus)."""
    S = 40
    C = 12  # 4 positions x {R, T1, T2}
    rng = np.random.default_rng(0)
    z = np.linspace(0.0, 1499.0, S)
    source_pos = np.stack([np.zeros(S), np.zeros(S), z], axis=1)
    L = rng.standard_normal((C, 3 * S)) * 1e-12
    coil_pos = np.repeat(
        np.array([[60.0, 0.0, 0.0], [-60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, -60.0, 0.0]]),
        3, axis=0,
    )
    coil_orient = np.tile(np.eye(3), (4, 1))
    labels = tuple(
        f"mag-{i // 3:04d}-{['R', 'T1', 'T2'][i % 3]}" for i in range(C)
    )
    lf = Leadfield(
        L=L, L_fT_per_nAm=L * 1e6, source_pos=source_pos,
        coil_pos=coil_pos, coil_orient=coil_orient, channel_names=labels,
        conductivities=np.array([3e-4, 4.3e-4]),
        tissue_labels=("vagus_left", "skin"), seed=0,
    )
    save_leadfield(cfg.outputs.forward_npz, lf)
    return lf


def test_fwhm_ms_basic() -> None:
    t = np.linspace(-5, 5, 1001)
    x = np.exp(-t ** 2 / (2 * 1.0 ** 2))  # sigma=1 Gaussian, FWHM ~= 2.355
    fwhm = _fwhm_ms(x, t)
    assert 2.0 < fwhm < 2.7


def test_fwhm_ms_single_nonzero_sample_returns_nan() -> None:
    t = np.linspace(0, 1, 10)
    x = np.zeros(10)
    x[0] = 1.0  # only one sample reaches the half-max threshold
    assert np.isnan(_fwhm_ms(x, t))


def test_render_cap_compare_writes_png(cfg, cervical_leadfield: Leadfield, tmp_path: Path) -> None:
    out_path = tmp_path / "cap_compare.png"
    out = render_cap_compare(
        cfg, n_fibres=10, duration_ms=6.0, fs_hz=5_000.0, out_path=out_path,
    )
    assert out == out_path
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_cap_compare_default_out_path(cfg, cervical_leadfield: Leadfield) -> None:
    out = render_cap_compare(cfg, n_fibres=10, duration_ms=6.0, fs_hz=5_000.0)
    # Figures are target-tagged (multi-target namespacing): the default vagus
    # leadfield yields the `_vagus` suffix so a spine run can't overwrite it.
    assert out == cfg.outputs.base / "cap_compare_vagus.png"
    assert out.exists()
