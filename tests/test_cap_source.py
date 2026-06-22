"""CAP propagating-source-model tests."""
from __future__ import annotations

import numpy as np
import pytest

from inob.sources.cap import (
    biphasic_waveform,
    cap_signal,
    conduction_velocity_m_per_s,
    lognormal_fibre_distribution,
    longitudinal_leadfield,
)


def test_lognormal_distribution_normalised() -> None:
    fd = lognormal_fibre_distribution()
    assert fd.diameters_um.shape == fd.weights.shape
    np.testing.assert_allclose(fd.weights.sum(), 1.0)
    assert (fd.weights >= 0).all()


def test_cv_branch_myelinated_vs_unmyelinated() -> None:
    cv = conduction_velocity_m_per_s(np.array([0.5, 1.0, 2.0, 5.0, 10.0]))
    assert cv[0] == cv[1]                # both unmyelinated → constant 0.5
    assert cv[3] > cv[2]                 # myelinated CV scales with diameter
    assert cv[-1] == pytest.approx(60.0) # 10 µm × k=6 → 60 m/s


def test_biphasic_unit_peak() -> None:
    t = np.linspace(-3, 3, 1000)
    w = biphasic_waveform(t, ap_width_ms=0.5)
    assert np.abs(w).max() == pytest.approx(1.0)
    assert (w[t < 0] > 0).any() and (w[t > 0] < 0).any()


def test_longitudinal_leadfield_picks_tangent() -> None:
    # 3 sources along the x-axis, leadfield with only x-component non-zero
    src = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.float64)
    C, S = 4, 3
    L = np.zeros((C, 3 * S))
    L[:, 0::3] = 1.5     # x-moment column for every source
    L_long, arc, tangents = longitudinal_leadfield(L, src)
    assert L_long.shape == (C, S)
    np.testing.assert_allclose(L_long, 1.5)
    np.testing.assert_allclose(arc, [0, 10, 20])
    np.testing.assert_allclose(tangents[:, 0], 1.0, atol=1e-9)


def test_cap_signal_runs_and_shape() -> None:
    src = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.float64)
    L = np.ones((4, 9)) * 1e-6
    fibres = lognormal_fibre_distribution(n_bins=5)
    _t, sig = cap_signal(
        L, src, fibres=fibres,
        ap_width_ms=0.5, fs_hz=10_000.0, duration_ms=10.0,
    )
    assert sig.shape == (4, 100)
    assert np.isfinite(sig).all()
