"""Tests for the Sarvas-vs-FEM comparison helpers (no DUNEuro/leadfield needed)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inob.analysis.sarvas_compare import (
    SarvasGeometry,
    _bootstrap_ratio_ci,
    _radial_coil_mask,
    build_sphere_centres,
    estimate_cervical_axis_xy,
    hamalainen_dipole_moment_nAm,
    sarvas_predict_at_coils,
    save_comparison_summary,
    vagus_tangents,
)
from inob.io.hdf5 import FemMesh


def test_hamalainen_dipole_moment_matches_hand_calc() -> None:
    # Q = pi * d^2 * sigma * dV / 4, with d in m, dV in V, sigma in S/m.
    Q = hamalainen_dipole_moment_nAm(
        fibre_diameter_um=10.0, sigma_intracellular_S_per_m=1.0,
        action_potential_mV=80.0,
    )
    d_m = 10.0e-6
    dV_V = 80.0e-3
    expected_Am = np.pi * d_m ** 2 * 1.0 * dV_V / 4.0
    assert Q == pytest.approx(expected_Am * 1e9)
    assert Q > 0


def test_hamalainen_dipole_moment_scales_with_diameter_squared() -> None:
    Q1 = hamalainen_dipole_moment_nAm(
        fibre_diameter_um=5.0, sigma_intracellular_S_per_m=1.0,
        action_potential_mV=70.0,
    )
    Q2 = hamalainen_dipole_moment_nAm(
        fibre_diameter_um=10.0, sigma_intracellular_S_per_m=1.0,
        action_potential_mV=70.0,
    )
    assert Q2 == pytest.approx(Q1 * 4.0)


def _fem_with_bone_and_vagus() -> FemMesh:
    # Two "bone" tets centred at (10, 0, z) and two "vagus_left" tets whose
    # centroid ends up off-axis, so the weighted estimate is checkable.
    nodes = np.array([
        [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0],
        [10.0, 0.0, 5.0], [11.0, 0.0, 5.0], [10.0, 1.0, 5.0], [10.0, 0.0, 6.0],
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    tets = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ], dtype=np.int32)
    tissue = np.array([1, 1, 2], dtype=np.int32)  # bone, bone, vagus_left
    return FemMesh(nodes=nodes, tets=tets, tissue=tissue,
                    tissue_labels=("bone", "vagus_left"), unit="mm")


def test_estimate_cervical_axis_xy_weights_toward_bone() -> None:
    fem = _fem_with_bone_and_vagus()
    src_pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    axis_xy = estimate_cervical_axis_xy(fem, src_pos)
    # bone centroid XY ~ (10.25, 0.25); source XY ~ (0.25, 0.25).
    # 2:1 weight toward bone => axis_x should be closer to 10 than to 0.
    assert axis_xy.shape == (2,)
    assert axis_xy[0] > 5.0


def test_estimate_cervical_axis_xy_falls_back_without_bone() -> None:
    nodes = np.array([
        [2.0, 4.0, 0.0], [3.0, 4.0, 0.0], [2.0, 5.0, 0.0], [2.0, 4.0, 1.0],
    ], dtype=np.float64)
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    tissue = np.array([1], dtype=np.int32)
    fem = FemMesh(nodes=nodes, tets=tets, tissue=tissue,
                   tissue_labels=("vagus_left",), unit="mm")
    src_pos = nodes[:1].repeat(2, axis=0)
    axis_xy = estimate_cervical_axis_xy(fem, src_pos)
    np.testing.assert_allclose(axis_xy, src_pos[:, :2].mean(axis=0))


def test_build_sphere_centres_places_z_per_source() -> None:
    src_pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    axis_xy = np.array([100.0, 200.0])
    centres = build_sphere_centres(src_pos, axis_xy)
    assert centres.shape == src_pos.shape
    np.testing.assert_array_equal(centres[:, 0], 100.0)
    np.testing.assert_array_equal(centres[:, 1], 200.0)
    np.testing.assert_array_equal(centres[:, 2], src_pos[:, 2])


def test_sarvas_predict_at_coils_zero_off_axis_gives_zero_field() -> None:
    # A radial dipole (moment along r0) produces zero magnetic field
    # everywhere (Sarvas/Biot-Savart symmetry for the radial component).
    sphere_centre = np.zeros(3)
    source_pos_mm = np.array([0.0, 0.0, 40.0])   # r0 along +Z
    moment_direction = np.array([0.0, 0.0, 1.0])  # radial moment
    coil_pos_mm = np.array([[60.0, 0.0, 0.0], [0.0, 60.0, 0.0]])
    coil_orient = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    B = sarvas_predict_at_coils(
        source_pos_mm, moment_direction, Q_nAm=70.0,
        coil_pos_mm=coil_pos_mm, coil_orient=coil_orient,
        sphere_centre_mm=sphere_centre,
    )
    assert B.shape == (2,)
    np.testing.assert_allclose(B, 0.0, atol=1e-20)


def test_sarvas_predict_at_coils_tangential_moment_nonzero() -> None:
    sphere_centre = np.zeros(3)
    source_pos_mm = np.array([0.0, 0.0, 40.0])
    moment_direction = np.array([1.0, 0.0, 0.0])  # tangential moment
    coil_pos_mm = np.array([[0.0, 60.0, 0.0]])
    coil_orient = np.array([[0.0, 0.0, 1.0]])
    B = sarvas_predict_at_coils(
        source_pos_mm, moment_direction, Q_nAm=70.0,
        coil_pos_mm=coil_pos_mm, coil_orient=coil_orient,
        sphere_centre_mm=sphere_centre,
    )
    assert B.shape == (1,)
    assert np.abs(B[0]) > 0


def test_vagus_tangents_straight_line() -> None:
    pos = np.column_stack([
        np.zeros(5), np.zeros(5), np.arange(5, dtype=np.float64) * 2.0,
    ])
    tangents = vagus_tangents(pos)
    assert tangents.shape == pos.shape
    for row in tangents:
        np.testing.assert_allclose(np.linalg.norm(row), 1.0)
        np.testing.assert_allclose(row, [0.0, 0.0, 1.0], atol=1e-8)


def test_vagus_tangents_two_points() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    tangents = vagus_tangents(pos)
    assert tangents.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)


def test_radial_coil_mask_triaxial_layout() -> None:
    # Block layout: all R channels first, then all T1, then all T2 —
    # matches conftest.tiny_sensors and inob.sensors.triaxial output.
    labels = (
        [f"mag-{i:04d}-R" for i in range(4)]
        + [f"mag-{i:04d}-T1" for i in range(4)]
        + [f"mag-{i:04d}-T2" for i in range(4)]
    )
    mask = _radial_coil_mask(labels)
    assert mask.sum() == 4
    for lab, keep in zip(labels, mask):
        assert keep == lab.endswith("-R")


def test_radial_coil_mask_fallback_suffix_only() -> None:
    labels = ["a-R", "a-T1", "b-R", "b-T2"]
    mask = _radial_coil_mask(labels)
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_bootstrap_ratio_ci_deterministic_and_ordered() -> None:
    rng = np.random.default_rng(1)
    sarvas = rng.normal(scale=1.0, size=50)
    fem = sarvas * 3.0
    lo1, med1, hi1 = _bootstrap_ratio_ci(sarvas_band=sarvas, fem_band=fem, n_boot=200, seed=0)
    lo2, med2, hi2 = _bootstrap_ratio_ci(sarvas_band=sarvas, fem_band=fem, n_boot=200, seed=0)
    assert (lo1, med1, hi1) == (lo2, med2, hi2)
    assert lo1 <= med1 <= hi1
    # fem is an exact scalar multiple of sarvas => ratio is always ~3.0
    assert med1 == pytest.approx(3.0, rel=1e-6)


def test_bootstrap_ratio_ci_empty_input() -> None:
    lo, med, hi = _bootstrap_ratio_ci(
        sarvas_band=np.array([]), fem_band=np.array([]),
    )
    assert np.isnan(lo) and np.isnan(med) and np.isnan(hi)


def test_save_comparison_summary_writes_expected_keys(tmp_path: Path) -> None:
    from inob.analysis.sarvas_compare import SarvasVsFemResult

    S, C = 3, 2
    axis_xy = np.array([0.0, 0.0])
    source_pos = np.array([[40.0, 0.0, 0.0], [40.0, 0.0, 5.0], [40.0, 0.0, 10.0]])
    centres = build_sphere_centres(source_pos, axis_xy)
    geom = SarvasGeometry(sphere_centres_mm=centres, axis_xy_mm=axis_xy)
    coil_pos = np.array([[98.5, 0.0, 0.0], [0.0, 98.5, 5.0]])
    distances = np.linalg.norm(
        coil_pos[:, None, :] - centres[None, :, :], axis=2,
    )
    result = SarvasVsFemResult(
        geometry=geom,
        Q_nAm=1.0,
        source_pos_mm=source_pos,
        sarvas_T=np.full((C, S), 1e-15),
        fem_T=np.full((C, S), 3e-15),
        coil_pos_mm=coil_pos,
        coil_orient=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        distance_to_axis_mm=distances,
    )
    out = save_comparison_summary(result, tmp_path / "sub" / "summary.json")
    assert out.exists()
    data = json.loads(out.read_text())
    for key in (
        "Q_nAm", "n_sources", "n_radial_coils", "n_band_pairs",
        "sarvas_peak_fT_per_nAm_full", "fem_peak_fT_per_nAm_full",
        "ratio_fem_to_sarvas_peak_band",
    ):
        assert key in data
    assert data["n_sources"] == S
    assert data["n_radial_coils"] == C
