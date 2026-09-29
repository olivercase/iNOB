"""Model-informed against naive sensor placement.

These pin the comparison's *mechanics* — that it is a fair fight and that
the arms can be told apart when they genuinely differ. The finding on the
shipped anatomy (the two arms come out within a few percent) lives in the
module docstring, not here: it is a property of the body, not of the code,
and a test that asserted it would fail for the right reasons on a geometry
where the model does help.
"""

from __future__ import annotations

import numpy as np
import pytest

from inob.analysis.placement import (
    _total_info_bits,
    compare_placement,
    format_sweep_table,
    nearest_array_info,
    optimise_array,
    position_groups,
    sweep_aiming_error,
    sweep_sensor_counts,
)


def _array(n_pos: int = 60, n_src: int = 4, seed: int = 0):
    """Positions on a ring with a leadfield that falls off with distance."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n_pos, endpoint=False)
    coil_pos = np.column_stack([100.0 * np.cos(theta), 100.0 * np.sin(theta), np.zeros(n_pos)])
    source_pos = rng.standard_normal((n_src, 3)) * 20.0
    dist = np.linalg.norm(coil_pos[:, None, :] - source_pos[None, :, :], axis=2)
    base = 1.0e4 / dist**2  # (n_pos, n_src)
    lf = np.repeat(base, 3, axis=1)  # three moments per source
    lf = np.vstack([lf, 0.5 * lf, 0.25 * lf])  # R, T1, T2 blocks
    return lf, coil_pos, source_pos


def test_informed_is_never_worse_than_naive() -> None:
    """The informed arm maximises the score both arms are graded on."""
    lf, coil_pos, source_pos = _array()
    for src in range(len(source_pos)):
        cmp_ = compare_placement(
            lf, coil_pos, source_pos, source_idx=src, n_sensors=10, noise_floor=200.0
        )
        assert cmp_.informed.amplitude >= cmp_.naive.amplitude - 1e-9
        assert cmp_.amplitude_ratio >= 1.0 - 1e-9


def test_trials_ratio_is_the_amplitude_ratio_squared() -> None:
    """Trials go as 1/SNR^2, so the two ratios must stay locked together."""
    lf, coil_pos, source_pos = _array()
    cmp_ = compare_placement(
        lf, coil_pos, source_pos, source_idx=0, n_sensors=10, noise_floor=200.0
    )
    assert cmp_.trials_ratio == pytest.approx(cmp_.amplitude_ratio**2, rel=1e-9)


def test_both_arms_respect_the_minimum_spacing() -> None:
    """Neither arm may win by crowding sensors the other could not use."""
    lf, coil_pos, source_pos = _array()
    cmp_ = compare_placement(
        lf,
        coil_pos,
        source_pos,
        source_idx=0,
        n_sensors=8,
        min_spacing_mm=40.0,
        noise_floor=200.0,
    )
    for arm in (cmp_.informed, cmp_.naive):
        chosen = coil_pos[arm.position_idx]
        gaps = np.linalg.norm(chosen[:, None, :] - chosen[None, :, :], axis=2)
        np.fill_diagonal(gaps, np.inf)
        assert gaps.min() >= 40.0 - 1e-9


def test_both_arms_select_the_requested_number_of_sensors() -> None:
    lf, coil_pos, source_pos = _array()
    cmp_ = compare_placement(
        lf, coil_pos, source_pos, source_idx=0, n_sensors=12, noise_floor=200.0
    )
    assert len(cmp_.informed.position_idx) == 12
    assert len(cmp_.naive.position_idx) == 12


def test_the_gap_opens_when_coupling_stops_tracking_distance() -> None:
    """The comparison can detect a real advantage where one exists.

    Here the nearest positions are deliberately the worst-coupled — a stand-in
    for a shield the naive arm cannot see and the forward model can.
    """
    lf, coil_pos, source_pos = _array()
    n_pos = len(coil_pos)
    dist = np.linalg.norm(coil_pos - source_pos[0], axis=1)
    # Blind the whole stretch the naive arm can reach at this spacing.
    blind = np.argsort(dist)[:30]  # the closest positions carry no signal
    for block in range(3):
        lf[block * n_pos + blind, 0:3] = 1e-6

    cmp_ = compare_placement(
        lf, coil_pos, source_pos, source_idx=0, n_sensors=10, noise_floor=200.0
    )
    # Against ~1.0 on the shipped anatomy, this is an unambiguous signal.
    assert cmp_.amplitude_ratio > 3.0


def test_naive_aim_moves_only_the_naive_arm() -> None:
    lf, coil_pos, source_pos = _array()
    kwargs = dict(source_idx=0, n_sensors=10, noise_floor=200.0)
    true = compare_placement(lf, coil_pos, source_pos, **kwargs)
    aimed = compare_placement(lf, coil_pos, source_pos, naive_aim=source_pos[0] + 60.0, **kwargs)
    assert aimed.informed.amplitude == pytest.approx(true.informed.amplitude)
    assert aimed.naive.amplitude < true.naive.amplitude


def test_sweep_aiming_error_starts_at_perfect_knowledge() -> None:
    lf, coil_pos, source_pos = _array()
    got = sweep_aiming_error(
        lf,
        coil_pos,
        source_pos,
        source_idx=0,
        errors_mm=(0.0, 40.0),
        n_draws=8,
        n_sensors=8,
        noise_floor=200.0,
    )
    assert set(got) == {0.0, 40.0}
    assert got[0.0] >= 1.0 - 1e-9
    assert got[40.0] >= got[0.0] - 1e-9


def test_sweep_sensor_counts_skips_counts_the_array_cannot_supply() -> None:
    lf, coil_pos, source_pos = _array(n_pos=30)
    got = sweep_sensor_counts(
        lf, coil_pos, source_pos, source_idx=0, counts=(4, 16, 512), noise_floor=200.0
    )
    assert [c.n_sensors for c in got] == [4, 16]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"source_idx": 99}, "outside"),
        ({"n_sensors": 0}, "must be >= 1"),
        ({"n_sensors": 10_000}, "only"),
        ({"noise_floor": 0.0}, "must be positive"),
    ],
)
def test_compare_placement_rejects_bad_inputs(kwargs: dict, match: str) -> None:
    lf, coil_pos, source_pos = _array()
    call = {"source_idx": 0, "n_sensors": 10, "noise_floor": 200.0} | kwargs
    with pytest.raises(ValueError, match=match):
        compare_placement(lf, coil_pos, source_pos, **call)


def test_compare_placement_rejects_a_non_triaxial_leadfield() -> None:
    lf, coil_pos, source_pos = _array()
    with pytest.raises(ValueError, match="not divisible by 3"):
        compare_placement(
            lf[:-1], coil_pos, source_pos, source_idx=0, n_sensors=4, noise_floor=200.0
        )


def test_format_sweep_table_reports_every_array_size() -> None:
    lf, coil_pos, source_pos = _array()
    sweep = sweep_sensor_counts(
        lf, coil_pos, source_pos, source_idx=0, counts=(4, 16), noise_floor=200.0
    )
    table = format_sweep_table(sweep)
    assert "sensors" in table and "informed" in table and "naive" in table
    for line_start in ("       4  ", "      16  "):
        assert line_start in table


def test_format_sweep_table_handles_an_empty_sweep() -> None:
    assert "no array sizes" in format_sweep_table([])


# ── information-capacity array optimisation ────────────────────────────────


def _two_source_problem(n_per_side: int = 10):
    """Positions in two clusters, each seeing only its own source strongly."""
    rng = np.random.default_rng(3)
    left = np.column_stack(
        [rng.uniform(-10, 10, n_per_side), np.zeros(n_per_side), np.arange(n_per_side) * 40.0]
    )
    right = left + np.array([500.0, 0.0, 0.0])
    pos = np.vstack([left, right])
    n = len(pos)
    L1 = np.zeros((3 * n, 6))
    # Left cluster sees source 0 at strength 10, right sees source 1 at 3:
    # a coupling ranking would fill up on the left before touching the right.
    for i in range(n):
        s, g = (0, 10.0) if i < n_per_side else (1, 3.0)
        for a in range(3):
            L1[a * n + i, 3 * s : 3 * s + 3] = g * rng.normal(size=3)
    coil_pos = np.vstack([pos, pos, pos])
    return L1, coil_pos, np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]])


def test_optimise_array_information_matches_direct_logdet() -> None:
    L, coil_pos, _ = _two_source_problem()
    res = optimise_array(
        L, coil_pos, n_sensors=6, noise_floor=2.0, moment_nAm=1.5, min_spacing_mm=0.0
    )
    X = L[res.channel_idx] * (1.5 / 2.0)
    assert res.info_bits[-1] == pytest.approx(_total_info_bits(X), rel=1e-9)
    assert np.all(np.diff(res.info_bits) > 0)
    assert res.info_bits[-1] <= res.full_info_bits + 1e-9


def test_optimise_array_covers_the_weaker_source_too() -> None:
    """Diminishing returns on one source must send sensors to the other."""
    L, coil_pos, _ = _two_source_problem()
    res = optimise_array(
        L, coil_pos, n_sensors=8, noise_floor=1.0, moment_nAm=1.0, min_spacing_mm=0.0
    )
    assert np.any(res.positions[res.position_idx, 0] > 250.0)
    assert np.any(res.positions[res.position_idx, 0] < 250.0)


def test_optimise_array_respects_spacing_and_single_axis() -> None:
    L, coil_pos, targets = _two_source_problem()
    n = len(coil_pos) // 3
    single = optimise_array(
        L[:n], coil_pos[:n], n_sensors=4, noise_floor=1.0, moment_nAm=1.0, min_spacing_mm=100.0
    )
    chosen = single.positions[single.position_idx]
    d = np.linalg.norm(chosen[:, None] - chosen[None], axis=2)
    assert d[np.triu_indices(len(chosen), 1)].min() >= 100.0
    assert len(single.channel_idx) == 4  # one channel per position
    base = nearest_array_info(
        L, coil_pos, targets, n_sensors=4, noise_floor=1.0, moment_nAm=1.0, min_spacing_mm=0.0
    )
    best = optimise_array(
        L, coil_pos, n_sensors=4, noise_floor=1.0, moment_nAm=1.0, min_spacing_mm=0.0
    )
    assert best.info_bits[-1] >= base - 1e-9


def test_position_groups_keeps_triaxial_channels_together() -> None:
    pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    uniq, groups = position_groups(np.vstack([pos, pos, pos]))
    assert len(uniq) == 2
    assert [g.tolist() for g in groups] == [[0, 2, 4], [1, 3, 5]]
