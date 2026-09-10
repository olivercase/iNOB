"""Per-target physiology profiles: dispatch, generality, and no vagus drift.

The point of :mod:`inob.physiology.profiles` is that adding spinal physiology
must not change what a vagus run produces, and that nothing downstream hard-
codes one target's assumptions onto another. These tests pin both directions.
"""

from __future__ import annotations

import numpy as np
import pytest

from inob.config import SOURCE_TARGETS
from inob.physiology.profiles import (
    PROFILES,
    SPINE_PROFILE,
    VAGUS_PROFILE,
    profile_for_tag,
)
from inob.physiology.scenarios import (
    median_nerve_ssep_scenario,
    tibial_nerve_ssep_scenario,
)
from inob.physiology.simulate import resolve_source_index

# The literal values the package used before profiles existed. If a change to
# the fibre model moves any of these, a vagus figure silently changes with it.
VAGUS_LEGACY = {
    "n_fibres": 200,
    "ap_width_ms": 0.5,
    "ap_amplitude_mV": 70.0,
    "sigma_in_Sm": 1.0,
    "propagation_span_mm": 50.0,
    "default_strength_nAm": 70.0,
}


@pytest.mark.parametrize(("field", "expected"), sorted(VAGUS_LEGACY.items()))
def test_vagus_profile_preserves_legacy_defaults(field: str, expected) -> None:
    assert getattr(VAGUS_PROFILE, field) == expected


def test_every_source_target_has_a_profile() -> None:
    """A new --source-target must not silently fall back to vagus physiology."""
    missing = set(SOURCE_TARGETS) - set(PROFILES)
    assert not missing, f"source targets without a physiology profile: {missing}"


def test_unknown_tag_falls_back_but_is_marked_provisional() -> None:
    p = profile_for_tag("kidney")
    assert p.validated is False
    assert "PROVISIONAL" in p.generator


def test_empty_tag_is_vagus() -> None:
    """An untagged legacy leadfield keeps the historical vagus behaviour."""
    assert profile_for_tag("") is VAGUS_PROFILE


def test_spine_and_vagus_differ_in_the_ways_that_matter() -> None:
    """The profiles must be genuinely distinct, not a relabelling."""
    assert SPINE_PROFILE.mean_cv_m_per_s > VAGUS_PROFILE.mean_cv_m_per_s
    assert SPINE_PROFILE.ap_width_ms > VAGUS_PROFILE.ap_width_ms
    # No localised generator: the volley crosses the whole modelled cord.
    assert SPINE_PROFILE.propagation_span_mm is None
    assert VAGUS_PROFILE.propagation_span_mm == 50.0
    # ...which is exactly why the stationary approximation cannot be reused.
    assert VAGUS_PROFILE.stationary_ok is True
    assert SPINE_PROFILE.stationary_ok is False


def test_spine_conduction_velocity_matches_magnetospinography() -> None:
    """Human cervical dorsal-column CV is 55-70 m/s (Kawabata 2002, Sasaki 2008)."""
    assert 55.0 <= SPINE_PROFILE.mean_cv_m_per_s <= 70.0


def test_spine_event_moment_is_single_nAm_scale() -> None:
    """Reported cord-response equivalent dipoles are single-nA·m, not tens."""
    assert 1.0 <= SPINE_PROFILE.event_moment_nAm <= 10.0


def test_stationary_ok_agrees_with_transit_vs_ap_width() -> None:
    """``stationary_ok`` must not contradict the profile's own numbers."""
    for name, p in PROFILES.items():
        span = p.propagation_span_mm
        if span is None:
            continue  # whole-polyline span is only known at render time
        ratio = p.transit_ms(span) / p.ap_width_ms
        if p.stationary_ok:
            assert ratio < 5.0, f"{name}: transit/AP = {ratio:.1f} but marked OK"


def test_transit_scales_with_span() -> None:
    assert SPINE_PROFILE.transit_ms(400) == pytest.approx(4 * SPINE_PROFILE.transit_ms(100))


# ── SSEP scenarios ─────────────────────────────────────────────────────────


def test_ssep_entry_segments_are_far_apart() -> None:
    """Median enters cervically, tibial lumbosacrally — the depth difference
    is the whole reason both are simulated."""
    med = median_nerve_ssep_scenario()
    tib = tibial_nerve_ssep_scenario()
    assert med.generator_z_mm > tib.generator_z_mm + 250.0


def test_ssep_scenarios_carry_their_own_ap_width() -> None:
    med = median_nerve_ssep_scenario()
    tib = tibial_nerve_ssep_scenario()
    # Longer peripheral path disperses the tibial volley further.
    assert tib.ap_width_ms > med.ap_width_ms
    assert med.ap_width_ms == SPINE_PROFILE.ap_width_ms


def test_ssep_event_count_follows_rate_and_duration() -> None:
    sc = median_nerve_ssep_scenario(duration_s=4.0, rate_hz=5.0)
    assert len(sc.events) == 21  # 4 s x 5 Hz + 1


def test_ssep_rate_avoids_mains_harmonics() -> None:
    """Non-integer rates keep 50/60 Hz line noise from summing across trials."""
    for sc in (median_nerve_ssep_scenario(), tibial_nerve_ssep_scenario()):
        assert 50.0 % sc.rate_hz != 0.0
        assert 60.0 % sc.rate_hz != 0.0


def test_spine_profile_builds_both_ssep_scenarios() -> None:
    names = {s.name for s in SPINE_PROFILE.scenarios()}
    assert names == {"ssep_median", "ssep_tibial"}


def test_vagus_profile_builds_its_own_scenarios() -> None:
    names = {s.name for s in VAGUS_PROFILE.scenarios()}
    assert names == {"baroreceptor", "respiratory_deep"}


def test_muscle_profile_refuses_to_invent_scenarios() -> None:
    """Muscle has no MUAP model; it must raise rather than reuse vagal ones."""
    with pytest.raises(NotImplementedError, match="no event-train scenarios"):
        PROFILES["muscle"].scenarios()


# ── generator placement ────────────────────────────────────────────────────


def _positions(z_values) -> np.ndarray:
    pos = np.zeros((len(z_values), 3))
    pos[:, 2] = z_values
    return pos


def test_generator_z_selects_the_nearest_source() -> None:
    pos = _positions(np.linspace(1030.0, 1480.0, 91))
    idx = resolve_source_index(pos, median_nerve_ssep_scenario())
    assert abs(pos[idx, 2] - 1390.0) <= 5.0


def test_tibial_and_median_resolve_to_different_sources() -> None:
    pos = _positions(np.linspace(1030.0, 1480.0, 91))
    med = resolve_source_index(pos, median_nerve_ssep_scenario())
    tib = resolve_source_index(pos, tibial_nerve_ssep_scenario())
    assert med != tib
    assert pos[med, 2] > pos[tib, 2]


def test_scenario_without_generator_uses_most_rostral_source() -> None:
    """Vagal scenarios keep the historical 'highest Z' behaviour."""
    from inob.physiology.scenarios import baroreceptor_scenario

    pos = _positions([1100.0, 1300.0, 1500.0, 1200.0])
    assert resolve_source_index(pos, baroreceptor_scenario()) == 2


def test_generator_outside_the_solved_region_warns(caplog) -> None:
    """A cord solved only cervically cannot host a tibial generator."""
    pos = _positions(np.linspace(1400.0, 1480.0, 10))
    with caplog.at_level("WARNING"):
        resolve_source_index(pos, tibial_nerve_ssep_scenario())
    assert "may not cover this generator" in caplog.text
