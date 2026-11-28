"""Propagating vs stationary source models, and the correction they imply."""

from __future__ import annotations

import numpy as np
import pytest

from inob.analysis.propagation import (
    compute_propagation_signals,
    peak_bipolar,
    peak_over_channels,
    propagation_ratio,
)
from inob.io.npz import Leadfield
from inob.physiology.profiles import SPINE_PROFILE, VAGUS_PROFILE


def _straight_line_leadfield(*, n_src: int = 40, n_chan: int = 12, seed: int = 0):
    """A polyline along +Z with a smooth, single-peaked sensitivity profile.

    Channels peak at different arc positions so the array as a whole sees the
    wave pass, which is what makes the propagating/stationary contrast
    meaningful rather than an artefact of one degenerate column.
    """
    rng = np.random.default_rng(seed)
    z = np.linspace(1000.0, 1400.0, n_src)
    source_pos = np.column_stack([np.zeros(n_src), np.zeros(n_src), z])
    # Each channel is a Gaussian bump in arc length, tangent (Z) component only.
    centres = np.linspace(z[0], z[-1], n_chan)
    L3 = np.zeros((n_chan, n_src, 3))
    for c, z0 in enumerate(centres):
        L3[c, :, 2] = np.exp(-((z - z0) ** 2) / (2 * 40.0**2))
    L = L3.reshape(n_chan, 3 * n_src)
    return Leadfield(
        L=L,
        L_fT_per_nAm=L * 1e6,
        source_pos=source_pos,
        coil_pos=rng.normal(size=(n_chan, 3)),
        coil_orient=np.tile([0.0, 0.0, 1.0], (n_chan, 1)),
        channel_names=tuple(f"mag-{i:03d}" for i in range(n_chan)),
        conductivities=np.array([0.4]),
        tissue_labels=("skin",),
        seed=0,
    )


def test_same_total_moment_under_both_models() -> None:
    """The comparison is only meaningful if both models carry the same charge.

    Integrating the moment map over time must match between the stationary lump
    and the travelling wavelet, or an amplitude ratio would be measuring a
    moment difference rather than propagation.
    """
    lf = _straight_line_leadfield()
    sig = compute_propagation_signals(lf, SPINE_PROFILE)
    # Both signals are the same leadfield applied to the same total moment, so
    # their time-integrated absolute response must be of the same order.
    stat = np.abs(sig.stationary).sum()
    prop = np.abs(sig.segment).sum()
    assert 0.1 < prop / stat < 10.0


def test_propagation_reduces_peak_for_a_sweeping_volley() -> None:
    """A wave crossing far more than its AP width must not beat the lump.

    This is the spine's whole problem: the stationary approximation
    overestimates, so the ratio belongs below 1.
    """
    lf = _straight_line_leadfield()
    sig = compute_propagation_signals(lf, SPINE_PROFILE)
    assert not SPINE_PROFILE.stationary_ok
    assert 0.0 < propagation_ratio(sig, peak_over_channels) < 1.0


def test_ratio_reference_position_matters() -> None:
    """The stationary lump position changes the ratio, so it must be explicit.

    Regression test for a real bug: the detectability figure measured its
    correction against a lump at the polyline's rostral end while its own
    signal came from the source under the array. The mismatch produced a ratio
    of ~8 — propagation apparently *amplifying* the signal — which is an
    artefact of comparing two different dipole positions.
    """
    lf = _straight_line_leadfield()
    at_rostral = compute_propagation_signals(lf, SPINE_PROFILE)
    at_middle = compute_propagation_signals(lf, SPINE_PROFILE, stationary_idx=5)

    assert at_rostral.stationary_idx == at_rostral.hot_idx
    assert at_middle.stationary_idx == 5
    # The propagating model is untouched by where the lump is put ...
    assert np.allclose(at_rostral.segment, at_middle.segment)
    # ... so any difference in the ratio is entirely the reference change.
    assert not np.isclose(
        propagation_ratio(at_rostral, peak_over_channels),
        propagation_ratio(at_middle, peak_over_channels),
    )


def test_bipolar_reduction_is_reference_invariant() -> None:
    """The EEG reduction must survive re-referencing, like the observable it
    corrects (:func:`inob.analysis.snr.per_source_best_bipolar`)."""
    lf = _straight_line_leadfield()
    sig = compute_propagation_signals(lf, SPINE_PROFILE)
    shifted = sig.segment + 3.7  # a common-mode offset on every channel
    assert np.isclose(peak_bipolar(sig.segment), peak_bipolar(shifted))


def _single_velocity(profile):
    """``profile`` with one fibre class, so the wave does not disperse."""
    from dataclasses import replace

    from inob.sources.cap import FibreDistribution

    return replace(
        profile,
        fibres=FibreDistribution(diameters_um=np.array([10.0]), weights=np.array([1.0])),
    )


def _uniform_leadfield(profile_fn, *, n_src: int = 400):
    """One channel over a straight 400 mm path with a prescribed sensitivity."""
    z = np.linspace(1000.0, 1400.0, n_src)
    L3 = np.zeros((1, n_src, 3))
    L3[0, :, 2] = profile_fn(z)
    return Leadfield(
        L=L3.reshape(1, 3 * n_src),
        L_fT_per_nAm=L3.reshape(1, 3 * n_src) * 1e6,
        source_pos=np.column_stack([np.zeros(n_src), np.zeros(n_src), z]),
        coil_pos=np.zeros((1, 3)),
        coil_orient=np.array([[0.0, 0.0, 1.0]]),
        channel_names=("mag-000",),
        conductivities=np.array([0.4]),
        tissue_labels=("skin",),
        seed=0,
    )


def test_travelling_wave_is_silent_under_a_uniform_sensitivity() -> None:
    """The two lobes are opposed and equal, so a sensor that weights every
    position equally sees nothing while the whole wave is on the path."""
    profile = _single_velocity(SPINE_PROFILE)
    lf = _uniform_leadfield(np.ones_like)
    sig = compute_propagation_signals(lf, profile)
    # Mid-transit: half the conduction time before the wave reaches the end.
    transit_ms = sig.total_span_mm / sig.cv_mean_m_per_s
    i_mid = int(np.argmin(np.abs(sig.t_ms - (0.4 * sig.duration_ms - 0.5 * transit_ms))))
    assert abs(sig.whole[0, i_mid]) < 1e-3 * np.abs(sig.stationary).max()


def test_each_lobe_carries_the_stationary_moment() -> None:
    """With a step in sensitivity the leading lobe alone is seen as it crosses
    the step, and its integrated moment equals the lumped moment Q."""
    profile = _single_velocity(SPINE_PROFILE)
    lf = _uniform_leadfield(lambda z: (z >= 1200.0).astype(float))
    sig = compute_propagation_signals(lf, profile)
    # The stationary peak is L * Q with L = 1 at the rostral end.
    assert np.abs(sig.whole).max() == pytest.approx(np.abs(sig.stationary).max(), rel=0.02)


def test_dispersion_lowers_the_peak() -> None:
    """A spread of conduction velocities smears the volley as it climbs."""
    lf = _uniform_leadfield(lambda z: (z >= 1200.0).astype(float))
    one = compute_propagation_signals(lf, _single_velocity(SPINE_PROFILE))
    many = compute_propagation_signals(lf, SPINE_PROFILE)
    assert np.abs(many.whole).max() < np.abs(one.whole).max()

def test_segment_is_whole_flag() -> None:
    """A profile with no localised generator spans the whole polyline."""
    lf = _straight_line_leadfield()
    assert SPINE_PROFILE.propagation_span_mm is None
    assert compute_propagation_signals(lf, SPINE_PROFILE).segment_is_whole
    assert not compute_propagation_signals(lf, VAGUS_PROFILE).segment_is_whole


def test_propagation_ratio_rejects_unknown_model() -> None:
    lf = _straight_line_leadfield()
    sig = compute_propagation_signals(lf, SPINE_PROFILE)
    with pytest.raises(ValueError, match="'segment' or 'whole'"):
        propagation_ratio(sig, model="bogus")


def test_ordered_polyline_detection() -> None:
    """Guards the propagating model against unordered (volume-fill) sources.

    Regression test: the combined spine+muscle EEG leadfield produced a
    propagation factor of 0.006 — a 170x attenuation — purely because
    cumulative arc length over an unordered point cloud is meaningless.
    """
    from inob.analysis.propagation import is_ordered_polyline

    rng = np.random.default_rng(7)
    # A path: consecutive points are neighbours.
    line = np.column_stack([np.zeros(60), np.zeros(60), np.linspace(0, 400, 60)])
    assert is_ordered_polyline(line)

    # A volume fill over a comparable extent, visited in random order.
    cloud = rng.uniform(-150, 150, size=(600, 3))
    assert not is_ordered_polyline(cloud)

    # Degenerate inputs must not raise.
    assert is_ordered_polyline(line[:2])
    assert not is_ordered_polyline(np.zeros((5, 3)))
