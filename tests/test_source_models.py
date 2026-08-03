"""Like-for-like OPM vs electrode comparison across cord source models."""
from __future__ import annotations

import numpy as np
import pytest

from inob.analysis.source_models import (
    ModelRow,
    compare_source_models,
)
from inob.io.npz import Leadfield
from inob.physiology.profiles import SPINE_PROFILE

N_SRC = 40


def _cord_sources() -> np.ndarray:
    z = np.linspace(1000.0, 1400.0, N_SRC)
    return np.column_stack([np.zeros(N_SRC), np.full(N_SRC, -60.0), z])


def _leadfield(source_pos, *, n_chan: int, scale: float, kind: str,
               seed: int = 0) -> Leadfield:
    """Channels with Gaussian sensitivity bumps at staggered arc positions.

    Tangent (Z) component only, so the longitudinal projection the comparison
    uses picks it up directly.
    """
    rng = np.random.default_rng(seed)
    z = source_pos[:, 2]
    L3 = np.zeros((n_chan, len(source_pos), 3))
    for c, z0 in enumerate(np.linspace(z[0], z[-1], n_chan)):
        L3[c, :, 2] = scale * np.exp(-((z - z0) ** 2) / (2 * 40.0 ** 2))
    L = L3.reshape(n_chan, 3 * len(source_pos))
    prefix = "mag" if kind == "meg" else "elec"
    return Leadfield(
        L=L * 1e-6, L_fT_per_nAm=L, source_pos=source_pos,
        coil_pos=rng.normal(size=(n_chan, 3)),
        coil_orient=np.tile([0.0, 0.0, 1.0], (n_chan, 1)),
        channel_names=tuple(f"{prefix}-{i:03d}" for i in range(n_chan)),
        conductivities=np.array([0.33]), tissue_labels=("skin",), seed=0,
    )


def _compare(*, Q_nAm: float = 5.0, meg_scale: float = 40.0,
             eeg_scale: float = 0.06, source_idx: int = N_SRC // 2):
    src = _cord_sources()
    return compare_source_models(
        _leadfield(src, n_chan=12, scale=meg_scale, kind="meg"),
        _leadfield(src, n_chan=8, scale=eeg_scale, kind="eeg"),
        SPINE_PROFILE, Q_nAm=Q_nAm, source_idx=source_idx,
        meg_sigma_fT=150.0, eeg_sigma_uV=2.2,
    )


# ── the point of the whole module: Q cancels out of every ratio ────────────

def test_modality_gap_is_independent_of_the_assumed_source_strength():
    """The comparison must survive not knowing the true source strength.

    This is why the module exists rather than an absolute-amplitude figure:
    the cord's absolute moment is contested, the modality gap is not.
    """
    weak, strong = _compare(Q_nAm=0.5), _compare(Q_nAm=500.0)
    for a, b in zip(weak.rows, strong.rows, strict=True):
        assert np.isclose(a.modality_gap, b.modality_gap)
    assert np.isclose(weak.coherence_penalty, strong.coherence_penalty)
    assert np.isclose(weak.meg_propagation_factor, strong.meg_propagation_factor)


def test_trial_ratios_are_q_independent_until_the_one_trial_floor():
    """The SNR gap always survives Q; the *trials* gap survives it only while
    both modalities still need averaging.

    Once a modality clears SNR 3 in one trial the count is floored at 1 and
    stops scaling, so a trials ratio quoted for a saturated source model is a
    statement about the floor, not about the modalities.
    """
    weak, strong = _compare(Q_nAm=0.5), _compare(Q_nAm=5.0)
    unsaturated = weak.by_name("ascending")
    assert np.isclose(unsaturated.trials_ratio,
                      strong.by_name("ascending").trials_ratio)
    assert strong.by_name("synchronous").meg_trials == 1.0


def test_absolute_amplitudes_scale_linearly_with_q():
    base, ten_x = _compare(Q_nAm=1.0), _compare(Q_nAm=10.0)
    for a, b in zip(base.rows, ten_x.rows, strict=True):
        assert np.isclose(b.meg_fT, 10.0 * a.meg_fT)
        assert np.isclose(b.eeg_uV, 10.0 * a.eeg_uV)


def test_modality_gap_tracks_the_leadfield_ratio_not_the_source():
    """Doubling the electrode array's sensitivity halves the gap, exactly."""
    base = _compare(eeg_scale=0.06)
    better = _compare(eeg_scale=0.12)
    for a, b in zip(base.rows, better.rows, strict=True):
        assert np.isclose(b.modality_gap, a.modality_gap / 2.0)


# ── the three source models ────────────────────────────────────────────────

def test_synchronous_is_the_largest_and_ascending_the_smallest():
    cmp = _compare()
    sync = cmp.by_name("synchronous")
    stat = cmp.by_name("stationary")
    asc = cmp.by_name("ascending")
    # Assuming the whole cord fires at once is an upper bound on both...
    assert sync.meg_fT > stat.meg_fT > asc.meg_fT
    assert sync.eeg_uV > stat.eeg_uV > asc.eeg_uV
    # ...and letting the same moment travel costs signal, not gains it.
    assert 0.0 < cmp.meg_propagation_factor < 1.0
    assert 0.0 < cmp.eeg_propagation_factor < 1.0
    assert cmp.coherence_penalty < 1.0


def test_synchronous_sums_in_phase_rather_than_redistributing():
    """Each source carries the full Q — the naive assumption, stated plainly.

    With a fixed total moment spread over the cord the synchronous case would
    be no larger than the stationary one, and the bound this model exists to
    provide would silently vanish.
    """
    cmp = _compare(Q_nAm=1.0)
    assert cmp.by_name("synchronous").meg_fT > 5 * cmp.by_name("stationary").meg_fT


def test_ascending_matches_the_stationary_case_times_the_propagation_factor():
    cmp = _compare()
    stat, asc = cmp.by_name("stationary"), cmp.by_name("ascending")
    assert np.isclose(asc.meg_fT, stat.meg_fT * cmp.meg_propagation_factor)
    assert np.isclose(asc.eeg_uV, stat.eeg_uV * cmp.eeg_propagation_factor)


def test_geometry_is_reported_so_the_reader_can_judge_the_lumping():
    cmp = _compare()
    assert cmp.n_sources == N_SRC
    assert cmp.span_mm == pytest.approx(400.0, abs=1.0)
    # The whole reason the stationary model is not defensible for a cord: the
    # wave takes far longer to cross it than one AP lasts.
    assert cmp.transit_ms > 5 * cmp.ap_width_ms


# ── reductions and guards ──────────────────────────────────────────────────

def test_eeg_uses_a_bipolar_pair_and_is_reference_invariant():
    """A common offset on every electrode must not change the EEG amplitude."""
    src = _cord_sources()
    eeg = _leadfield(src, n_chan=8, scale=0.06, kind="eeg")
    shifted = Leadfield(
        L=eeg.L, L_fT_per_nAm=eeg.L_fT_per_nAm + 3.7,
        source_pos=eeg.source_pos, coil_pos=eeg.coil_pos,
        coil_orient=eeg.coil_orient, channel_names=eeg.channel_names,
        conductivities=eeg.conductivities, tissue_labels=eeg.tissue_labels,
        seed=eeg.seed,
    )
    meg = _leadfield(src, n_chan=12, scale=40.0, kind="meg")
    kw = dict(Q_nAm=5.0, source_idx=N_SRC // 2, meg_sigma_fT=150.0,
              eeg_sigma_uV=2.2)
    a = compare_source_models(meg, eeg, SPINE_PROFILE, **kw)
    b = compare_source_models(meg, shifted, SPINE_PROFILE, **kw)
    for ra, rb in zip(a.rows, b.rows, strict=True):
        assert np.isclose(ra.eeg_uV, rb.eeg_uV)


def test_mismatched_source_spaces_are_rejected():
    """Two solves with different source counts are not a like-for-like pair."""
    src = _cord_sources()
    meg = _leadfield(src, n_chan=12, scale=40.0, kind="meg")
    eeg = _leadfield(src[:-1], n_chan=8, scale=0.06, kind="eeg")
    with pytest.raises(ValueError, match="different source spaces"):
        compare_source_models(meg, eeg, SPINE_PROFILE, Q_nAm=5.0, source_idx=3,
                              meg_sigma_fT=150.0, eeg_sigma_uV=2.2)


def test_unordered_sources_are_rejected_for_the_ascending_model():
    rng = np.random.default_rng(0)
    cloud = rng.normal(scale=50.0, size=(N_SRC, 3)) + np.array([0.0, -60.0, 1200.0])
    meg = _leadfield(cloud, n_chan=12, scale=40.0, kind="meg")
    eeg = _leadfield(cloud, n_chan=8, scale=0.06, kind="eeg")
    with pytest.raises(ValueError, match="not an ordered path"):
        compare_source_models(meg, eeg, SPINE_PROFILE, Q_nAm=5.0, source_idx=3,
                              meg_sigma_fT=150.0, eeg_sigma_uV=2.2)


# ── trial arithmetic ───────────────────────────────────────────────────────

def test_trials_ratio_is_the_square_of_the_snr_gap():
    row = ModelRow("t", "", meg_fT=30.0, eeg_uV=0.01,
                   meg_sigma_fT=150.0, eeg_sigma_uV=2.2)
    assert np.isclose(row.trials_ratio, row.modality_gap ** 2)


def test_a_single_trial_is_the_floor_on_trial_count():
    """You cannot average a fraction of a trial."""
    row = ModelRow("t", "", meg_fT=10_000.0, eeg_uV=100.0,
                   meg_sigma_fT=150.0, eeg_sigma_uV=2.2)
    assert row.meg_trials == 1.0
    assert row.eeg_trials == 1.0
