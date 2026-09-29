"""Re-referencing an electrode leadfield, and fixing the dipole orientation.

Reviewer questions these answer: would a far reference beat the common
average (Mellor), and how does a dipole pointing out of the back compare with
the best-of-three-axes source (Mellor)? Both are linear on the stored
leadfield, so they are tested against the closed forms here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inob.analysis.reference import (
    EegReference,
    parse_reference,
    reference_electrodes,
    rereference,
)
from inob.analysis.snr import (
    compute_noise_floors,
    parse_orientation,
    per_source_amplitude,
    per_source_best_bipolar,
    per_source_peak,
    per_source_triaxial_peak,
)
from inob.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def _array(seed: int = 0, C: int = 10, S: int = 4):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((C, 3 * S))
    elec = rng.uniform(-100, 100, size=(C, 3))
    src = rng.uniform(-20, 20, size=(S, 3))
    car = raw - raw.mean(axis=0, keepdims=True)
    return raw, car, elec, src


# ── parsing ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, EegReference("average")),
        ("average", EegReference("average")),
        ("farthest", EegReference("farthest")),
        ("electrode:3", EegReference("electrode", index=3)),
        ("7", EegReference("electrode", index=7)),
        ("nearest:0,-100,1600", EegReference("nearest", point_mm=(0.0, -100.0, 1600.0))),
    ],
)
def test_parse_reference(spec, expected) -> None:
    ref = parse_reference(spec)
    assert ref == expected
    assert parse_reference(ref.label) == ref  # the label round-trips


@pytest.mark.parametrize("bad", ["nose", "electrode:", "nearest:1,2", "electrode:-1", "farthest:3"])
def test_parse_reference_rejects(bad) -> None:
    with pytest.raises(ValueError, match="reference"):
        parse_reference(bad)


# ── re-referencing ─────────────────────────────────────────────────────────


def test_average_of_average_is_identity() -> None:
    _, car, elec, src = _array()
    assert np.allclose(rereference(car, "average", electrode_pos=elec, source_pos=src), car)


def test_electrode_reference_recovers_raw_difference() -> None:
    """The stored CAR loses only a per-column constant, so any single-contact
    reference is recovered exactly: CAR - CAR[r] == raw - raw[r]."""
    raw, car, elec, src = _array(1)
    got = rereference(car, "electrode:4", electrode_pos=elec, source_pos=src)
    assert np.allclose(got, raw - raw[4][None, :])
    assert np.allclose(got[4], 0.0)  # the reference channel reads zero


def test_farthest_reference_is_per_source() -> None:
    raw, car, elec, src = _array(2)
    rows = reference_electrodes(parse_reference("farthest"), elec, src)
    d = np.linalg.norm(elec[:, None, :] - src[None, :, :], axis=2)
    assert rows.tolist() == np.argmax(d, axis=0).tolist()
    got = rereference(car, "farthest", electrode_pos=elec, source_pos=src)
    for s, r in enumerate(rows):
        cols = slice(3 * s, 3 * s + 3)
        assert np.allclose(got[:, cols], raw[:, cols] - raw[r, cols][None, :])


def test_nearest_reference_picks_closest_contact() -> None:
    _, _car, elec, src = _array(3)
    target = elec[6] + 1.0
    rows = reference_electrodes(parse_reference("nearest:{},{},{}".format(*target)), elec, src)
    assert set(rows.tolist()) == {6}


def test_rereference_is_linear() -> None:
    _, car, elec, src = _array(4)
    other = _array(5)[1]
    for spec in ("average", "electrode:2", "farthest", "nearest:0,0,0"):
        lhs = rereference(2.0 * car + other, spec, electrode_pos=elec, source_pos=src)
        rhs = 2.0 * rereference(car, spec, electrode_pos=elec, source_pos=src) + rereference(
            other, spec, electrode_pos=elec, source_pos=src
        )
        assert np.allclose(lhs, rhs)


def test_bipolar_is_unchanged_and_bounds_every_reference() -> None:
    """Bipolar amplitudes are reference-free, and no in-array reference can
    push the single-channel peak above them."""
    _, car, elec, src = _array(6)
    bip = per_source_best_bipolar(car)
    for spec in ("average", "electrode:0", "electrode:9", "farthest", "nearest:50,50,50"):
        L = rereference(car, spec, electrode_pos=elec, source_pos=src)
        assert np.allclose(per_source_best_bipolar(L), bip)
        assert (per_source_peak(L) <= bip + 1e-12).all()


def test_rereference_rejects_mismatched_positions() -> None:
    _, car, elec, src = _array(7)
    with pytest.raises(ValueError, match="electrode positions"):
        rereference(car, "farthest", electrode_pos=elec[:-1], source_pos=src)
    with pytest.raises(ValueError, match="out of range"):
        rereference(car, "electrode:99", electrode_pos=elec, source_pos=src)


# ── orientation ────────────────────────────────────────────────────────────


def test_parse_orientation() -> None:
    assert parse_orientation(None) is None
    assert parse_orientation("best") is None
    # Anterior is -Y (docs/COORDINATES.md), so "out of the back" is +Y.
    assert np.allclose(parse_orientation("posterior"), [0, 1, 0])
    assert np.allclose(parse_orientation("z"), [0, 0, 1])
    assert np.allclose(parse_orientation("3,0,4"), [0.6, 0.0, 0.8])
    for bad in ("up", "1,2", "0,0,0", "a,b,c"):
        with pytest.raises(ValueError):
            parse_orientation(bad)


def test_fixed_orientation_matches_projection() -> None:
    raw, *_ = _array(8)
    C, S = raw.shape[0], raw.shape[1] // 3
    q = parse_orientation("1,2,2")
    proj = raw.reshape(C, S, 3) @ q  # (C, S)
    assert np.allclose(per_source_peak(raw, orientation=q), np.abs(proj).max(axis=0))
    assert np.allclose(
        per_source_best_bipolar(raw, orientation=q), proj.max(axis=0) - proj.min(axis=0)
    )
    assert np.allclose(per_source_amplitude(raw, orientation=q), np.sqrt(np.mean(proj**2, axis=0)))


def test_axis_orientation_never_exceeds_best() -> None:
    """'best' is the max over the three axes, so each axis is bounded by it,
    and the max over the three axes reproduces it exactly."""
    raw, *_ = _array(9)
    best = per_source_peak(raw)
    axes = [per_source_peak(raw, orientation=parse_orientation(a)) for a in "xyz"]
    assert np.allclose(np.max(axes, axis=0), best)
    best_bip = per_source_best_bipolar(raw)
    axes_bip = [per_source_best_bipolar(raw, orientation=parse_orientation(a)) for a in "xyz"]
    assert np.allclose(np.max(axes_bip, axis=0), best_bip)


def test_orientation_sign_does_not_matter() -> None:
    raw, *_ = _array(10)
    post = parse_orientation("posterior")
    ant = parse_orientation("anterior")
    assert np.allclose(
        per_source_peak(raw, orientation=post), per_source_peak(raw, orientation=ant)
    )


def test_triaxial_peak_with_orientation() -> None:
    rng = np.random.default_rng(11)
    L = rng.standard_normal((3 * 5, 3 * 2))
    q = parse_orientation("y")
    assert np.allclose(
        per_source_triaxial_peak(L, orientation=q),
        per_source_triaxial_peak(L[:, 1::3].repeat(3, axis=1)),
    )


# ── contact temperature ────────────────────────────────────────────────────


def test_contact_temperature_default_reproduces_core_temperature() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert cfg.noise.eeg_contact_temperature_K == pytest.approx(310.15)


def test_contact_temperature_moves_only_the_johnson_term() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    cool = replace(cfg, noise=replace(cfg.noise, eeg_contact_temperature_K=305.0))
    hot = compute_noise_floors(cfg).eeg_per_channel_uV
    cold = compute_noise_floors(cool).eeg_per_channel_uV
    # Johnson at 10 kOhm is ~0.013 uV/sqrt(Hz) against a 0.1 amplifier, so a
    # 5 K drop changes the total floor by ~0.014%, not the ~1.7% it moves the
    # Johnson power.
    assert cold < hot
    assert (hot - cold) / hot == pytest.approx(1.4e-4, rel=0.05)
    # The MEG floor has no temperature in it.
    assert (
        compute_noise_floors(cool).meg_per_channel_fT
        == compute_noise_floors(cfg).meg_per_channel_fT
    )


def test_contact_temperature_must_be_positive() -> None:
    with pytest.raises(ConfigError, match="eeg_contact_temperature_K"):
        load_config(
            REPO_ROOT / "configs" / "default.yaml",
            overrides=["noise.eeg_contact_temperature_K=0"],
        )
