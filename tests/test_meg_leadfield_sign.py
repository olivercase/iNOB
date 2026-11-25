"""Fast tests (no DUNEuro) pinning how the MEG primary and secondary fields combine.

DUNEuro's ``applyMEGTransfer`` returns the volume-current field with the
opposite sign to the one that adds to the primary field: the total field is
``primary - secondary``. Brainstorm (``bst_duneuro.m``: ``Bp - Bs``) and
FieldTrip (``leadfield_duneuro.m``: ``Bp - lf``) both subtract. The real-solver
proof is in tests/test_meg_secondary_sign_real_duneuro.py; these tests keep the
convention from regressing in the fast suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from inob.forward.duneuro_driver import MEG_MM_MODE_TO_SI, compute_meg_leadfield

SRC = Path(__file__).resolve().parents[1] / "src" / "inob"


class _Driver:
    def __init__(self, pri: np.ndarray, sec: np.ndarray):
        self.pri, self.sec = pri, sec
        self.calls: list[str] = []

    def applyMEGTransfer(self, T, dipoles, cfg):
        self.calls.append("secondary")
        return [self.sec[:, j] for j in range(self.sec.shape[1])], None

    def computeMEGPrimaryField(self, dipoles, cfg):
        self.calls.append("primary")
        return [self.pri[:, j] for j in range(self.pri.shape[1])]


def _fields(seed: int = 0, n_coils: int = 7, n_dip: int = 6):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_coils, n_dip)), rng.normal(size=(n_coils, n_dip))


def test_total_is_primary_minus_secondary() -> None:
    pri, sec = _fields()
    L = compute_meg_leadfield(_Driver(pri, sec), None, [None] * 6, {})
    np.testing.assert_allclose(L, (pri - sec) * MEG_MM_MODE_TO_SI, rtol=0, atol=1e-15)


def test_total_is_not_primary_plus_secondary() -> None:
    pri, sec = _fields(1)
    L = compute_meg_leadfield(_Driver(pri, sec), None, [None] * 6, {})
    assert not np.allclose(L, (pri + sec) * MEG_MM_MODE_TO_SI)


def test_zero_secondary_gives_primary() -> None:
    pri, _ = _fields(2)
    L = compute_meg_leadfield(_Driver(pri, np.zeros_like(pri)), None, [None] * 6, {})
    np.testing.assert_allclose(L, pri * MEG_MM_MODE_TO_SI)


def test_secondary_equal_to_primary_cancels() -> None:
    """A radial source in a sphere: the volume currents cancel the primary field."""
    pri, _ = _fields(3)
    L = compute_meg_leadfield(_Driver(pri, pri.copy()), None, [None] * 6, {})
    np.testing.assert_allclose(L, 0.0, atol=1e-15)


def test_zero_primary_gives_minus_secondary() -> None:
    _, sec = _fields(4)
    L = compute_meg_leadfield(_Driver(np.zeros_like(sec), sec), None, [None] * 6, {})
    np.testing.assert_allclose(L, -sec * MEG_MM_MODE_TO_SI)


def test_shape_dtype_and_both_fields_requested() -> None:
    pri, sec = _fields(5, n_coils=4, n_dip=9)
    drv = _Driver(pri, sec)
    L = compute_meg_leadfield(drv, None, [None] * 9, {})
    assert L.shape == (4, 9)
    assert L.dtype == np.float64
    assert sorted(drv.calls) == ["primary", "secondary"]


def test_linear_in_both_fields() -> None:
    pri, sec = _fields(6)
    a = compute_meg_leadfield(_Driver(pri, sec), None, [None] * 6, {})
    b = compute_meg_leadfield(_Driver(2 * pri, 2 * sec), None, [None] * 6, {})
    np.testing.assert_allclose(b, 2 * a)


def test_components_function_matches_total() -> None:
    from inob.forward.duneuro_driver import compute_meg_field_components

    pri, sec = _fields(7)
    p, s = compute_meg_field_components(_Driver(pri, sec), None, [None] * 6, {})
    np.testing.assert_allclose(p, pri * MEG_MM_MODE_TO_SI)
    np.testing.assert_allclose(s, sec * MEG_MM_MODE_TO_SI)
    L = compute_meg_leadfield(_Driver(pri, sec), None, [None] * 6, {})
    np.testing.assert_allclose(L, p - s)


def test_mm_mode_constant_unchanged() -> None:
    assert MEG_MM_MODE_TO_SI == pytest.approx(0.1)


def test_fields_are_combined_in_one_place_only() -> None:
    """No module but duneuro_driver may call the raw DUNEuro MEG apply functions.

    A second, hand-written combination is how a wrong sign survives a fix.
    """
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name == "duneuro_driver.py":
            continue
        text = path.read_text()
        if re.search(r"\.(applyMEGTransfer|computeMEGPrimaryField)\(", text):
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"raw MEG apply calls outside duneuro_driver: {offenders}"


def test_no_source_text_says_secondary_plus_primary() -> None:
    bad = []
    for path in SRC.rglob("*.py"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(
                r"secondary\s*\+\s*primary|primary\s*\+\s*secondary|sec\s*\+\s*pri\b", line
            ):
                bad.append(f"{path.relative_to(SRC)}:{i}")
    assert not bad, f"'+' combination of MEG fields at: {bad}"
