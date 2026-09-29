"""The brain cross-check that answers "is EG too small, or is its noise too big?"

The arithmetic is pinned on synthetic leadfields so that a number the script
prints for the real brain means what the column header says.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from brain_sanity import compare  # noqa: E402


def _one_source_leadfields(mg_peak: float, eg_hi: float, eg_lo: float):
    # Two channels, one source, three moments; only the z moment couples.
    L_mg = np.array([[0.0, 0.0, mg_peak], [0.0, 0.0, -0.5 * mg_peak]])
    L_eg = np.array([[0.0, 0.0, eg_hi], [0.0, 0.0, eg_lo]])
    return L_mg, L_eg


def test_signals_are_best_channel_for_mg_and_contact_difference_for_eg() -> None:
    L_mg, L_eg = _one_source_leadfields(100.0, 0.3, -0.1)
    pos = np.array([[0.0, 0.0, 0.0]])
    coils = np.array([[0.0, 0.0, 35.0]])
    (row,) = compare(L_mg, pos, L_eg, pos, coils, standoff_mm=10.0, sigma_fT=200.0, sigma_uV=2.0)
    assert row.mg_fT_per_nAm == pytest.approx(100.0)
    assert row.eg_uV_per_nAm == pytest.approx(0.4)
    # SNR_EG / SNR_MG = (0.4 / 2) / (100 / 200)
    assert row.snr_ratio_eg_over_mg == pytest.approx(0.4)
    # 35 mm to the coil less the 10 mm stand-off puts it in the 20-30 mm bin.
    assert (row.lo_mm, row.hi_mm) == (20.0, 30.0)


def test_sources_are_paired_by_position_and_unmatched_ones_dropped() -> None:
    L_mg = np.zeros((1, 6))
    L_mg[0, 2], L_mg[0, 5] = 50.0, 80.0
    L_eg = np.zeros((2, 6))
    # EG sources listed in the opposite order to MG's.
    L_eg[:, 2] = [0.0, 0.2]  # source at the second MG position
    L_eg[:, 5] = [0.0, 0.1]  # source at the first MG position
    pos_mg = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]])
    pos_eg = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 0.0]])
    coils = np.array([[0.0, 0.0, 50.0]])  # both sources 40 mm below skin
    (row,) = compare(
        L_mg, pos_mg, L_eg, pos_eg, coils, standoff_mm=10.0, sigma_fT=1.0, sigma_uV=1.0
    )
    assert row.n == 2
    assert row.eg_uV_per_nAm == pytest.approx(0.15)  # median of 0.1 and 0.2

    far = pos_eg + 5.0
    assert (
        compare(L_mg, pos_mg, L_eg, far, coils, standoff_mm=10.0, sigma_fT=1.0, sigma_uV=1.0) == []
    )
