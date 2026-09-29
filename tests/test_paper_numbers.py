"""The helpers behind scripts/paper_numbers.py and the two sweep scripts.

The sections themselves read solved leadfields and are exercised by running
the script; what is pinned here is the arithmetic every section shares, so a
trial count the paper quotes means what the text says it means.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eeg_calibration_sweep  # noqa: E402
import paper_numbers  # noqa: E402
import sensor_sweep  # noqa: E402


@pytest.mark.parametrize(
    ("signal", "sigma", "expected"),
    [
        (3.0, 1.0, 1),  # SNR exactly 3: one trial
        (30.0, 1.0, 1),  # far above threshold: still one, never a fraction
        (2.97, 1.0, 2),  # just short: (3 / 2.97)^2 = 1.02 rounds up to 2
        (1.0, 1.0, 9),  # (3 / 1)^2
    ],
)
def test_trials_are_whole_and_reach_snr_three(signal, sigma, expected) -> None:
    assert paper_numbers._n(signal, sigma) == expected


def test_trials_scale_as_inverse_square_of_moment() -> None:
    # The ratio between modalities is independent of Q only because this holds.
    assert paper_numbers._n(0.1, 1.0) == 900
    assert paper_numbers._n(0.2, 1.0) == 225


def test_arc_step_reports_mean_step_and_length() -> None:
    pos = np.array([[0, 0, 0], [0, 0, 5], [0, 0, 15]], dtype=float)
    step, span = paper_numbers._arc_step(pos)
    assert step == pytest.approx(7.5)
    assert span == pytest.approx(15.0)


def test_scripts_expose_their_entry_points() -> None:
    assert set(paper_numbers.SECTIONS) == {"noise", "spine", "vagus", "muscle", "placement"}
    for mod in (paper_numbers, sensor_sweep, eeg_calibration_sweep):
        assert callable(mod.main)
