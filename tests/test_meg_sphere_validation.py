"""Real-DUNEuro MEG magnitude validation against the Sarvas analytic sphere.

This is the check that was missing when the MEG forward shipped: EEG is
calibrated to Berg–Scherg, but nothing validated the MEG magnitude. Its absence
let two bugs through — the primary Biot–Savart field being omitted, and a 10x
mm-mode scale error. On a homogeneous sphere the FEM MEG field must match Sarvas
to within a few percent (RDM) at unit magnitude ratio (MAG).

Auto-skips when duneuropy is not importable (same convention as the other
real-DUNEuro tests).
"""
from __future__ import annotations

import pytest

pytest.importorskip("duneuropy", reason="duneuropy not built locally")

from inob.analysis.sphere_calibration import validate_meg_sphere
from inob.forward.duneuro_driver import MEG_MM_MODE_TO_SI


@pytest.mark.duneuro
def test_meg_forward_matches_sarvas_on_sphere(tmp_path) -> None:
    v = validate_meg_sphere(out_dir=tmp_path)
    # Topography: FEM must reproduce the analytic field shape.
    assert v.rdm < 0.02, f"RDM {v.rdm:.4f} too high — MEG topography wrong"
    # Magnitude: absolute calibration must be correct (this is what the missing
    # primary field and the 10x scale error each broke).
    assert abs(v.mag - 1.0) < 0.02, f"MAG {v.mag:.4f} — MEG magnitude mis-scaled"
    # Sanity on the constant itself.
    assert MEG_MM_MODE_TO_SI == pytest.approx(0.1)
    assert v.fem_peak_fT_per_nAm == pytest.approx(v.sarvas_peak_fT_per_nAm, rel=0.02)
