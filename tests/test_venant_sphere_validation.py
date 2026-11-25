"""Real-DUNEuro check that the St. Venant source model works and is accurate.

The Venant keys have no defaults on the DUNEuro side (``params.get<T>("…")``
with no fallback), so a missing or misspelled one is a C++ throw, not a warning
— exactly the kind of failure that would otherwise surface hours into a cluster
array job. This runs the same homogeneous-sphere validation the partial-
integration path is held to, with the source model swapped, and asserts Venant
clears the same accuracy bar against Sarvas.

Auto-skips when duneuropy is not importable (same convention as the other
real-DUNEuro tests).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("duneuropy", reason="duneuropy not built locally")

from inob.analysis.sphere_calibration import validate_meg_sphere
from inob.config import load_config
from inob.forward.duneuro_driver import build_source_model_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"


@pytest.mark.duneuro
def test_venant_source_model_matches_sarvas_on_sphere(tmp_path) -> None:
    cfg = load_config(DEFAULT_CFG, overrides=["forward.source_model.type=venant"])
    v = validate_meg_sphere(out_dir=tmp_path, source_model=build_source_model_config(cfg))
    # Same bar as partial integration in test_meg_sphere_validation.py: if
    # Venant cannot clear it on a homogeneous sphere, it is not a candidate for
    # the production solve.
    assert v.rdm_radial < 0.02, f"RDM {v.rdm_radial:.4f} — Venant MEG topography wrong"
    assert abs(v.mag_radial - 1.0) < 0.02, f"MAG {v.mag_radial:.4f} — Venant magnitude mis-scaled"
    # Tangential coils: the only ones that see the volume-current field.
    # Measured on this mesh: MAG 0.980, RDM 0.091 (wrong sign: about 5 and 1.4).
    assert abs(v.mag_tangential - 1.0) < 0.04, f"tangential MAG {v.mag_tangential:.4f}"
    assert v.rdm_tangential < 0.12, f"tangential RDM {v.rdm_tangential:.4f}"
