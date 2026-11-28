"""Real-DUNEuro check on a non-spherical conductor with an exact answer.

A dipole along the axis of an axially symmetric conductor is magnetically
silent outside it (see tests/_meg_cylinder_worker.py for the argument). This
is the geometry of a cord-aligned source in the neck, where the volume
currents cancel most of the primary field. The sphere tests cannot show that
the solver reproduces such a cancellation away from a sphere; this one does.

Set INOB_MEG_CYL_NPZ to reuse an existing result.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("duneuropy", reason="duneuropy not built locally")

from inob.config import SOURCE_MODEL_TYPES

pytestmark = [pytest.mark.duneuro, pytest.mark.slow]

WORKER = Path(__file__).with_name("_meg_cylinder_worker.py")
models = pytest.mark.parametrize("model", SOURCE_MODEL_TYPES)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict[str, np.ndarray]:
    cached = os.environ.get("INOB_MEG_CYL_NPZ")
    out = Path(cached) if cached else tmp_path_factory.mktemp("megcyl") / "cyl.npz"
    if not out.exists():
        subprocess.run([sys.executable, str(WORKER), str(out)], check=False)
    assert out.exists(), "cylinder worker wrote no result"
    return dict(np.load(out, allow_pickle=False))


def _ratio(run, model: str, col: int) -> float:
    p, s = run[f"pri__{model}"][:, col], run[f"sec__{model}"][:, col]
    return float(np.linalg.norm(p - s) / np.linalg.norm(p))


@models
@pytest.mark.parametrize("i_src", [0, 1], ids=["centre", "30mm_up_the_axis"])
def test_axial_dipole_on_the_axis_is_silent(run, model, i_src) -> None:
    """|total| / |primary|: 0 exactly; about 2 with the secondary field added.

    On this 51,000-element mesh the residual is 0.005 (multipolar Venant),
    0.02 (Venant) and 0.11 (partial integration). Partial integration gets a
    looser bound because that residual is its known discretisation error, not
    a sign or unit fault; it is the reason the magnetic solves use Venant.
    """
    limit = 0.20 if model == "partial_integration" else 0.05
    assert _ratio(run, model, 3 * i_src + 2) < limit


@models
def test_axial_dipole_off_the_axis_is_not_silent(run, model) -> None:
    """The cancellation above is the symmetry, not a solver that returns zero."""
    assert _ratio(run, model, 3 * 2 + 2) > 0.15


@models
def test_transverse_dipole_on_the_axis_is_not_silent(run, model) -> None:
    assert _ratio(run, model, 0) > 0.15


@models
def test_volume_currents_reduce_the_field_of_every_source(run, model) -> None:
    """In a compact convex conductor the total field is smaller than the primary."""
    p, s = run[f"pri__{model}"], run[f"sec__{model}"]
    assert np.linalg.norm(p - s) < np.linalg.norm(p)
    assert np.linalg.norm(p + s) > 1.5 * np.linalg.norm(p)  # what the old sign gave
