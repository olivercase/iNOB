"""Real-DUNEuro proof that the MEG volume-current field is subtracted.

The older sphere check (tests/test_meg_sphere_validation.py) used radial coils
only. Outside a sphere the volume currents make no radial field, so that check
passes whichever sign the secondary field is given. These tests use tangential
coils, radial dipoles and closed loops, all of which fail with the wrong sign.

One worker process does every solve (tests/_meg_sign_worker.py) and the tests
read its npz. Set INOB_MEG_SIGN_NPZ to reuse an existing result.

The test mesh is coarse (about 33,000 tetrahedra) so the suite stays runnable.
The limits below were set from the values measured on it and are loose on
topography for that reason; what they separate is the sign. Measured with the
correct sign / with the wrong sign, worst source model:

    best-fit coefficient on the secondary    -0.976        (same fit)
    magnitude against Sarvas, tangential      0.94 - 1.01  / 5.3
    RDM against Sarvas, tangential            0.45         / 1.39
    radial dipole, |total| / |primary|        0.15         / 1.98
    circulation, |total| / |primary|          0.20         / 2.0
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("duneuropy", reason="duneuropy not built locally")

from inob.analysis.analytic_sphere import infinite_medium_meg_field, sarvas_meg_field
from inob.config import SOURCE_MODEL_TYPES

pytestmark = [pytest.mark.duneuro, pytest.mark.slow]

WORKER = Path(__file__).with_name("_meg_sign_worker.py")
DEPTHS = (30, 50, 70, 85)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict[str, np.ndarray]:
    cached = os.environ.get("INOB_MEG_SIGN_NPZ")
    out = Path(cached) if cached else tmp_path_factory.mktemp("megsign") / "sign.npz"
    if not out.exists():
        subprocess.run([sys.executable, str(WORKER), str(out)], check=False)
    assert out.exists(), "MEG sign worker wrote no result"
    d = dict(np.load(out, allow_pickle=False))
    assert "L_layered" in d, "MEG sign worker did not finish"
    return d


def _proj(B: np.ndarray, ori: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", B, ori)


def _analytic(run, fn) -> np.ndarray:
    """(n_coils, 3 * n_src) analytic lead field, T per A.m, same column order."""
    cols = []
    for p in run["src"]:
        for e in np.eye(3):
            cols.append(_proj(fn(p * 1e-3, e, run["coilpos"] * 1e-3), run["coilori"]))
    return np.column_stack(cols)


def _moments(run, which: str) -> np.ndarray:
    """(3 * n_src, n_src [* 2]) weights turning x/y/z columns into radial or tangential dipoles."""
    src = run["src"]
    n = len(src)
    cols = []
    for i, p in enumerate(src):
        r = p / np.linalg.norm(p)
        t1 = np.cross(r, [1.0, 0.0, 0.0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(r, t1)
        for m in [r] if which == "radial" else [t1, t2]:
            w = np.zeros(3 * n)
            w[3 * i : 3 * i + 3] = m
            cols.append(w)
    return np.column_stack(cols)


def _mag_rdm(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(na / nb), float(np.linalg.norm(a / na - b / nb))


TRI = slice(0, 36)  # the triaxial sites
models = pytest.mark.parametrize("model", SOURCE_MODEL_TYPES)


# ── what DUNEuro returns ───────────────────────────────────────────────────


@models
def test_primary_is_free_space_biot_savart(run, model) -> None:
    bs = _analytic(run, infinite_medium_meg_field)
    mag, rdm = _mag_rdm(run[f"pri__{model}"], bs)
    assert abs(mag - 1) < 1e-6 and rdm < 1e-6, (mag, rdm)


@models
def test_secondary_is_not_small_on_tangential_coils(run, model) -> None:
    """Otherwise every test below would be vacuous."""
    tan = run["kind"] == 1
    ratio = np.linalg.norm(run[f"sec__{model}"][tan]) / np.linalg.norm(run[f"pri__{model}"][tan])
    assert ratio > 0.3, ratio


@models
def test_secondary_vanishes_on_radial_coils(run, model) -> None:
    """Why the radial-only sphere check could not see the sign."""
    rad = run["kind"] == 0
    ratio = np.linalg.norm(run[f"sec__{model}"][rad]) / np.linalg.norm(run[f"pri__{model}"][rad])
    assert ratio < 0.05, ratio


@models
def test_best_fit_secondary_coefficient_is_minus_one(run, model) -> None:
    """Fit c in Sarvas = primary + c * secondary, with no assumption about the sign."""
    W = _moments(run, "tangential")
    resid = ((_analytic(run, sarvas_meg_field) - run[f"pri__{model}"]) @ W)[TRI].ravel()
    sec = (run[f"sec__{model}"] @ W)[TRI].ravel()
    c = float(sec @ resid / (sec @ sec))
    assert -1.05 < c < -0.95, c


# ── the production lead field ──────────────────────────────────────────────


@models
@pytest.mark.parametrize("coils", ["all", "tangential", "radial"])
def test_leadfield_matches_sarvas(run, model, coils) -> None:
    W = _moments(run, "tangential")
    sel = {"all": run["kind"] < 2, "tangential": run["kind"] == 1, "radial": run["kind"] == 0}[
        coils
    ]
    fem = (run[f"L__{model}"] @ W)[sel]
    ref = (_analytic(run, sarvas_meg_field) @ W)[sel]
    mag, rdm = _mag_rdm(fem, ref)
    mag_tol, rdm_max = (0.01, 0.01) if coils == "radial" else (0.10, 0.60)
    assert abs(mag - 1) < mag_tol, f"MAG {mag:.3f}"
    assert rdm < rdm_max, f"RDM {rdm:.3f}"


@models
@pytest.mark.parametrize("i_src", range(len(DEPTHS)), ids=[f"r{d}mm" for d in DEPTHS])
def test_leadfield_matches_sarvas_at_each_depth(run, model, i_src) -> None:
    W = _moments(run, "tangential")[:, 2 * i_src : 2 * i_src + 2]
    fem = (run[f"L__{model}"] @ W)[TRI]
    ref = (_analytic(run, sarvas_meg_field) @ W)[TRI]
    mag, rdm = _mag_rdm(fem, ref)
    assert abs(mag - 1) < 0.15, f"MAG {mag:.3f}"
    assert rdm < 0.55, f"RDM {rdm:.3f}"


@models
def test_radial_dipole_is_silent(run, model) -> None:
    """A radial dipole in a sphere makes no field outside, on any component.

    With the wrong sign the lead field is about twice the primary field.
    """
    W = _moments(run, "radial")
    ratio = np.linalg.norm((run[f"L__{model}"] @ W)[TRI]) / np.linalg.norm(
        (run[f"pri__{model}"] @ W)[TRI]
    )
    assert ratio < 0.25, ratio


@models
@pytest.mark.parametrize("loop", [2, 3], ids=["small_loop", "great_circle"])
def test_ampere_circulation_is_zero(run, model, loop) -> None:
    """No current threads a loop outside the conductor, so the circulation of B is zero.

    Needs no analytic sphere solution. The primary field alone has a large
    circulation, which the volume-current field has to cancel.
    """
    sel = run["kind"] == loop
    circ_L = np.abs(run[f"L__{model}"][sel].sum(axis=0))
    circ_p = np.abs(run[f"pri__{model}"][sel].sum(axis=0))
    assert circ_p.max() > 0, "primary circulation is zero: test would be vacuous"
    ratio = float(np.linalg.norm(circ_L) / np.linalg.norm(circ_p))
    assert ratio < 0.30, ratio


def test_source_models_agree(run) -> None:
    ref = run["L__partial_integration"][TRI]
    for model in SOURCE_MODEL_TYPES[1:]:
        mag, rdm = _mag_rdm(run[f"L__{model}"][TRI], ref)
        assert abs(mag - 1) < 0.08 and rdm < 0.40, (model, mag, rdm)


# ── conductivity ───────────────────────────────────────────────────────────


def test_field_outside_does_not_depend_on_homogeneous_conductivity(run) -> None:
    a = run["L__partial_integration"][run["sub"]]
    mag, rdm = _mag_rdm(run["L_sigma1"], a)
    assert abs(mag - 1) < 1e-3 and rdm < 1e-3, (mag, rdm)


def test_layered_sphere_matches_sarvas(run) -> None:
    """Sarvas holds for any radially symmetric conductor, so shells change nothing."""
    W = _moments(run, "tangential")
    fem = run["L_layered"] @ W
    ref = (_analytic(run, sarvas_meg_field) @ W)[run["sub"]]
    mag, rdm = _mag_rdm(fem, ref)
    assert abs(mag - 1) < 0.10, f"MAG {mag:.3f}"
    assert rdm < 0.45, f"RDM {rdm:.3f}"


def test_layered_sphere_radial_dipole_is_silent(run) -> None:
    W = _moments(run, "radial")
    ratio = np.linalg.norm(run["L_layered"] @ W) / np.linalg.norm(
        (run["pri__partial_integration"] @ W)[run["sub"]]
    )
    assert ratio < 0.30, ratio
