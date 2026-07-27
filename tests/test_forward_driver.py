"""Tests for the conductivity-vector builder (no DUNEuro required)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inob.config import load_config
from inob.forward.duneuro_driver import (
    build_conductivity_vector,
    build_driver_config,
)
from inob.io.hdf5 import FemMesh

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"


def test_conductivity_vector_full_4_tissue() -> None:
    cfg = load_config(DEFAULT_CFG)
    nodes = np.eye(4, 3) * 50.0
    nodes[0] = 0
    tets = np.array([[0, 1, 2, 3]] * 4, dtype=np.int32)
    tissue = np.array([1, 2, 3, 4], dtype=np.int32)
    fem = FemMesh(nodes, tets, tissue,
                  ("vagus_left", "vagus_right", "bone", "skin"), "mm")
    cond = build_conductivity_vector(cfg, fem)
    expected = np.array([0.30, 0.30, 0.0042, 0.43]) * 1e-3
    np.testing.assert_allclose(cond, expected)


def test_conductivity_missing_label_raises(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CFG)
    nodes = np.eye(4, 3) * 50.0
    nodes[0] = 0
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    tissue = np.array([1], dtype=np.int32)
    # a tissue label with no entry in forward.conductivities_sm must fail loudly
    fem = FemMesh(nodes, tets, tissue, ("ghost_tissue",), "mm")
    with pytest.raises(KeyError, match="ghost_tissue"):
        build_conductivity_vector(cfg, fem)


def test_driver_config_shape_and_solver_settings() -> None:
    cfg = load_config(DEFAULT_CFG)
    nodes = np.eye(4, 3) * 50.0
    nodes[0] = 0
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    tissue = np.array([1], dtype=np.int32)
    fem = FemMesh(nodes, tets, tissue, ("vagus_left",), "mm")
    cond = np.array([3e-4])
    d = build_driver_config(cfg, fem, cond)
    assert d["type"] == "fitted"
    assert d["solver_type"] == "cg"
    assert d["solver"]["scheme"] == "sipg"
    assert d["solver"]["reduction"] == "1e-10"
    assert d["meg"]["intorderadd"] == "5"
    np.testing.assert_array_equal(d["volume_conductor"]["grid"]["elements"], tets)
    np.testing.assert_array_equal(d["volume_conductor"]["tensors"]["labels"],
                                   tissue.astype(np.int64) - 1)


# ── fibre-aligned muscle anisotropy ────────────────────────────────────────

def _muscle_fem() -> FemMesh:
    """4 tets: two muscle (near each test STL), one bone, one skin."""
    nodes = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ])
    # All tets share the same 4 nodes; only the tissue tag matters here, and
    # every muscle centroid lands near the origin -> nearest-STL assignment is
    # driven by the STL centres we write in the fixture below.
    tets = np.array([[0, 1, 2, 3]] * 4, dtype=np.int32)
    tissue = np.array([1, 1, 2, 3], dtype=np.int32)
    return FemMesh(nodes, tets, tissue, ("muscle", "bone", "skin"), "mm")


def _write_box_stl(path: Path, extents, centre) -> None:
    import trimesh
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(centre)
    m.export(path)


@pytest.fixture
def muscle_stls(tmp_path: Path) -> Path:
    """Two muscles: one elongated along Z, one along X."""
    d = tmp_path / "muscle"
    d.mkdir()
    _write_box_stl(d / "a_long_z.stl", (4.0, 4.0, 80.0), (0.0, 0.0, 0.0))
    _write_box_stl(d / "b_long_x.stl", (80.0, 4.0, 4.0), (500.0, 0.0, 0.0))
    return d


def test_anisotropy_disabled_returns_none(muscle_stls: Path) -> None:
    """Default config keeps the isotropic scalar path (vagus/spine unchanged)."""
    from inob.forward.duneuro_driver import build_conductivity_tensors
    cfg = load_config(DEFAULT_CFG, overrides=[f"data.muscle_dir={muscle_stls}"])
    # mode="auto" with a non-muscle source_tissue -> isotropic scalar path.
    assert cfg.forward.muscle_anisotropy.mode == "auto"
    assert cfg.forward.muscle_anisotropy.active_for(cfg.forward.source_tissue) is False
    assert build_conductivity_tensors(cfg, _muscle_fem()) is None


def test_anisotropy_tensor_eigenvalues_and_orientation(muscle_stls: Path) -> None:
    from inob.forward.duneuro_driver import build_conductivity_tensors
    cfg = load_config(DEFAULT_CFG, overrides=[
        "forward.muscle_anisotropy.mode=on",
        f"data.muscle_dir={muscle_stls}",
    ])
    fem = _muscle_fem()
    labels, tensors = build_conductivity_tensors(cfg, fem)

    a = cfg.forward.muscle_anisotropy
    scale = cfg.forward.sigma_unit_scale
    n_base = int(np.unique(fem.tissue).max())
    assert len(tensors) == n_base + 2          # 3 base tissues + 2 muscle STLs

    # Every label must be a valid index into the tensor table (duneuro requires it).
    assert labels.min() >= 0 and labels.max() < len(tensors)
    # Muscle tets are relabelled into the appended block; others keep tissue-1.
    mask = fem.tissue == fem.label_to_id["muscle"]
    assert (labels[mask] >= n_base).all()
    np.testing.assert_array_equal(labels[~mask], fem.tissue[~mask] - 1)

    # Each muscle tensor has eigenvalues (sigma_trans, sigma_trans, sigma_long).
    expected_ev = np.array([a.sigma_trans_sm, a.sigma_trans_sm, a.sigma_long_sm]) * scale
    for T in tensors[n_base:]:
        np.testing.assert_allclose(T, T.T, atol=1e-18)          # symmetric
        np.testing.assert_allclose(np.linalg.eigvalsh(T), expected_ev, rtol=1e-9)

    # The high-conductivity axis follows each muscle's long axis (sign-free).
    for tensor, axis in ((tensors[n_base], [0, 0, 1]), (tensors[n_base + 1], [1, 0, 0])):
        ev, evec = np.linalg.eigh(tensor)
        dominant = evec[:, np.argmax(ev)]
        assert abs(float(np.dot(dominant, axis))) > 0.99


def test_driver_config_switches_to_tensor_mode(muscle_stls: Path) -> None:
    """Anisotropy uses duneuro's `tensors` key; isotropic keeps `conductivities`."""
    from inob.forward.duneuro_driver import build_conductivity_tensors
    cfg = load_config(DEFAULT_CFG, overrides=[
        "forward.muscle_anisotropy.mode=on",
        f"data.muscle_dir={muscle_stls}",
    ])
    fem = _muscle_fem()
    cond = build_conductivity_vector(cfg, fem)
    aniso = build_conductivity_tensors(cfg, fem)

    vc = build_driver_config(cfg, fem, cond, aniso_tensors=aniso)["volume_conductor"]
    assert set(vc["tensors"]) == {"labels", "tensors"}
    assert all(t.shape == (3, 3) for t in vc["tensors"]["tensors"])

    vc_iso = build_driver_config(cfg, fem, cond)["volume_conductor"]
    assert set(vc_iso["tensors"]) == {"labels", "conductivities"}
