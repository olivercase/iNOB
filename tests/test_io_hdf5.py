"""HDF5 I/O tests: schema, atomic writes, validation."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inob.io.hdf5 import (
    CompartmentMesh,
    FemMesh,
    Geometry,
    SchemaError,
    SensorArray,
    atomic_write_hdf5,
    load_fem,
    load_geometry,
    load_sensors,
    save_geometry,
    validate_fem,
    validate_geometry,
    validate_sensors,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── geometry ───────────────────────────────────────────────────────────────

def test_geometry_roundtrip(tmp_path: Path, tiny_geometry: Geometry) -> None:
    p = tmp_path / "geom.mat"
    save_geometry(p, tiny_geometry)
    loaded = load_geometry(p)
    assert set(loaded.compartments) == set(tiny_geometry.compartments)
    for name in tiny_geometry.compartments:
        a, b = tiny_geometry.compartments[name], loaded.compartments[name]
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(a.faces, b.faces)


def test_geometry_validate_passes(tiny_geometry: Geometry) -> None:
    validate_geometry(tiny_geometry)


def test_geometry_validate_rejects_bad_face_index(tiny_geometry: Geometry) -> None:
    bad = dict(tiny_geometry.compartments)
    skin = bad["mesh_skin"]
    bad["mesh_skin"] = replace(skin, faces=skin.faces.copy())
    bad["mesh_skin"].faces[0, 0] = len(skin.vertices) + 5
    with pytest.raises(SchemaError, match="out of range"):
        validate_geometry(Geometry(compartments=bad))


def test_geometry_validate_rejects_nonfinite() -> None:
    verts = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    g = Geometry(compartments={
        "mesh_skin": CompartmentMesh("mesh_skin", verts, faces),
    })
    with pytest.raises(SchemaError, match="non-finite"):
        validate_geometry(g)


def test_geometry_unit_guard_rejects_metres() -> None:
    # 0.1 m cube — extent 0.1, below 1 mm threshold
    verts = np.array([
        [0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1],
        [0.1, 0.1, 0], [0.1, 0, 0.1], [0, 0.1, 0.1], [0.1, 0.1, 0.1],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    g = Geometry(compartments={"mesh_skin": CompartmentMesh("mesh_skin", verts, faces)})
    with pytest.raises(SchemaError, match="implausible for mm"):
        validate_geometry(g)


# ── FEM ────────────────────────────────────────────────────────────────────

def test_fem_roundtrip(tiny_fem_path: Path, tiny_fem: FemMesh) -> None:
    loaded = load_fem(tiny_fem_path)
    np.testing.assert_array_equal(loaded.nodes, tiny_fem.nodes)
    np.testing.assert_array_equal(loaded.tets, tiny_fem.tets)
    np.testing.assert_array_equal(loaded.tissue, tiny_fem.tissue)
    assert loaded.tissue_labels == tiny_fem.tissue_labels
    assert loaded.unit == "mm"


def test_fem_validate_passes(tiny_fem: FemMesh) -> None:
    validate_fem(tiny_fem)


def test_fem_validate_rejects_oob_tet() -> None:
    nodes = np.zeros((4, 3))
    nodes[1, 0] = nodes[2, 1] = nodes[3, 2] = 10.0
    tets = np.array([[0, 1, 2, 99]], dtype=np.int32)
    mesh = FemMesh(nodes=nodes, tets=tets, tissue=np.array([1], dtype=np.int32),
                   tissue_labels=("a",))
    with pytest.raises(SchemaError, match="out of range"):
        validate_fem(mesh)


def test_fem_validate_rejects_noncontiguous_tissues() -> None:
    nodes = np.eye(4, 3) * 10
    nodes[0] = 0
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = FemMesh(nodes=nodes, tets=tets, tissue=np.array([3], dtype=np.int32),
                   tissue_labels=("a", "b", "c"))
    with pytest.raises(SchemaError, match="not contiguous"):
        validate_fem(mesh)


def test_fem_load_legacy_artifact() -> None:
    """A locally-built fem_vagus.mat (in outputs/fem/) must load + validate.

    Asserts structural invariants only — NOT a hardcoded tissue list. The set
    of tissues is config-driven (adding spinal_cord, or any structure, changes
    it), so pinning an exact tuple here would make an ordinary config change
    fail an unrelated I/O test.
    """
    artifact = REPO_ROOT / "outputs" / "fem" / "fem_vagus.mat"
    if not artifact.exists():
        pytest.skip("FEM artifact not present")
    mesh = load_fem(artifact)
    validate_fem(mesh)
    assert mesh.nodes.shape[1] == 3
    assert mesh.tets.shape[1] == 4
    assert len(mesh.tissue_labels) >= 1
    assert all(isinstance(t, str) and t for t in mesh.tissue_labels)
    # Every tissue id used by a tet must have a corresponding label.
    assert int(mesh.tissue.max()) <= len(mesh.tissue_labels)


def test_fem_load_legacy_geometry() -> None:
    artifact = REPO_ROOT / "outputs" / "geometry" / "vagus_geometry.mat"
    if not artifact.exists():
        pytest.skip("legacy geometry artifact not present")
    geom = load_geometry(artifact)
    validate_geometry(geom)
    assert {"mesh_skin", "mesh_bone", "mesh_vagus_left", "mesh_vagus_right"} <= set(
        geom.compartments
    )


# ── sensors ────────────────────────────────────────────────────────────────

def test_sensors_roundtrip(tiny_sensors_path: Path, tiny_sensors: SensorArray) -> None:
    loaded = load_sensors(tiny_sensors_path)
    np.testing.assert_array_equal(loaded.coilpos, tiny_sensors.coilpos)
    np.testing.assert_allclose(loaded.coilori, tiny_sensors.coilori, atol=1e-12)
    assert loaded.labels == tiny_sensors.labels
    assert loaded.chantype == tiny_sensors.chantype
    assert loaded.unit == "mm"


def test_sensors_validate_passes(tiny_sensors: SensorArray) -> None:
    validate_sensors(tiny_sensors)


def test_sensors_validate_rejects_label_count_mismatch(tiny_sensors: SensorArray) -> None:
    bad = SensorArray(
        coilpos=tiny_sensors.coilpos,
        coilori=tiny_sensors.coilori,
        labels=tiny_sensors.labels[:-1],
        chantype=tiny_sensors.chantype,
        chanunit=tiny_sensors.chanunit,
        unit="mm",
    )
    with pytest.raises(SchemaError, match="labels length"):
        validate_sensors(bad)


def test_sensors_load_legacy_artifact() -> None:
    artifact = REPO_ROOT / "outputs" / "sensors" / "sensor_array.mat"
    if not artifact.exists():
        pytest.skip("legacy sensors artifact not present")
    s = load_sensors(artifact)
    validate_sensors(s)
    assert s.coilpos.shape[1] == 3
    assert len(s.labels) == s.coilpos.shape[0]


# ── atomic write ───────────────────────────────────────────────────────────

def test_atomic_write_leaves_no_partial_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "broken.h5"
    sentinel = "preexisting"
    target.write_text(sentinel)
    with pytest.raises(RuntimeError):
        with atomic_write_hdf5(target) as h:
            h.create_dataset("x", data=np.arange(3))
            raise RuntimeError("boom")
    # original content untouched
    assert target.read_text() == sentinel
    siblings = [p for p in tmp_path.iterdir() if p.name.startswith(f".{target.name}.")]
    assert siblings == [], f"orphan tempfile(s) left behind: {siblings}"


def test_atomic_write_replaces_atomically(tmp_path: Path) -> None:
    target = tmp_path / "ok.h5"
    target.write_text("old content")
    with atomic_write_hdf5(target) as h:
        h.create_dataset("v", data=np.arange(5))
    # File is now real HDF5
    import h5py
    with h5py.File(str(target), "r") as f:
        np.testing.assert_array_equal(f["v"][...], np.arange(5))
