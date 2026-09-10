"""Shared pytest fixtures: tiny synthetic meshes, FEM, sensor arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from inob.io.hdf5 import (
    CompartmentMesh,
    FemMesh,
    Geometry,
    SensorArray,
    save_fem,
    save_geometry,
    save_sensors,
)


@pytest.fixture
def cube_trimesh() -> trimesh.Trimesh:
    """A 100 mm cube — enough extent to pass the unit-mm guard."""
    return trimesh.creation.box(extents=(100.0, 100.0, 100.0))


@pytest.fixture
def sphere_trimesh() -> trimesh.Trimesh:
    """A 50 mm-radius icosphere."""
    return trimesh.creation.icosphere(radius=50.0, subdivisions=3)


@pytest.fixture
def tiny_geometry(cube_trimesh: trimesh.Trimesh, sphere_trimesh: trimesh.Trimesh) -> Geometry:
    """Two-compartment geometry: skin (cube) and vagus_left (sphere inside)."""
    return Geometry(
        compartments={
            "mesh_skin": CompartmentMesh(
                name="mesh_skin",
                vertices=np.asarray(cube_trimesh.vertices, dtype=np.float64),
                faces=np.asarray(cube_trimesh.faces, dtype=np.int64),
            ),
            "mesh_vagus_left": CompartmentMesh(
                name="mesh_vagus_left",
                vertices=np.asarray(sphere_trimesh.vertices, dtype=np.float64),
                faces=np.asarray(sphere_trimesh.faces, dtype=np.int64),
            ),
        }
    )


@pytest.fixture
def tiny_fem() -> FemMesh:
    """A 5-node, 2-tet FEM mesh with both tissue ids represented (1 and 2)."""
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [10.0, 10.0, 10.0],
        ],
        dtype=np.float64,
    )
    tets = np.array(
        [
            [0, 1, 2, 3],  # tissue 1 (vagus_left)
            [1, 2, 3, 4],  # tissue 2 (skin)
        ],
        dtype=np.int32,
    )
    tissue = np.array([1, 2], dtype=np.int32)
    return FemMesh(
        nodes=nodes,
        tets=tets,
        tissue=tissue,
        tissue_labels=("vagus_left", "skin"),
        unit="mm",
    )


@pytest.fixture
def tiny_sensors() -> SensorArray:
    """4 positions × 3 axes (R, T1, T2) = 12 channels, all unit-norm orientations."""
    pos = np.array(
        [
            [60.0, 0.0, 0.0],
            [-60.0, 0.0, 0.0],
            [0.0, 60.0, 0.0],
            [0.0, -60.0, 0.0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3)[[0, 0, 1, 1]] * np.array([[1], [-1], [1], [-1]])
    T1 = np.eye(3)[[1, 1, 0, 0]]
    T2 = np.eye(3)[[2, 2, 2, 2]]
    coilpos = np.repeat(pos, 3, axis=0)
    coilori = np.empty((12, 3))
    for i in range(4):
        coilori[3 * i + 0] = R[i]
        coilori[3 * i + 1] = T1[i]
        coilori[3 * i + 2] = T2[i]
    coilori /= np.linalg.norm(coilori, axis=1, keepdims=True)
    labels = tuple(f"mag-{i // 3:04d}-{['R', 'T1', 'T2'][i % 3]}" for i in range(12))
    return SensorArray(
        coilpos=coilpos,
        coilori=coilori,
        labels=labels,
        chantype=tuple(["megmag"] * 12),
        chanunit=tuple(["T"] * 12),
        unit="mm",
    )


@pytest.fixture
def tiny_fem_path(tmp_path: Path, tiny_fem: FemMesh) -> Path:
    p = tmp_path / "fem_tiny.mat"
    save_fem(p, tiny_fem)
    return p


@pytest.fixture
def tiny_geometry_path(tmp_path: Path, tiny_geometry: Geometry) -> Path:
    p = tmp_path / "geom_tiny.mat"
    save_geometry(p, tiny_geometry)
    return p


@pytest.fixture
def tiny_sensors_path(tmp_path: Path, tiny_sensors: SensorArray) -> Path:
    p = tmp_path / "sensors_tiny.mat"
    save_sensors(p, tiny_sensors)
    return p
