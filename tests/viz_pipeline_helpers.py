"""Shared, self-contained pipeline builder for viz smoke tests.

Not a test module itself (no ``test_`` prefix) — pytest will not collect it.
Builds a tiny but complete on-disk pipeline (geometry, MEG/EEG sensors,
MEG/EEG leadfields) under a tmp_path project root, using ``configs/tiny_test.yaml``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from inob.config import Config, load_config
from inob.io.hdf5 import (
    CompartmentMesh,
    Geometry,
    SensorArray,
    save_geometry,
    save_sensors,
)
from inob.io.npz import Leadfield, save_leadfield

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"

N_SOURCES = 8
SOURCE_Z = np.linspace(-150.0, 150.0, N_SOURCES)


def _build_meg_sensors(n_rings: int = 6) -> SensorArray:
    """Triaxial ring array around the cervical Z range, same layout convention
    as ``tests/conftest.py::tiny_sensors`` (R/T1/T2 per position)."""
    zs = np.linspace(-200.0, 200.0, n_rings)
    pos = np.array([[60.0, 0.0, z] for z in zs] + [[-60.0, 0.0, z] for z in zs])
    n = len(pos)
    R = np.tile([1.0, 0.0, 0.0], (n, 1))
    T1 = np.tile([0.0, 1.0, 0.0], (n, 1))
    T2 = np.tile([0.0, 0.0, 1.0], (n, 1))
    coilpos = np.repeat(pos, 3, axis=0)
    coilori = np.empty((3 * n, 3))
    for i in range(n):
        coilori[3 * i + 0] = R[i]
        coilori[3 * i + 1] = T1[i]
        coilori[3 * i + 2] = T2[i]
    labels = tuple(
        f"mag-{i // 3:04d}-{['R', 'T1', 'T2'][i % 3]}" for i in range(3 * n)
    )
    return SensorArray(
        coilpos=coilpos, coilori=coilori, labels=labels,
        chantype=tuple(["megmag"] * (3 * n)), chanunit=tuple(["T"] * (3 * n)),
        unit="mm",
    )


def _build_electrodes(rows: int = 2, cols: int = 4, pitch: float = 5.0) -> SensorArray:
    xs = (np.arange(cols) - (cols - 1) / 2.0) * pitch
    zs = (np.arange(rows) - (rows - 1) / 2.0) * pitch
    pos = np.array([[x, 30.0, z] for z in zs for x in xs])
    n = len(pos)
    labels = tuple(f"elec-{r:02d}-{c:02d}" for r in range(rows) for c in range(cols))
    return SensorArray(
        coilpos=pos, coilori=np.tile([0.0, -1.0, 0.0], (n, 1)),
        labels=labels, chantype=tuple(["eeg"] * n), chanunit=tuple(["V"] * n),
        unit="mm",
    )


def _build_leadfield(sensors: SensorArray, *, seed: int, scale: float) -> Leadfield:
    n_c = sensors.coilpos.shape[0]
    rng = np.random.default_rng(seed)
    L_fT = rng.standard_normal((n_c, 3 * N_SOURCES)) * scale
    source_pos = np.column_stack(
        [np.zeros(N_SOURCES), np.zeros(N_SOURCES), SOURCE_Z]
    )
    return Leadfield(
        L=L_fT * 1e-6,
        L_fT_per_nAm=L_fT,
        source_pos=source_pos,
        coil_pos=sensors.coilpos,
        coil_orient=sensors.coilori,
        channel_names=sensors.labels,
        conductivities=np.array([0.3, 0.43]),
        tissue_labels=("vagus_left", "skin"),
        seed=seed,
    )


def build_pipeline_cfg(tmp_path: Path) -> Config:
    """Write a full tiny synthetic pipeline to ``tmp_path`` and return its Config."""
    cfg = load_config(TINY_CFG, project_root=tmp_path)

    # An elongated ellipsoid (not a box) so vertices are spread continuously
    # along Z — surface_topoplot's Z-band cropping needs vertices at every Z.
    skin = trimesh.creation.icosphere(radius=1.0, subdivisions=4)
    skin.vertices = skin.vertices * np.array([100.0, 100.0, 300.0])
    vagus = trimesh.creation.icosphere(radius=15.0, subdivisions=2)
    geom = Geometry(compartments={
        "mesh_skin": CompartmentMesh(
            name="mesh_skin",
            vertices=np.asarray(skin.vertices, dtype=np.float64),
            faces=np.asarray(skin.faces, dtype=np.int64),
        ),
        "mesh_vagus_left": CompartmentMesh(
            name="mesh_vagus_left",
            vertices=np.asarray(vagus.vertices, dtype=np.float64),
            faces=np.asarray(vagus.faces, dtype=np.int64),
        ),
    })
    save_geometry(cfg.outputs.geometry_mat, geom)

    meg_sensors = _build_meg_sensors()
    electrodes = _build_electrodes()
    save_sensors(cfg.outputs.sensors_mat, meg_sensors)
    save_sensors(cfg.outputs.electrodes_mat, electrodes)

    meg_lf = _build_leadfield(meg_sensors, seed=1, scale=50.0)
    eeg_lf = _build_leadfield(electrodes, seed=2, scale=5.0)
    save_leadfield(cfg.outputs.forward_npz, meg_lf)
    save_leadfield(cfg.outputs.forward_eeg_npz, eeg_lf)

    return cfg
