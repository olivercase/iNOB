"""HD-electrode placement tests."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from vagus_fm.io.hdf5 import FemMesh, validate_sensors
from vagus_fm.sensors.electrodes import ElectrodeArrayParams, build_electrode_array


def _vagus_tube_in_skin() -> tuple[trimesh.Trimesh, FemMesh]:
    skin = trimesh.creation.icosphere(radius=80.0, subdivisions=4)
    # Build a synthetic vagus_left FEM with tets stretching from z=0 to z=80
    n = 30
    nodes_list = []
    tets_list = []
    base = 0
    for k in range(n):
        z = (k / (n - 1)) * 70.0
        nodes_list.append([
            [0, 0, z], [1, 0, z], [0, 1, z], [0, 0, z + 1],
        ])
        tets_list.append([base, base + 1, base + 2, base + 3])
        base += 4
    nodes = np.vstack(nodes_list).astype(np.float64)
    tets = np.array(tets_list, dtype=np.int32)
    tissue = np.ones(len(tets), dtype=np.int32)
    fem = FemMesh(nodes, tets, tissue, tissue_labels=("vagus_left",), unit="mm")
    return skin, fem


def test_electrode_grid_size_and_validation() -> None:
    skin, fem = _vagus_tube_in_skin()
    arr = build_electrode_array(
        skin, fem,
        ElectrodeArrayParams(rows=2, cols=4, contact_pitch_mm=5.0,
                             target_z_low_factor=0.0, target_z_high_factor=1.0),
    )
    validate_sensors(arr)
    assert len(arr.coilpos) == 8
    assert all(t == "eeg" for t in arr.chantype)
    assert all(u == "V" for u in arr.chanunit)
    # Labels should follow elec-RR-CC pattern
    assert arr.labels[0].startswith("elec-00-00")


def test_electrodes_lie_on_skin_surface() -> None:
    skin, fem = _vagus_tube_in_skin()
    arr = build_electrode_array(
        skin, fem,
        ElectrodeArrayParams(rows=2, cols=2, contact_pitch_mm=10.0,
                             target_z_low_factor=0.0, target_z_high_factor=1.0),
    )
    # Each contact should be within ~1e-6 of the skin surface (closest-point projection)
    _closest, dists, _ = trimesh.proximity.closest_point(skin, arr.coilpos)
    assert (dists < 1e-3).all(), f"max dist to skin: {dists.max()}"


def test_unknown_target_tissue_raises() -> None:
    skin, fem = _vagus_tube_in_skin()
    with pytest.raises(Exception, match="not in FEM"):
        build_electrode_array(
            skin, fem,
            ElectrodeArrayParams(rows=1, cols=2, contact_pitch_mm=5.0,
                                 target_tissue="bone"),
        )
