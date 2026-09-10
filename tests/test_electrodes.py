"""HD-electrode placement tests."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from inob.io.hdf5 import FemMesh, SchemaError, validate_sensors
from inob.sensors.electrodes import ElectrodeArrayParams, build_electrode_array


def _vagus_tube_in_skin() -> tuple[trimesh.Trimesh, FemMesh]:
    skin = trimesh.creation.icosphere(radius=80.0, subdivisions=4)
    # Build a synthetic vagus_left FEM with tets stretching from z=0 to z=80
    n = 30
    nodes_list = []
    tets_list = []
    base = 0
    for k in range(n):
        z = (k / (n - 1)) * 70.0
        nodes_list.append(
            [
                [0, 0, z],
                [1, 0, z],
                [0, 1, z],
                [0, 0, z + 1],
            ]
        )
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
        skin,
        fem,
        ElectrodeArrayParams(
            rows=2, cols=4, contact_pitch_mm=5.0, target_z_low_factor=0.0, target_z_high_factor=1.0
        ),
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
        skin,
        fem,
        ElectrodeArrayParams(
            rows=2, cols=2, contact_pitch_mm=10.0, target_z_low_factor=0.0, target_z_high_factor=1.0
        ),
    )
    # Each contact should be within ~1e-6 of the skin surface (closest-point projection)
    _closest, dists, _ = trimesh.proximity.closest_point(skin, arr.coilpos)
    assert (dists < 1e-3).all(), f"max dist to skin: {dists.max()}"


def test_unknown_target_tissue_raises() -> None:
    skin, fem = _vagus_tube_in_skin()
    with pytest.raises(Exception, match="not in FEM"):
        build_electrode_array(
            skin,
            fem,
            ElectrodeArrayParams(rows=1, cols=2, contact_pitch_mm=5.0, target_tissue="bone"),
        )


# ── target-driven placement ────────────────────────────────────────────────
#
# The OPM array wraps the whole torso and is target-agnostic. The 32-contact
# patch is neither: it is small and directional, so siting it over the vagus
# for a spine solve would make the MEG-vs-EEG comparison measure patch
# placement rather than modality.


def _two_tissue_fem() -> FemMesh:
    """Anterior 'vagus_left' and posterior 'spinal_cord' columns inside a sphere."""
    nodes_list, tets_list, tissue_list = [], [], []
    base = 0
    for tissue_id, y in ((1, -40.0), (2, +40.0)):
        for k in range(30):
            z = (k / 29.0) * 70.0
            nodes_list.append([[0, y, z], [1, y, z], [0, y + 1, z], [0, y, z + 1]])
            tets_list.append([base, base + 1, base + 2, base + 3])
            tissue_list.append(tissue_id)
            base += 4
    return FemMesh(
        np.vstack(nodes_list).astype(np.float64),
        np.array(tets_list, dtype=np.int32),
        np.array(tissue_list, dtype=np.int32),
        tissue_labels=("vagus_left", "spinal_cord"),
        unit="mm",
    )


def test_patch_follows_the_target_tissue() -> None:
    """Retargeting moves the patch to the other side of the body."""
    skin = trimesh.creation.icosphere(radius=80.0, subdivisions=4)
    fem = _two_tissue_fem()

    def centre_for(tissue: str) -> np.ndarray:
        arr = build_electrode_array(
            skin,
            fem,
            ElectrodeArrayParams(
                rows=3,
                cols=3,
                contact_pitch_mm=5.0,
                shape="rectangular",
                target_tissue=tissue,
                target_z_low_factor=0.0,
                target_z_high_factor=1.0,
            ),
        )
        return arr.coilpos.mean(axis=0)

    vagus_c = centre_for("vagus_left")
    cord_c = centre_for("spinal_cord")
    # Opposite sides in the anterior-posterior axis, not a few mm apart.
    assert vagus_c[1] < 0 < cord_c[1]
    assert np.linalg.norm(vagus_c - cord_c) > 50.0


def test_unknown_target_tissue_is_rejected() -> None:
    skin = trimesh.creation.icosphere(radius=80.0, subdivisions=4)
    with pytest.raises(SchemaError, match="not in FEM"):
        build_electrode_array(
            skin,
            _two_tissue_fem(),
            ElectrodeArrayParams(rows=2, cols=2, contact_pitch_mm=5.0, target_tissue="pancreas"),
        )


def test_every_source_target_declares_an_electrode_tissue() -> None:
    """--source-target must set the patch as well as the source, or a spine
    solve silently keeps whatever patch the last run wrote."""
    from inob.config import SOURCE_TARGETS

    for tag, spec in SOURCE_TARGETS.items():
        assert "electrodes" in spec, tag
        assert spec["electrodes"], tag


def test_source_target_couples_patch_tissue_and_output_path() -> None:
    import argparse
    from pathlib import Path

    from inob.cli._common import add_common_args, setup

    p = argparse.ArgumentParser()
    add_common_args(p)
    seen = {}
    for tag in ("vagus", "spine"):
        args = p.parse_args(["--source-target", tag, "--config", "configs/default.yaml"])
        cfg = setup(args, log_prefix="test")
        seen[tag] = (cfg.electrodes.target_tissue, cfg.outputs.electrodes_mat)

    assert seen["vagus"][0] == "vagus_left"
    assert seen["spine"][0] == "spinal_cord"
    # Tagged output paths, so one target's array cannot overwrite another's.
    assert seen["vagus"][1] != seen["spine"][1]
    assert "spine" in Path(seen["spine"][1]).stem
