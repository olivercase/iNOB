"""Watertight geometry build pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from inob.config import Config, ShrinkwrapParams, load_config
from inob.geometry.builder import (
    _find_vagus_side,
    _trimesh_to_compartment,
    _validate_compartment,
    build_geometry,
    check_existing,
    gather_inputs,
    watertighten,
)
from inob.io.hdf5 import SchemaError, load_geometry

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return load_config(TINY_CFG, project_root=tmp_path)


def _write_stl(path: Path, mesh: trimesh.Trimesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


@pytest.fixture
def populated_data_dirs(cfg: Config) -> Config:
    """Write minimal STL inputs for every compartment gather_inputs expects."""
    box = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    sphere = trimesh.creation.icosphere(radius=10.0)

    _write_stl(cfg.data.torso_skin, box)
    _write_stl(cfg.data.bone_dir / "bone1.stl", box)
    _write_stl(cfg.data.muscle_dir / "m1.stl", box)
    _write_stl(cfg.data.vessel_dir / "v1.stl", box)
    (cfg.project_root / "data" / "vagus").mkdir(parents=True, exist_ok=True)
    _write_stl(cfg.project_root / "data" / "vagus" / "left_a.stl", sphere)
    _write_stl(cfg.project_root / "data" / "vagus" / "right_a.stl", sphere)
    return cfg


# ── _find_vagus_side ─────────────────────────────────────────────────────

def test_find_vagus_side_matches_word_boundary() -> None:
    # "_" counts as a word char, so "left" only matches with a non-word
    # separator (here "-") or at a string boundary either side.
    paths = [Path("vagus-left-a.stl"), Path("vagus-right-a.stl"), Path("leftover.stl")]
    out = _find_vagus_side(paths, "left")
    assert out == [Path("vagus-left-a.stl")]


def test_find_vagus_side_case_insensitive() -> None:
    paths = [Path("Vagus-LEFT.stl")]
    assert _find_vagus_side(paths, "left") == paths


def test_find_vagus_side_no_match_returns_empty() -> None:
    assert _find_vagus_side([Path("skin.stl")], "left") == []


def test_find_vagus_side_underscore_separated_does_not_match() -> None:
    # Underscore is a word character, so \bleft\b does NOT match "vagus_left".
    assert _find_vagus_side([Path("vagus_left.stl")], "left") == []


def test_find_vagus_side_sorted() -> None:
    paths = [Path("b-left.stl"), Path("a-left.stl")]
    assert _find_vagus_side(paths, "left") == [Path("a-left.stl"), Path("b-left.stl")]


# ── gather_inputs ────────────────────────────────────────────────────────

def test_gather_inputs_missing_bone_raises(cfg: Config) -> None:
    with pytest.raises(SchemaError, match="bone"):
        gather_inputs(cfg)


def test_gather_inputs_missing_torso_skin_raises(cfg: Config) -> None:
    (cfg.data.bone_dir).mkdir(parents=True, exist_ok=True)
    _write_stl(cfg.data.bone_dir / "b.stl", trimesh.creation.box())
    with pytest.raises(SchemaError, match="torso skin"):
        gather_inputs(cfg)


def test_gather_inputs_resolves_all_compartments(populated_data_dirs: Config) -> None:
    out = gather_inputs(populated_data_dirs)
    assert set(out) == {
        "mesh_skin", "mesh_bone", "mesh_muscle", "mesh_spinal_cord",
        "mesh_blood_vessel", "mesh_vagus_left", "mesh_vagus_right",
    }
    assert out["mesh_skin"] == [populated_data_dirs.data.torso_skin]
    assert len(out["mesh_bone"]) == 1
    assert len(out["mesh_vagus_left"]) == 1
    assert len(out["mesh_vagus_right"]) == 1


def test_gather_inputs_prefers_cleaned_bone_dir(populated_data_dirs: Config) -> None:
    clean_dir = populated_data_dirs.outputs.intermediate_bone_clean
    _write_stl(clean_dir / "cleaned.stl", trimesh.creation.box())
    out = gather_inputs(populated_data_dirs)
    assert out["mesh_bone"] == [clean_dir / "cleaned.stl"]


# ── watertighten ─────────────────────────────────────────────────────────

def test_watertighten_returns_watertight_mesh_for_clean_sphere() -> None:
    m = trimesh.creation.icosphere(radius=10.0)
    params = ShrinkwrapParams(
        pitch=1.0, n_samples=2000, close_iter=1, decimate_target=500, smooth_iter=1,
    )
    out = watertighten(m, "mesh_vagus_left", params)
    assert isinstance(out, trimesh.Trimesh)
    assert out.is_watertight


def test_watertighten_repairs_duplicate_vertices() -> None:
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    v_dup = np.vstack([box.vertices, box.vertices])
    f_dup = np.vstack([box.faces, box.faces + len(box.vertices)])
    raw = trimesh.Trimesh(v_dup, f_dup, process=False)
    params = ShrinkwrapParams(
        pitch=1.0, n_samples=2000, close_iter=1, decimate_target=500, smooth_iter=1,
    )
    out = watertighten(raw, "mesh_skin", params)
    assert out.is_watertight


# ── _validate_compartment ────────────────────────────────────────────────

def test_validate_compartment_passes_for_watertight_mesh(cfg: Config) -> None:
    m = trimesh.creation.icosphere(radius=5.0)
    _validate_compartment(m, "mesh_skin", cfg)  # should not raise


def test_validate_compartment_raises_when_not_watertight(cfg: Config) -> None:
    m = trimesh.creation.icosphere(radius=5.0)
    m.update_faces(np.arange(1, len(m.faces)))
    with pytest.raises(SchemaError, match="not watertight"):
        _validate_compartment(m, "mesh_skin", cfg)


def test_validate_compartment_euler_downgraded_to_warning_when_watertight(
    tmp_path: Path,
) -> None:
    cfg = load_config(TINY_CFG, project_root=tmp_path)
    # A torus (genus 1) is watertight and winding-consistent but euler != 2.
    torus = trimesh.creation.torus(major_radius=10.0, minor_radius=3.0)
    if not (torus.is_watertight and torus.is_winding_consistent):
        pytest.skip("trimesh torus primitive not watertight on this version")
    _validate_compartment(torus, "mesh_skin", cfg)  # warns, does not raise


# ── _trimesh_to_compartment ──────────────────────────────────────────────

def test_trimesh_to_compartment_round_trips_arrays() -> None:
    m = trimesh.creation.box(extents=(5.0, 5.0, 5.0))
    comp = _trimesh_to_compartment("mesh_skin", m)
    assert comp.name == "mesh_skin"
    np.testing.assert_array_equal(comp.vertices, np.asarray(m.vertices, dtype=np.float64))
    np.testing.assert_array_equal(comp.faces, np.asarray(m.faces, dtype=np.int64))
    assert comp.vertices.dtype == np.float64
    assert comp.faces.dtype == np.int64


# ── build_geometry (end-to-end, tiny meshes) ─────────────────────────────

def test_build_geometry_end_to_end(populated_data_dirs: Config) -> None:
    out_path = build_geometry(populated_data_dirs, only_compartments=("mesh_skin", "mesh_vagus_left"))
    assert out_path == populated_data_dirs.outputs.geometry_mat
    assert out_path.is_file()
    geom = load_geometry(out_path)
    assert set(geom.compartments) == {"mesh_skin", "mesh_vagus_left"}


def test_build_geometry_raises_when_only_filters_out_everything(
    populated_data_dirs: Config,
) -> None:
    with pytest.raises(SchemaError, match="no compartments built"):
        build_geometry(populated_data_dirs, only_compartments=("mesh_does_not_exist",))


def test_build_geometry_raises_when_shrinkwrap_params_missing(
    populated_data_dirs: Config,
) -> None:
    # tiny_test.yaml only defines shrinkwrap params for mesh_skin / mesh_vagus_left.
    with pytest.raises(SchemaError, match="shrinkwrap params"):
        build_geometry(populated_data_dirs, only_compartments=("mesh_bone",))


# ── check_existing ───────────────────────────────────────────────────────

def test_check_existing_raises_if_output_missing(cfg: Config) -> None:
    with pytest.raises(FileNotFoundError):
        check_existing(cfg)


def test_check_existing_true_for_valid_build(populated_data_dirs: Config) -> None:
    build_geometry(populated_data_dirs, only_compartments=("mesh_skin", "mesh_vagus_left"))
    assert check_existing(populated_data_dirs) is True
