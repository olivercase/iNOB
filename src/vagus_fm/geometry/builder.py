"""Build a watertight, manifold geometry from per-tissue STLs.

Replaces the legacy ``build_geom.py`` script with a config-driven, logging,
typed-IO version. Per-compartment shrinkwrap parameters come from
``cfg.geometry.shrinkwrap``; outputs land at ``cfg.outputs.geometry_mat``.

Pipeline (per compartment):

    1. Cheap repair (merge verts, drop degenerates, fix normals).
    2. Boolean union of overlapping components if multi-component.
    3. Voxel shrinkwrap (configurable per-tissue parameters).
    4. pymeshfix as a final hole-closer.

Each compartment is validated (watertight ∧ winding-consistent ∧ Euler==2)
before being saved — the build raises rather than persists a broken mesh.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import trimesh

from vagus_fm.config import Config, ShrinkwrapParams
from vagus_fm.io.hdf5 import (
    CompartmentMesh,
    Geometry,
    SchemaError,
    save_geometry,
    validate_geometry,
)
from vagus_fm.io.stl import concat_stls, load_stl, load_stl_glob
from vagus_fm.mesh.repair import (
    boolean_union_overlapping,
    cheap_repair,
    is_perfect,
    pymeshfix_pass,
)
from vagus_fm.mesh.shrinkwrap import shrinkwrap_mesh

logger = logging.getLogger(__name__)


def _find_vagus_side(paths: list[Path], side: str) -> list[Path]:
    pattern = re.compile(rf"\b{side}\b", re.IGNORECASE)
    return sorted(p for p in paths if pattern.search(p.stem))


def gather_inputs(cfg: Config) -> dict[str, list[Path]]:
    """Resolve per-compartment STL inputs.

    Bone uses cleaned per-bone STLs in ``cfg.outputs.intermediate_bone_clean``
    if present, else falls back to ``cfg.data.bone_dir``. The torso skin is
    a single file; vagus left/right are matched by glob.
    """
    bone_clean = cfg.outputs.intermediate_bone_clean
    if bone_clean.exists() and any(bone_clean.glob("*.stl")):
        bone_paths = sorted(bone_clean.glob("*.stl"))
        logger.info("bone: using cleaned STLs from %s (%d files)", bone_clean, len(bone_paths))
    else:
        bone_paths = sorted(Path(cfg.data.bone_dir).glob("*.stl"))
        logger.info("bone: using raw STLs from %s (%d files)", cfg.data.bone_dir, len(bone_paths))
    if not bone_paths:
        raise SchemaError(
            f"no bone STLs in {cfg.outputs.intermediate_bone_clean} or {cfg.data.bone_dir}"
        )

    if not cfg.data.torso_skin.exists():
        raise SchemaError(f"torso skin STL not found: {cfg.data.torso_skin}")
    torso_paths = [cfg.data.torso_skin]

    vl = load_stl_glob(cfg.data.vagus_left_glob)
    vr = load_stl_glob(cfg.data.vagus_right_glob)
    muscle_paths = sorted(Path(cfg.data.muscle_dir).glob("*.stl"))
    vessel_paths = sorted(Path(cfg.data.vessel_dir).glob("*.stl"))
    logger.info("muscle: %d STLs from %s", len(muscle_paths), cfg.data.muscle_dir)
    logger.info("blood_vessel: %d STLs from %s", len(vessel_paths), cfg.data.vessel_dir)
    return {
        "mesh_skin": torso_paths,
        "mesh_bone": bone_paths,
        "mesh_muscle": muscle_paths,
        "mesh_blood_vessel": vessel_paths,
        "mesh_vagus_left": vl,
        "mesh_vagus_right": vr,
    }


def watertighten(
    raw: trimesh.Trimesh, label: str, params: ShrinkwrapParams,
    *, force_shrinkwrap: bool = False,
) -> trimesh.Trimesh:
    """Run the layered watertightening pipeline on a single compartment."""
    logger.info("[build] %s: input %d V / %d F", label, len(raw.vertices), len(raw.faces))

    # 1. Cheap repair
    m = cheap_repair(raw)
    logger.info(
        "  cheap repair: wt=%s euler=%s wcons=%s",
        m.is_watertight, m.euler_number, m.is_winding_consistent,
    )
    if not force_shrinkwrap and is_perfect(m):
        logger.info("  [done] cheap repair sufficed")
        return m

    # 2. Boolean union
    if not force_shrinkwrap and len(m.split(only_watertight=False)) > 1:
        try:
            mu = cheap_repair(boolean_union_overlapping(m))
            logger.info("  boolean union: wt=%s euler=%s", mu.is_watertight, mu.euler_number)
            if is_perfect(mu):
                logger.info("  [done] boolean union sufficed")
                return mu
            m = mu
        except Exception as e:
            logger.warning("boolean union step failed: %s", e)

    # 3. Shrinkwrap
    logger.info(
        "  shrinkwrap pitch=%g n=%d close=%d decim=%d smooth=%d",
        params.pitch, params.n_samples, params.close_iter,
        params.decimate_target, params.smooth_iter,
    )
    out = shrinkwrap_mesh(m, params)
    logger.info(
        "  shrinkwrap: V=%d F=%d wt=%s euler=%s",
        len(out.vertices), len(out.faces), out.is_watertight, out.euler_number,
    )

    # Bone-specific escalation: try heavier closing, then coarsen.
    if label == "mesh_bone" and not is_perfect(out):
        for ci in (4, 6):
            logger.info("  [retry] bone shrinkwrap close_iter=%d", ci)
            out = shrinkwrap_mesh(
                m,
                ShrinkwrapParams(
                    pitch=params.pitch, n_samples=params.n_samples,
                    close_iter=ci, decimate_target=params.decimate_target,
                    smooth_iter=params.smooth_iter,
                ),
            )
            logger.info("    wt=%s euler=%s", out.is_watertight, out.euler_number)
            if is_perfect(out):
                break
        if not is_perfect(out):
            logger.info("  [retry] bone last-resort: pitch=5.0, close_iter=6")
            out = shrinkwrap_mesh(
                m,
                ShrinkwrapParams(
                    pitch=5.0, n_samples=400_000, close_iter=6,
                    decimate_target=15_000, smooth_iter=12,
                ),
            )

    # 4. pymeshfix
    if not is_perfect(out):
        try:
            logger.info("  pymeshfix: closing remaining holes")
            out2 = pymeshfix_pass(out)
            logger.info("    wt=%s euler=%s", out2.is_watertight, out2.euler_number)
            if is_perfect(out2):
                out = out2
        except Exception as e:
            logger.warning("pymeshfix failed: %s", e)
    return out


def _validate_compartment(m: trimesh.Trimesh, label: str, cfg: Config) -> None:
    wt = bool(m.is_watertight)
    wc = bool(m.is_winding_consistent)
    eu = int(m.euler_number)
    vol = float(m.volume) if wt else float("nan")
    logger.info(
        "[validate] %s: watertight=%s winding=%s Euler=%d volume=%.0f mm^3",
        label, wt, wc, eu, vol,
    )
    g = cfg.geometry.validate
    if g.require_watertight and not wt:
        raise SchemaError(f"{label} not watertight")
    if g.require_winding_consistent and not wc:
        raise SchemaError(f"{label} winding inconsistent")
    if g.require_euler_2 and eu != 2:
        raise SchemaError(f"{label} Euler number {eu} != 2 (genus 0)")


def _trimesh_to_compartment(name: str, m: trimesh.Trimesh) -> CompartmentMesh:
    return CompartmentMesh(
        name=name,
        vertices=np.asarray(m.vertices, dtype=np.float64),
        faces=np.asarray(m.faces, dtype=np.int64),
    )


def build_geometry(
    cfg: Config, *, force_shrinkwrap: bool = False,
    only_compartments: tuple[str, ...] | None = None,
) -> Path:
    """Build the multi-compartment geometry HDF5 from ``cfg``.

    Returns the absolute output path.
    """
    inputs = gather_inputs(cfg)
    logger.info("project_root=%s", cfg.project_root)

    compartments: dict[str, CompartmentMesh] = {}
    for label, paths in inputs.items():
        if only_compartments and label not in only_compartments:
            logger.info("[skip] %s (not in --only)", label)
            continue
        if not paths:
            logger.warning("[skip] %s: no input files", label)
            continue
        if label not in cfg.geometry.shrinkwrap:
            raise SchemaError(
                f"no shrinkwrap params for compartment {label!r} in cfg.geometry.shrinkwrap"
            )
        params = cfg.geometry.shrinkwrap[label]
        if len(paths) == 1:
            raw = load_stl(paths[0], check_units_mm=True)
        else:
            raw = concat_stls(paths, check_units_mm=True)
        clean = watertighten(raw, label, params, force_shrinkwrap=force_shrinkwrap)
        _validate_compartment(clean, label, cfg)
        compartments[label] = _trimesh_to_compartment(label, clean)

    if not compartments:
        raise SchemaError("no compartments built (check inputs and --only)")

    geom = Geometry(compartments=compartments)
    validate_geometry(geom)

    out_path = cfg.outputs.geometry_mat
    save_geometry(out_path, geom)
    size_mb = out_path.stat().st_size / 1e6
    logger.info("[saved] %s (%.1f MB, %d compartments)", out_path, size_mb, len(compartments))
    return out_path


def check_existing(cfg: Config) -> bool:
    """Validate compartments already in ``cfg.outputs.geometry_mat`` without rebuilding.

    Returns True iff every compartment passes; logs failures.
    """
    p = cfg.outputs.geometry_mat
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist; nothing to check")
    from vagus_fm.io.hdf5 import load_geometry
    geom = load_geometry(p)
    validate_geometry(geom)
    all_ok = True
    for name, comp in geom.compartments.items():
        m = trimesh.Trimesh(comp.vertices, comp.faces, process=False)
        try:
            _validate_compartment(m, name, cfg)
        except SchemaError as e:
            logger.error("[FAIL] %s", e)
            all_ok = False
    return all_ok
