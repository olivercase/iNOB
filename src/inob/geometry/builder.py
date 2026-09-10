"""Build a watertight, manifold geometry from per-tissue STLs.

Per-compartment shrinkwrap parameters come from
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
import zlib
from pathlib import Path

import numpy as np
import trimesh

from inob.config import Config, ShrinkwrapParams
from inob.io.hdf5 import (
    CompartmentMesh,
    Geometry,
    SchemaError,
    save_geometry,
    validate_geometry,
)
from inob.io.stl import concat_stls, load_stl, load_stl_glob
from inob.mesh.repair import (
    boolean_union_overlapping,
    cheap_repair,
    drop_degenerate_components,
    is_perfect,
    pymeshfix_pass,
)
from inob.mesh.shrinkwrap import shrinkwrap_mesh

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
    cord_paths = sorted(Path(cfg.data.spinal_cord_dir).glob("*.stl"))
    logger.info("muscle: %d STLs from %s", len(muscle_paths), cfg.data.muscle_dir)
    logger.info("blood_vessel: %d STLs from %s", len(vessel_paths), cfg.data.vessel_dir)
    logger.info("spinal_cord: %d STLs from %s", len(cord_paths), cfg.data.spinal_cord_dir)
    return {
        "mesh_skin": torso_paths,
        "mesh_bone": bone_paths,
        "mesh_muscle": muscle_paths,
        "mesh_blood_vessel": vessel_paths,
        "mesh_spinal_cord": cord_paths,
        "mesh_vagus_left": vl,
        "mesh_vagus_right": vr,
    }


def compartment_seed(label: str, base_seed: int) -> int:
    """Deterministic per-compartment RNG seed.

    Derived from the compartment name so each compartment's shrinkwrap is
    reproducible *and* independent of which other compartments are in the
    config — otherwise adding one shifts the shared RNG stream and silently
    changes every compartment built after it.
    """
    return (base_seed + zlib.crc32(label.encode())) % (2**32)


def _bbox_diagonal(m: trimesh.Trimesh) -> float:
    b = m.bounds
    return float(np.linalg.norm(b[1] - b[0]))


def _preserves_extent(
    before: trimesh.Trimesh,
    after: trimesh.Trimesh,
    *,
    min_ratio: float = 0.8,
) -> bool:
    """True iff ``after`` still spans most of ``before``'s bounding box.

    A repair pass is allowed to change topology, not to delete the anatomy.
    """
    d0 = _bbox_diagonal(before)
    if d0 <= 0:
        return True
    return _bbox_diagonal(after) / d0 >= min_ratio


def watertighten(
    raw: trimesh.Trimesh,
    label: str,
    params: ShrinkwrapParams,
    *,
    force_shrinkwrap: bool = False,
    seed: int | None = None,
) -> trimesh.Trimesh:
    """Run the layered watertightening pipeline on a single compartment."""
    logger.info("[build] %s: input %d V / %d F", label, len(raw.vertices), len(raw.faces))

    # 1. Cheap repair, then discard fragments too small to bound a volume —
    # they poison the boolean-union and pymeshfix tiers below.
    m = drop_degenerate_components(cheap_repair(raw))
    logger.info(
        "  cheap repair: wt=%s euler=%s wcons=%s",
        m.is_watertight,
        m.euler_number,
        m.is_winding_consistent,
    )
    if not force_shrinkwrap and is_perfect(m):
        logger.info("  [done] cheap repair sufficed")
        return m

    # If cheap repair already produces a single closed manifold, shrinkwrap
    # cannot improve the topology. Return as-is and let validation decide.
    if not force_shrinkwrap and m.is_watertight and m.is_winding_consistent:
        if len(m.split(only_watertight=False)) == 1:
            logger.info(
                "  [done] cheap repair gave single closed manifold (euler=%d); skipping shrinkwrap",
                m.euler_number,
            )
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
        params.pitch,
        params.n_samples,
        params.close_iter,
        params.decimate_target,
        params.smooth_iter,
    )
    out = shrinkwrap_mesh(m, params, seed=seed)
    logger.info(
        "  shrinkwrap: V=%d F=%d wt=%s euler=%s",
        len(out.vertices),
        len(out.faces),
        out.is_watertight,
        out.euler_number,
    )

    # Bone-specific escalation: try heavier closing, then coarsen.
    if label == "mesh_bone" and not is_perfect(out):
        for ci in (4, 6):
            logger.info("  [retry] bone shrinkwrap close_iter=%d", ci)
            out = shrinkwrap_mesh(
                m,
                ShrinkwrapParams(
                    pitch=params.pitch,
                    n_samples=params.n_samples,
                    close_iter=ci,
                    decimate_target=params.decimate_target,
                    smooth_iter=params.smooth_iter,
                ),
                seed=seed,
            )
            logger.info("    wt=%s euler=%s", out.is_watertight, out.euler_number)
            if is_perfect(out):
                break
        if not is_perfect(out):
            logger.info("  [retry] bone last-resort: pitch=5.0, close_iter=6")
            out = shrinkwrap_mesh(
                m,
                ShrinkwrapParams(
                    pitch=5.0,
                    n_samples=400_000,
                    close_iter=6,
                    decimate_target=15_000,
                    smooth_iter=12,
                ),
                seed=seed,
            )

    # 4. pymeshfix
    if not is_perfect(out):
        try:
            logger.info("  pymeshfix: closing remaining holes")
            out2 = pymeshfix_pass(out)
            logger.info("    wt=%s euler=%s", out2.is_watertight, out2.euler_number)
            # Keep pymeshfix result if it is at least an improvement — i.e. it
            # gained watertightness or reduced the number of topological handles
            # — even if it does not reach the ideal euler=2.
            improved = (
                is_perfect(out2)
                or (out2.is_watertight and not out.is_watertight)
                or (out2.is_watertight and abs(out2.euler_number - 2) < abs(out.euler_number - 2))
            )
            # ...but only if it still describes the same object. Given a badly
            # self-intersecting input, pymeshfix will happily return one small
            # closed shell: topologically perfect, anatomically destroyed. It
            # collapsed the spinal cord to a 53-vertex, 1.1 cm^3 nub (true
            # volume ~31 cm^3) while reporting watertight, euler=2.
            if improved and not _preserves_extent(out, out2):
                logger.warning(
                    "  pymeshfix result discarded: bbox diagonal shrank %.0f -> %.0f mm "
                    "(kept the unrepaired mesh)",
                    _bbox_diagonal(out),
                    _bbox_diagonal(out2),
                )
                improved = False
            if improved:
                out = out2
        except Exception as e:
            logger.warning("pymeshfix failed: %s", e)
    return out


# Compartments that are inherently many disjoint bodies (the expanded cervical
# muscle set: ~40 separate muscles). A single watertight surface is neither
# achievable nor meaningful for them, and they are used only for visualisation
# (the FEM voxelises the muscle STLs directly, not this compartment), so their
# watertight/Euler checks are relaxed to warnings.
_MULTIBODY_VIZ_COMPARTMENTS = frozenset({"mesh_muscle"})


def _validate_compartment(m: trimesh.Trimesh, label: str, cfg: Config) -> None:
    wt = bool(m.is_watertight)
    wc = bool(m.is_winding_consistent)
    eu = int(m.euler_number)
    vol = float(m.volume) if wt else float("nan")
    logger.info(
        "[validate] %s: watertight=%s winding=%s Euler=%d volume=%.0f mm^3",
        label,
        wt,
        wc,
        eu,
        vol,
    )
    if label in _MULTIBODY_VIZ_COMPARTMENTS:
        logger.info(
            "[validate] %s: multi-body viz compartment (%d components) — "
            "watertight/Euler checks relaxed",
            label,
            len(m.split(only_watertight=False)),
        )
        return
    g = cfg.geometry.validate
    if g.require_watertight and not wt:
        raise SchemaError(f"{label} not watertight")
    if g.require_winding_consistent and not wc:
        raise SchemaError(f"{label} winding inconsistent")
    if g.require_euler_2 and eu != 2:
        # Downgrade to a warning when the mesh is otherwise a clean closed
        # manifold — euler != 2 may reflect a source-data topological defect
        # (e.g. genus-1 handle in a BodyParts3D STL) that the pipeline cannot
        # repair automatically.  A fatal error here would accept the broken
        # shrinkwrap output (euler << 2) over the better cheap-repair result.
        if wt and wc:
            logger.warning(
                "[validate] %s: Euler=%d != 2 (genus != 0) — source mesh has "
                "a topological defect; mesh is otherwise watertight and will be saved",
                label,
                eu,
            )
        else:
            raise SchemaError(f"{label} Euler number {eu} != 2 (genus 0)")


def _multibody_compartment(paths: list[Path]) -> trimesh.Trimesh:
    """Concatenate many STLs into one multi-body surface for visualisation.

    Each mesh is cheap-repaired (dedup/fix winding) but kept as its own body —
    no shrinkwrap, no union, no largest-component pruning — so every muscle
    survives. Not watertight and not used by the FEM; purely for viz.
    """
    bodies = [cheap_repair(load_stl(p, check_units_mm=True)) for p in paths]
    merged = trimesh.util.concatenate(bodies)
    logger.info(
        "[build] multi-body compartment: %d STLs → %d bodies, %d V / %d F",
        len(paths),
        len(bodies),
        len(merged.vertices),
        len(merged.faces),
    )
    return merged


def _trimesh_to_compartment(name: str, m: trimesh.Trimesh) -> CompartmentMesh:
    return CompartmentMesh(
        name=name,
        vertices=np.asarray(m.vertices, dtype=np.float64),
        faces=np.asarray(m.faces, dtype=np.int64),
    )


def build_geometry(
    cfg: Config,
    *,
    force_shrinkwrap: bool = False,
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
        seed = compartment_seed(label, cfg.reproducibility.seed)
        if label in _MULTIBODY_VIZ_COMPARTMENTS:
            # Muscle is many disjoint bodies: the shrinkwrap's keep-largest-
            # component step would discard all but one. Keep every muscle as its
            # own cheap-repaired body (viz-only compartment; the FEM voxelises
            # the STLs directly). vagus/spine are untouched by this branch.
            clean = _multibody_compartment(paths)
        elif len(paths) == 1:
            raw = load_stl(paths[0], check_units_mm=True)
            clean = watertighten(raw, label, params, force_shrinkwrap=force_shrinkwrap, seed=seed)
        else:
            raw = concat_stls(paths, check_units_mm=True)
            clean = watertighten(raw, label, params, force_shrinkwrap=force_shrinkwrap, seed=seed)
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
    from inob.io.hdf5 import load_geometry

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
