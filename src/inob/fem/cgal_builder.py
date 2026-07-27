"""Build a multi-tissue tetrahedral FEM via iso2mesh + CGAL.

Voxelises each tissue (vagus_left, vagus_right, bone, skin) into a single
labelled image, then calls ``iso2mesh.cgalv2m`` (CGAL 3-D Mesh_3) to produce
a multi-region tet mesh in one call. No watertight / manifold requirements
on input surfaces, so this path tolerates the messy real-world bone STLs.

Pipeline:

  1. Load the per-compartment surfaces (skin, bones, vagus left/right).
  2. Build a single global voxel grid covering all tissues with padding.
  3. Voxelise:
       - skin via :func:`voxelize_mesh` + flood + cavity-fill
       - bones via per-bone convex-hull union, clipped to skin
       - vagus tubes via exact ray-trace + small dilation
  4. Compose into a labelled image (vagus_left=1 .. skin=4) — innermost
     tissues painted last so they overwrite the envelope.
  5. ``iso2mesh.cgalv2m`` → (nodes_voxels, elements, faces).
  6. Convert nodes back to mm; remap region IDs to contiguous 1..K.
  7. Validate (mesh quality, units, contiguous tissue IDs) and save.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from inob.config import Config
from inob.io.hdf5 import FemMesh, save_fem, validate_fem
from inob.io.stl import load_stl, load_stl_glob
from inob.mesh.quality import (
    assert_mesh_ok,
    assert_units_mm,
    compute_quality,
)
from inob.mesh.voxelize import (
    fill_internal_cavities,
    keep_largest_component,
    voxelize_mesh,
    voxelize_solid_for_mesh,
    voxelize_watertight,
)

logger = logging.getLogger(__name__)


# How each multi-STL tissue is turned into voxels. Declared per tissue rather
# than branched inline, so adding a target means adding a row here.
#
#   method="solid"    convex-hull fill. Correct only for convex, blob-like
#                     structures. A hull chords across any curvature or
#                     concavity, over-filling and displacing the centroid.
#   method="surface"  surface splat + closing + exterior flood-fill. Follows
#                     the true shape; the right choice for anything thin,
#                     curved, or sheet-like.
#
# Measured on the shipped atlas: switching the spinal cord from "solid" to
# "surface" cut its volume from 201 cm^3 to 60 cm^3 (true ~31 cm^3 at this
# 3 mm pitch) and moved its mid-thoracic centroid 24 mm to within 1.5 mm of
# the true cord axis — which is where every sampled source dipole sits.
VOXELISATION: dict[str, dict] = {
    # Individually compact bones; hulls are a good approximation and fast.
    "bone":         {"method": "solid"},
    # Thin curved sheets (platysma, splenius) wrap the neck; a hull would
    # fill it with non-muscle tissue and misplace muscle source dipoles.
    "muscle":       {"method": "surface"},
    # Narrow tubes: dilate so they survive the voxel grid at all.
    "blood_vessel": {"method": "solid", "dilate_voxels": 1},
    # One long mesh following the cervical lordosis / thoracic kyphosis.
    "spinal_cord":  {"method": "surface"},
}


def _build_grid(
    all_v: np.ndarray, *, pitch: float, pad: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int],
           np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mn, mx, shape, X, Y, Z)`` for the global voxel grid."""
    mn = all_v.min(0) - pad
    mx = all_v.max(0) + pad
    shape = tuple(((mx - mn) / pitch).astype(int) + 1)
    xs = mn[0] + (np.arange(shape[0]) + 0.5) * pitch
    ys = mn[1] + (np.arange(shape[1]) + 0.5) * pitch
    zs = mn[2] + (np.arange(shape[2]) + 0.5) * pitch
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return mn, mx, shape, X, Y, Z


def _load_tissue_surfaces(cfg: Config):
    """Return ``(m_skin, m_vagus_left, m_vagus_right, bone_paths, muscle_paths, vessel_paths, spinal_cord_paths)``."""
    m_skin = load_stl(cfg.data.torso_skin)
    vl_paths = load_stl_glob(cfg.data.vagus_left_glob)
    vr_paths = load_stl_glob(cfg.data.vagus_right_glob)
    m_vl = load_stl(vl_paths[0])
    m_vr = load_stl(vr_paths[0])

    bone_clean = cfg.outputs.intermediate_bone_clean
    if bone_clean.exists() and any(bone_clean.glob("*.stl")):
        bone_paths = sorted(bone_clean.glob("*.stl"))
        logger.info("bone: using cleaned STLs from %s (%d files)", bone_clean, len(bone_paths))
    else:
        bone_paths = sorted(Path(cfg.data.bone_dir).glob("*.stl"))
        logger.info("bone: using raw STLs from %s (%d files)", cfg.data.bone_dir, len(bone_paths))
    if not bone_paths:
        from inob.io.stl import STLLoadError
        raise STLLoadError(f"no bone STLs in {bone_clean} or {cfg.data.bone_dir}")

    muscle_paths = sorted(Path(cfg.data.muscle_dir).glob("*.stl"))
    vessel_paths = sorted(Path(cfg.data.vessel_dir).glob("*.stl"))
    spinal_cord_paths = sorted(Path(cfg.data.spinal_cord_dir).glob("*.stl"))
    logger.info("muscle: %d STLs from %s", len(muscle_paths), cfg.data.muscle_dir)
    logger.info("blood_vessel: %d STLs from %s", len(vessel_paths), cfg.data.vessel_dir)
    logger.info("spinal_cord: %d STLs from %s", len(spinal_cord_paths), cfg.data.spinal_cord_dir)
    return m_skin, m_vl, m_vr, bone_paths, muscle_paths, vessel_paths, spinal_cord_paths


def _voxelise_skin(m_skin, X, *, pitch, mn, closing_mm: float) -> np.ndarray:
    logger.info("Voxelising skin (closing=%.1f mm)…", closing_mm)
    occ = voxelize_mesh(m_skin, X, pitch=pitch, mn=mn, closing_mm=closing_mm)
    occ = keep_largest_component(occ)
    occ = fill_internal_cavities(occ)
    logger.info("  skin voxels: %d", int(occ.sum()))
    return occ


def _voxelise_group_solid(
    paths, X, Y, Z, *, pitch, mn, skin_occ, label: str, dilate_voxels: int = 0,
    method: str = "solid", closing_mm: float = 2.0,
) -> np.ndarray:
    """Per-mesh voxelisation, unioned and clipped to skin.

    Used for bone, muscle and blood_vessel — each is a set of independent
    surfaces. ``dilate_voxels`` thickens thin structures (vessels) so they
    survive the voxel grid.

    ``method`` selects how each mesh is filled:
      * ``"solid"``   — convex-hull occupancy. Fast and fine for blob-like
        structures (bone, compact muscles), but a convex hull grossly
        over-fills thin, curved, or C-shaped meshes (e.g. platysma wraps the
        neck, so its hull is a solid wedge ~5× the true volume).
      * ``"surface"`` — surface-splat + closing + exterior flood-fill (the same
        routine used for skin). Follows the true mesh shape, so it is the right
        choice for anatomically non-convex muscles.
    """
    logger.info("Voxelising %s (%d meshes, per-mesh %s, clipped to skin)…",
                label, len(paths), method)
    occ = np.zeros(X.shape, dtype=bool)
    for i, p in enumerate(paths):
        bm = load_stl(p, check_units_mm=False)
        if method == "surface":
            occ |= voxelize_mesh(bm, X, pitch=pitch, mn=mn, closing_mm=closing_mm)
        else:
            occ |= voxelize_solid_for_mesh(bm, X, Y, Z, pitch=pitch, mn=mn)
        if (i + 1) % 25 == 0:
            logger.debug("  %s %d/%d", label, i + 1, len(paths))
    if dilate_voxels > 0:
        from scipy.ndimage import binary_dilation
        occ = binary_dilation(occ, iterations=dilate_voxels)
    occ &= skin_occ
    logger.info("  %s voxels: %d", label, int(occ.sum()))
    return occ


def _voxelise_vagus(mesh, X, *, pitch, mn, dilate_voxels: int, skin_occ) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    occ = voxelize_watertight(mesh, X, pitch=pitch, mn=mn)
    if dilate_voxels > 0:
        occ = binary_dilation(occ, iterations=dilate_voxels)
    occ &= skin_occ
    return occ


def _assemble_label_volume(
    cfg: Config, *,
    skin_occ, bone_occ, vl_occ, vr_occ, muscle_occ=None, vessel_occ=None,
    spinal_cord_occ=None,
) -> tuple[np.ndarray, list[str]]:
    """Compose a labelled image; tissue order matches ``cfg.fem.tissues``.

    Tissues are painted in declaration order so later (innermost) tissues
    overwrite earlier ones. Returns the volume + the actual order used.
    """
    label_vol = np.zeros(skin_occ.shape, dtype=np.uint8)
    zero = np.zeros_like(skin_occ)
    tissue_to_occ = {
        "skin": skin_occ, "bone": bone_occ,
        "muscle": muscle_occ if muscle_occ is not None else zero,
        "blood_vessel": vessel_occ if vessel_occ is not None else zero,
        "spinal_cord": spinal_cord_occ if spinal_cord_occ is not None else zero,
        "vagus_left": vl_occ, "vagus_right": vr_occ,
    }
    # Paint outermost → innermost so the target (vagus) and the conductive
    # vessel survive over the surrounding muscle/bone at voxel interfaces.
    # Spinal cord sits inside the vertebral canal (inside bone), so it is
    # painted after bone.
    paint_order = [t for t in ("skin", "muscle", "bone", "spinal_cord",
                               "blood_vessel", "vagus_right", "vagus_left")
                   if t in cfg.fem.tissues]
    # The CGAL region IDs we want match ``cfg.fem.tissues`` order (1..K).
    label_to_id = {lab: i + 1 for i, lab in enumerate(cfg.fem.tissues)}
    for lab in paint_order:
        occ = tissue_to_occ[lab]
        if occ is None or not occ.any():
            logger.warning("  tissue %r is empty — check inputs", lab)
            continue
        label_vol[occ] = label_to_id[lab]
        logger.info("  painted %s → id %d (%d voxels)", lab, label_to_id[lab], int(occ.sum()))
    return label_vol, list(cfg.fem.tissues)


def _remap_region_ids(
    elem_regions: np.ndarray, declared: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Map raw CGAL region IDs (whatever values they emit) to contiguous 1..K.

    ``declared`` is the desired tissue order; output IDs follow that order
    over the regions actually present in the mesh, dropping declared tissues
    that did not survive the meshing step.
    """
    present = sorted(set(int(r) for r in np.unique(elem_regions)))
    # Trust the painter: each declared id 1..K either appears in `present` or not.
    order = [r for r in range(1, len(declared) + 1) if r in present]
    remap = {r: i + 1 for i, r in enumerate(order)}
    new_ids = np.array([remap[int(r)] for r in elem_regions], dtype=np.int32)
    new_labels = [declared[r - 1] for r in order]
    return new_ids, new_labels


def build_fem(cfg: Config) -> Path:
    """Build the FEM mesh per ``cfg`` and write ``cfg.outputs.fem_mat``."""
    import iso2mesh as im

    fcfg = cfg.fem
    pitch = fcfg.pitch_mm
    pad = max(fcfg.pad_mm, 4 * pitch)

    m_skin, m_vl, m_vr, bone_paths, muscle_paths, vessel_paths, spinal_cord_paths = _load_tissue_surfaces(cfg)
    logger.info("  skin       : %d V / %d F", len(m_skin.vertices), len(m_skin.faces))
    logger.info("  vagus_L    : %d V / %d F", len(m_vl.vertices), len(m_vl.faces))
    logger.info("  vagus_R    : %d V / %d F", len(m_vr.vertices), len(m_vr.faces))
    logger.info("  bones      : %d STLs", len(bone_paths))
    logger.info("  muscles    : %d STLs", len(muscle_paths))
    logger.info("  vessels    : %d STLs", len(vessel_paths))
    logger.info("  spinal cord: %d STLs", len(spinal_cord_paths))

    # Assemble bbox over all tissues
    all_v = [m_skin.vertices, m_vl.vertices, m_vr.vertices]
    for bp in (*bone_paths, *muscle_paths, *vessel_paths, *spinal_cord_paths):
        all_v.append(np.asarray(load_stl(bp, check_units_mm=False).vertices))
    all_v = np.vstack(all_v)
    mn, _mx, shape, X, Y, Z = _build_grid(all_v, pitch=pitch, pad=pad)
    logger.info("Grid shape=%s (%.1fM voxels)", shape, np.prod(shape) / 1e6)

    skin_occ = _voxelise_skin(m_skin, X, pitch=pitch, mn=mn, closing_mm=fcfg.bone_closing_mm)

    def _voxelise(label: str, paths, **kw) -> np.ndarray:
        if label not in fcfg.tissues or not paths:
            return np.zeros_like(skin_occ)
        opts = {**VOXELISATION[label], **kw}
        return _voxelise_group_solid(paths, X, Y, Z, pitch=pitch, mn=mn,
                                     skin_occ=skin_occ, label=label, **opts)

    bone_occ = _voxelise("bone", bone_paths)
    muscle_occ = _voxelise("muscle", muscle_paths)
    vessel_occ = _voxelise("blood_vessel", vessel_paths)
    spinal_cord_occ = _voxelise("spinal_cord", spinal_cord_paths)
    vl_occ = _voxelise_vagus(m_vl, X, pitch=pitch, mn=mn,
                              dilate_voxels=fcfg.vagus_dilate_voxels, skin_occ=skin_occ) \
        if "vagus_left" in fcfg.tissues else np.zeros_like(skin_occ)
    vr_occ = _voxelise_vagus(m_vr, X, pitch=pitch, mn=mn,
                              dilate_voxels=fcfg.vagus_dilate_voxels, skin_occ=skin_occ) \
        if "vagus_right" in fcfg.tissues else np.zeros_like(skin_occ)

    label_vol, declared = _assemble_label_volume(
        cfg, skin_occ=skin_occ, bone_occ=bone_occ, vl_occ=vl_occ, vr_occ=vr_occ,
        muscle_occ=muscle_occ, vessel_occ=vessel_occ, spinal_cord_occ=spinal_cord_occ,
    )

    logger.info("Running iso2mesh.cgalv2m (radbound=%g, maxvol=%g)",
                fcfg.radbound, fcfg.maxvol)
    node, elem, _face = im.cgalv2m(label_vol, fcfg.radbound, fcfg.maxvol)
    node_mm = node[:, :3] * pitch + mn
    elem_int = np.asarray(elem, dtype=np.int64)
    tets_0idx = elem_int[:, :4] - 1
    raw_regions = elem_int[:, 4]

    tissue_ids, present_labels = _remap_region_ids(raw_regions, declared)
    logger.info("CGAL output: %d nodes, %d tets, regions=%s",
                len(node_mm), len(tets_0idx), present_labels)
    for r in range(1, len(present_labels) + 1):
        cnt = int((tissue_ids == r).sum())
        logger.info("  region %d (%-11s): %d tets", r, present_labels[r - 1], cnt)

    mesh = FemMesh(
        nodes=np.asarray(node_mm, dtype=np.float64),
        tets=tets_0idx.astype(np.int32),
        tissue=tissue_ids,
        tissue_labels=tuple(present_labels),
    )

    if fcfg.validate.require_units_mm:
        assert_units_mm(mesh.nodes)
    quality = compute_quality(mesh.nodes, mesh.tets)
    logger.info("Mesh quality: min=%.4f p1=%.4f p5=%.4f mean=%.4f max=%.4f",
                quality.min, quality.p1, quality.p5, quality.mean, quality.max)
    assert_mesh_ok(quality, min_quality=fcfg.validate.min_mesh_quality)
    validate_fem(
        mesh,
        require_contiguous_tissue_ids=fcfg.validate.require_contiguous_tissue_ids,
        require_units_mm=fcfg.validate.require_units_mm,
    )

    save_fem(cfg.outputs.fem_mat, mesh)
    logger.info("[saved] %s (%.1f MB)",
                cfg.outputs.fem_mat, cfg.outputs.fem_mat.stat().st_size / 1e6)
    return cfg.outputs.fem_mat
