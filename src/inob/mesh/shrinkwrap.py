"""Voxel-shrinkwrap reconstruction.

Build a watertight, manifold approximation of a (possibly broken) surface mesh
by voxelising it, dilating + closing the occupancy, and re-extracting the
surface via marching cubes. Used as the third tier of the watertightening
pipeline in :mod:`inob.geometry.builder`.
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.measure import marching_cubes

from inob.config import ShrinkwrapParams
from inob.mesh.repair import cheap_repair

logger = logging.getLogger(__name__)


def occupancy_from_mesh(
    mesh: trimesh.Trimesh,
    *,
    pitch: float,
    n_samples: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a binary occupancy grid for ``mesh``.

    For watertight inputs we use ``trimesh.voxelized().fill()`` (interior
    determined by ray-cast). For everything else we splat surface samples
    into a voxel grid; subsequent ``binary_closing`` + ``binary_fill_holes``
    reconstructs the interior.

    ``seed`` is passed through to the surface sampler. Leaving it ``None``
    draws from the *global* numpy RNG, which makes each compartment's result
    depend on how many compartments were built before it — so adding or
    removing one silently perturbs the geometry of all the others. Callers
    should pass an explicit per-compartment seed.

    Returns ``(occ, transform)`` where ``transform`` is the ``(4, 4)``
    voxel-index → world-mm matrix.
    """
    if mesh.is_watertight and mesh.euler_number == 2:
        try:
            vox = mesh.voxelized(pitch=pitch).fill()
            return (
                np.asarray(vox.matrix, dtype=bool),
                np.asarray(vox.transform, dtype=np.float64),
            )
        except Exception as e:
            logger.debug("ray-trace voxelize failed (%s); falling back to surface splat", e)
    pts, _ = trimesh.sample.sample_surface(mesh, count=n_samples, seed=seed)
    pad = pitch * 4
    mn = pts.min(0) - pad
    mx = pts.max(0) + pad
    shape = tuple(((mx - mn) / pitch).astype(int) + 1)
    ijk = np.clip(((pts - mn) / pitch).astype(int), 0, np.array(shape) - 1)
    occ = np.zeros(shape, dtype=bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    T = np.eye(4)
    T[:3, :3] = np.eye(3) * pitch
    T[:3, 3] = mn
    return occ, T


def shrinkwrap_mesh(
    mesh: trimesh.Trimesh,
    params: ShrinkwrapParams,
    *,
    seed: int | None = None,
) -> trimesh.Trimesh:
    """Voxel-shrinkwrap pipeline.

    Sample → occupancy → close + fill → marching cubes → quadric decimation
    → Taubin smoothing → cheap repair. Result is intrinsically watertight
    (a single closed surface from marching cubes).

    ``seed`` makes the surface sampling reproducible and independent of build
    order — see :func:`occupancy_from_mesh`.
    """
    occ, T = occupancy_from_mesh(
        mesh,
        pitch=params.pitch,
        n_samples=params.n_samples,
        seed=seed,
    )
    if params.close_iter > 0:
        occ = binary_closing(occ, iterations=params.close_iter)
    occ = binary_fill_holes(occ)

    occ_p = np.pad(occ, 1, mode="constant", constant_values=False)
    v_mc, f_mc, _, _ = marching_cubes(occ_p.astype(np.float32), level=0.5)
    v_mc = v_mc - 1.0
    v_world = v_mc @ T[:3, :3].T + T[:3, 3]

    out = trimesh.Trimesh(v_world, f_mc, process=True)
    out.merge_vertices()
    out.remove_unreferenced_vertices()

    if len(out.faces) > params.decimate_target:
        try:
            out_s = out.simplify_quadric_decimation(face_count=params.decimate_target)
            if out_s is not None and len(out_s.faces) > 0:
                out = out_s
        except Exception as e:
            logger.warning("quadric decimation failed: %s", e)

    if params.smooth_iter > 0:
        # Taubin smoothing occasionally ejects a handful of vertices far outside
        # the surface, leaving a long thin spike: the mesh stays watertight with
        # euler == 2, so nothing downstream notices, but the bounding box is
        # wrong. On the spinal cord it added a 100 mm spike below the conus.
        # Smoothing may move vertices; it may not grow the object.
        pre_bounds = out.bounds.copy()
        pre_vertices = np.asarray(out.vertices).copy()
        try:
            trimesh.smoothing.filter_taubin(out, iterations=params.smooth_iter)
        except Exception as e:
            logger.warning("Taubin smoothing failed: %s", e)
        else:
            margin = 2.0 * params.pitch
            escaped = (out.vertices < pre_bounds[0] - margin).any() or (
                out.vertices > pre_bounds[1] + margin
            ).any()
            if escaped:
                logger.warning(
                    "Taubin smoothing pushed vertices outside the pre-smoothing "
                    "bbox (extent %s -> %s mm); reverting to the unsmoothed mesh",
                    np.round(pre_bounds[1] - pre_bounds[0], 1),
                    np.round(out.bounds[1] - out.bounds[0], 1),
                )
                out.vertices = pre_vertices

    out = cheap_repair(out)
    parts = out.split(only_watertight=False)
    if len(parts) > 1:
        parts.sort(key=lambda p: len(p.faces), reverse=True)
        out = cheap_repair(parts[0])
    return out
