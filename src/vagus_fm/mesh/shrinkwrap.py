"""Voxel-shrinkwrap reconstruction.

Build a watertight, manifold approximation of a (possibly broken) surface mesh
by voxelising it, dilating + closing the occupancy, and re-extracting the
surface via marching cubes. Used as the third tier of the watertightening
pipeline in :mod:`vagus_fm.geometry.builder`.
"""
from __future__ import annotations

import logging

import numpy as np
import trimesh
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.measure import marching_cubes

from vagus_fm.config import ShrinkwrapParams
from vagus_fm.mesh.repair import cheap_repair

logger = logging.getLogger(__name__)


def occupancy_from_mesh(
    mesh: trimesh.Trimesh, *, pitch: float, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a binary occupancy grid for ``mesh``.

    For watertight inputs we use ``trimesh.voxelized().fill()`` (interior
    determined by ray-cast). For everything else we splat surface samples
    into a voxel grid; subsequent ``binary_closing`` + ``binary_fill_holes``
    reconstructs the interior.

    Returns ``(occ, transform)`` where ``transform`` is the ``(4, 4)``
    voxel-index → world-mm matrix.
    """
    if mesh.is_watertight:
        try:
            vox = mesh.voxelized(pitch=pitch).fill()
            return (
                np.asarray(vox.matrix, dtype=bool),
                np.asarray(vox.transform, dtype=np.float64),
            )
        except Exception as e:
            logger.debug("ray-trace voxelize failed (%s); falling back to surface splat", e)
    pts, _ = trimesh.sample.sample_surface(mesh, count=n_samples)
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
    mesh: trimesh.Trimesh, params: ShrinkwrapParams
) -> trimesh.Trimesh:
    """Voxel-shrinkwrap pipeline.

    Sample → occupancy → close + fill → marching cubes → quadric decimation
    → Taubin smoothing → cheap repair. Result is intrinsically watertight
    (a single closed surface from marching cubes).
    """
    occ, T = occupancy_from_mesh(
        mesh, pitch=params.pitch, n_samples=params.n_samples
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
        try:
            trimesh.smoothing.filter_taubin(out, iterations=params.smooth_iter)
        except Exception as e:
            logger.warning("Taubin smoothing failed: %s", e)

    out = cheap_repair(out)
    parts = out.split(only_watertight=False)
    if len(parts) > 1:
        parts.sort(key=lambda p: len(p.faces), reverse=True)
        out = cheap_repair(parts[0])
    return out
