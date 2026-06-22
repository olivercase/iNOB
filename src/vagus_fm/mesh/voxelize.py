"""Voxelisation strategies for the CGAL-driven FEM builder.

Three orthogonal strategies, used by :mod:`vagus_fm.fem.cgal_builder` to
rasterise per-tissue surfaces into a labelled image before calling
``iso2mesh.cgalv2m``:

  * :func:`voxelize_solid`     — convex-hull occupancy (per-bone fast path)
  * :func:`voxelize_watertight` — exact ray-cast + flood fill (vagus tubes)
  * :func:`voxelize_mesh`      — surface splat + closing + exterior flood
                                   (skin, accommodates anatomical openings)
"""
from __future__ import annotations

import logging

import numpy as np
import trimesh
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
)
from scipy.ndimage import label as nd_label
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


def voxelize_solid(
    equations: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    *, eps: float = 1e-6,
) -> np.ndarray:
    """Test which grid cell centres lie inside a convex hull.

    ``equations`` is the ``(n, 4)`` matrix from :class:`scipy.spatial.ConvexHull`,
    where each row ``[a, b, c, d]`` defines the half-space ``a·p + d ≤ 0``.
    Returns a boolean array shaped like ``X``.
    """
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    inside = ((equations[:, :3] @ P.T + equations[:, 3:4]) <= eps).all(axis=0)
    return inside.reshape(X.shape)


def voxelize_watertight(
    mesh: trimesh.Trimesh, X: np.ndarray, *, pitch: float, mn: np.ndarray,
) -> np.ndarray:
    """Voxelise a *watertight* mesh exactly via trimesh ray-tracing + flood fill.

    Returns a boolean grid the shape of ``X``. Empty if the mesh produces
    no interior voxels (caller should treat this as a no-op).
    """
    vox = mesh.voxelized(pitch=pitch).fill()
    vmat = np.asarray(vox.matrix, dtype=bool)
    if not vmat.any():
        return np.zeros(X.shape, dtype=bool)
    i_l, j_l, k_l = np.nonzero(vmat)
    local_xyz = np.stack(
        [i_l, j_l, k_l, np.ones_like(i_l)], axis=1
    ).astype(float)
    world_xyz = (vox.transform @ local_xyz.T).T[:, :3]
    ijk = np.clip(((world_xyz - mn) / pitch).astype(int), 0, np.array(X.shape) - 1)
    out = np.zeros(X.shape, dtype=bool)
    out[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return out


def voxelize_mesh(
    mesh: trimesh.Trimesh, X: np.ndarray, *, pitch: float, mn: np.ndarray,
    closing_mm: float = 15.0, dilate_mm: float = 2.0,
) -> np.ndarray:
    """Surface-sample voxelisation for *non-watertight* meshes (e.g. skin).

    Pipeline: sample surface → splat into grid → dilate (~2 mm) →
    morphological close (``closing_mm``) → exterior flood fill → erode back.
    The exterior flood is critical because anatomical openings (mouth, nostrils,
    anus) leak the body interior to the outside, breaking the simpler
    ``binary_fill_holes`` approach.
    """
    shape = X.shape
    # ~8 samples per voxel face so the rasterised shell is solid even at fine pitch
    surf_area = float(mesh.area)
    n_target = max(800_000, int(8 * surf_area / pitch**2))
    pts, _ = trimesh.sample.sample_surface(mesh, count=n_target)
    occ = np.zeros(shape, dtype=bool)
    ijk = np.clip(((pts - mn) / pitch).astype(int), 0, np.array(shape) - 1)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    dilate_voxels = max(1, int(np.ceil(dilate_mm / pitch)))
    iters = max(1, int(np.ceil(closing_mm / pitch)))
    occ = binary_dilation(occ, iterations=dilate_voxels)
    occ = binary_closing(occ, iterations=iters)

    ext = ~occ
    lab, _ = nd_label(ext)
    boundary_label = lab[0, 0, 0]
    if boundary_label == 0:
        # entire grid is body — degenerate; return as-is
        logger.warning("voxelize_mesh: exterior corner voxel is occupied; "
                       "skipping flood-fill — check pad_mm")
        return occ
    occ = lab != boundary_label
    return binary_erosion(occ, iterations=dilate_voxels)


def voxelize_solid_for_mesh(
    mesh: trimesh.Trimesh, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    *, pitch: float, mn: np.ndarray,
) -> np.ndarray:
    """Convex-hull voxelisation restricted to the mesh's local bbox.

    A faster wrapper around :func:`voxelize_solid` that crops the test grid
    to the mesh bounding box before evaluating hull half-spaces. Returns a
    full-grid boolean array.
    """
    pts = np.asarray(mesh.vertices)
    if len(pts) < 4:
        return np.zeros(X.shape, dtype=bool)
    try:
        ch = ConvexHull(pts)
    except Exception as e:
        logger.debug("ConvexHull failed (%d pts): %s", len(pts), e)
        return np.zeros(X.shape, dtype=bool)
    b_mn = pts.min(0)
    b_mx = pts.max(0)
    shape = X.shape
    i0 = max(0, int((b_mn[0] - mn[0]) / pitch) - 1)
    i1 = min(shape[0], int((b_mx[0] - mn[0]) / pitch) + 2)
    j0 = max(0, int((b_mn[1] - mn[1]) / pitch) - 1)
    j1 = min(shape[1], int((b_mx[1] - mn[1]) / pitch) + 2)
    k0 = max(0, int((b_mn[2] - mn[2]) / pitch) - 1)
    k1 = min(shape[2], int((b_mx[2] - mn[2]) / pitch) + 2)
    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        return np.zeros(shape, dtype=bool)
    sub_x = X[i0:i1, j0:j1, k0:k1]
    sub_y = Y[i0:i1, j0:j1, k0:k1]
    sub_z = Z[i0:i1, j0:j1, k0:k1]
    sub_P = np.stack([sub_x.ravel(), sub_y.ravel(), sub_z.ravel()], axis=1)
    eqs = ch.equations
    inside = ((eqs[:, :3] @ sub_P.T + eqs[:, 3:4]) <= 1e-6).all(axis=0)
    out = np.zeros(shape, dtype=bool)
    out[i0:i1, j0:j1, k0:k1] = inside.reshape(sub_x.shape)
    return out


def keep_largest_component(occ: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (closing artefacts removal)."""
    lab, n = nd_label(occ)
    if n <= 1:
        return occ
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    biggest = int(sizes.argmax())
    return lab == biggest


def fill_internal_cavities(occ: np.ndarray) -> np.ndarray:
    """Seal all enclosed cavities (lung air, gut air) so the body is simply connected."""
    return binary_fill_holes(occ)
