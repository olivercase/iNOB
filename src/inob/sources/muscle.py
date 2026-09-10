"""Source-dipole sampling for skeletal muscle activity.

Unlike the vagus/spinal cord — thin, near-midline longitudinal structures where
one dipole per axial Z-slab is a faithful model — muscle is a set of bulky,
*bilateral* bodies filling the neck. Averaging tet centroids per Z-slab (as
:func:`inob.sources.vagus.vagus_sources` does) would collapse the left and right
muscle groups into a single point on the body midline, i.e. *outside* any muscle.

Instead we volume-fill: keep one dipole per occupied voxel of a regular grid at
``spacing_mm``, each taken from a real muscle tet centroid so every source is
guaranteed to sit inside muscle tissue. Each source is oriented along the long
axis of the individual muscle it belongs to (a fibre-direction proxy — neck
strap/scalene muscles run roughly along their long axis), computed per connected
component of the muscle tissue.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from inob.io.hdf5 import FemMesh

logger = logging.getLogger(__name__)


def _muscle_tet_centroids(fem: FemMesh, tissue_label: str = "muscle") -> np.ndarray:
    """Centroids ``(M, 3)`` mm of every tet tagged ``tissue_label``."""
    if tissue_label not in fem.tissue_labels:
        raise ValueError(f"tissue {tissue_label!r} not in FEM (have {list(fem.tissue_labels)})")
    tissue_id = fem.label_to_id[tissue_label]
    mask = fem.tissue == tissue_id
    if not mask.any():
        raise ValueError(f"no tets with tissue id {tissue_id} ({tissue_label!r})")
    return fem.nodes[fem.tets[mask]].mean(axis=1)


def muscle_sources(
    fem: FemMesh,
    *,
    spacing_mm: float,
    tissue_label: str = "muscle",
) -> np.ndarray:
    """Volume-fill dipole positions inside the muscle tissue, ``(S, 3)`` mm.

    A regular grid at ``spacing_mm`` is laid over the muscle bounding box; for
    each occupied voxel we keep the muscle tet centroid nearest the voxel
    centre. Every returned position is therefore a real muscle interior point,
    and the density is ~uniform at ``spacing_mm``. Deterministic ordering (by
    voxel index) so cluster chunks agree.
    """
    centroids = _muscle_tet_centroids(fem, tissue_label)
    origin = centroids.min(axis=0)
    vox = np.floor((centroids - origin) / spacing_mm).astype(np.int64)  # (M, 3)

    # Group centroids by voxel; pick the one closest to its voxel centre.
    order = np.lexsort((vox[:, 2], vox[:, 1], vox[:, 0]))
    vox_s = vox[order]
    cent_s = centroids[order]
    boundaries = np.any(np.diff(vox_s, axis=0) != 0, axis=1)
    starts = np.concatenate(([0], np.nonzero(boundaries)[0] + 1))
    ends = np.concatenate((starts[1:], [len(vox_s)]))

    picked: list[np.ndarray] = []
    for s, e in zip(starts, ends, strict=True):
        vc = origin + (vox_s[s] + 0.5) * spacing_mm  # voxel centre
        group = cent_s[s:e]
        picked.append(group[np.argmin(np.sum((group - vc) ** 2, axis=1))])
    pos = np.asarray(picked, dtype=np.float64)
    logger.info(
        "%d muscle dipole positions (volume-fill, spacing %g mm, X=%g..%g Y=%g..%g Z=%g..%g)",
        len(pos),
        spacing_mm,
        pos[:, 0].min(),
        pos[:, 0].max(),
        pos[:, 1].min(),
        pos[:, 1].max(),
        pos[:, 2].min(),
        pos[:, 2].max(),
    )
    return pos


def _muscle_components(fem: FemMesh, tissue_label: str = "muscle"):
    """Split the muscle tissue into connected components (individual muscles).

    Returns ``(labels, centroids)`` where ``labels[i]`` is the component id of
    muscle tet ``i`` and ``centroids`` are that tet's centroid. Two tets are
    connected when they share a mesh node, so each anatomical muscle body
    (left/right sternocleidomastoid, scalenes, …) becomes one component.
    """
    tissue_id = fem.label_to_id[tissue_label]
    mask = fem.tissue == tissue_id
    tets = fem.tets[mask]  # (M, 4) node indices
    centroids = fem.nodes[tets].mean(axis=1)
    n_tets = len(tets)

    # Tet↔node incidence → tet-tet adjacency via A A^T (share a node ⇒ linked).
    rows = np.repeat(np.arange(n_tets), 4)
    cols = tets.reshape(-1)
    incidence = coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)),
        shape=(n_tets, int(fem.nodes.shape[0])),
    ).tocsr()
    adj = incidence @ incidence.T
    n_comp, labels = connected_components(adj, directed=False)
    logger.info("muscle split into %d connected components (individual muscles)", n_comp)
    return labels, centroids


def _principal_axis(pts: np.ndarray) -> np.ndarray:
    """Unit long axis of a point cloud (largest-eigenvalue covariance axis)."""
    cov = np.cov((pts - pts.mean(axis=0)).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    if axis[2] < 0:  # consistent sign (superior)
        axis = -axis
    return axis / max(np.linalg.norm(axis), 1e-12)


def _muscle_stl_axes(muscle_dir) -> tuple[np.ndarray, np.ndarray]:
    """Per-muscle fibre long-axis and centroid from the muscle STLs.

    Returns ``(axes (K, 3), centres (K, 3))`` — one unit long-axis (fibre
    proxy) and one centroid per muscle STL in ``muscle_dir``. Shared by source
    orientation and the anisotropic-conductivity tensor builder so both use the
    identical per-muscle fibre directions. CGAL meshing fuses the muscle bodies
    into one FEM region, so the individual STLs — not the FEM components — are
    the only place per-muscle orientation survives.
    """
    from pathlib import Path

    import trimesh

    axes: list[np.ndarray] = []
    centres: list[np.ndarray] = []
    for stl in sorted(Path(muscle_dir).glob("*.stl")):
        verts = np.asarray(trimesh.load(stl, process=False).vertices)
        axes.append(_principal_axis(verts))
        centres.append(verts.mean(axis=0))
    if not axes:
        raise ValueError(f"no muscle STLs in {muscle_dir}")
    logger.info("per-muscle fibre axes from %d STLs in %s", len(axes), muscle_dir)
    return np.asarray(axes), np.asarray(centres)


def muscle_tet_fibre_axes(
    fem: FemMesh,
    *,
    muscle_dir,
    tissue_label: str = "muscle",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fibre axis of every muscle tet, for anisotropic conductivity.

    Returns ``(group (M,), axes (K, 3), mask (n_tets,))`` where ``mask`` selects
    the muscle tets (in FEM tet order), ``group[i]`` is the nearest-STL index of
    the i-th muscle tet, and ``axes[k]`` is muscle-STL ``k``'s unit fibre axis.
    A muscle tet's fibre direction is thus ``axes[group[i]]``. Assignment is by
    nearest STL centroid — the same rule as :func:`muscle_source_orientations`,
    so source dipoles and the conductivity tensor share one fibre field.
    """
    tissue_id = fem.label_to_id[tissue_label]
    mask = fem.tissue == tissue_id
    if not mask.any():
        raise ValueError(f"no tets with tissue id {tissue_id} ({tissue_label!r})")
    centroids = fem.nodes[fem.tets[mask]].mean(axis=1)  # (M, 3)
    axes, centres = _muscle_stl_axes(muscle_dir)
    # nearest STL centre for each muscle tet
    d2 = np.sum((centroids[:, None, :] - centres[None, :, :]) ** 2, axis=2)  # (M, K)
    group = np.argmin(d2, axis=1).astype(np.int64)
    return group, axes, mask


def _nearest_index(positions: np.ndarray, centres_arr: np.ndarray) -> np.ndarray:
    """Index of the nearest row of ``centres_arr`` for every position, ``(S,)``."""
    idx = np.empty(len(positions), dtype=np.int64)
    for i, p in enumerate(positions):
        idx[i] = np.argmin(np.sum((centres_arr - p) ** 2, axis=1))
    return idx


def muscle_source_stl_assignment(positions: np.ndarray, *, muscle_dir) -> np.ndarray:
    """Nearest-muscle-STL index (into ``sorted(Path(muscle_dir).glob("*.stl"))``)
    for every source position, ``(S,)``.

    The same nearest-STL rule :func:`muscle_source_orientations` uses to pick
    each source's fibre axis, exposed on its own so a caller (e.g. a figure
    that colours sources by which individual muscle they belong to) can use
    the identical per-source assignment rather than re-deriving it.
    """
    _axes_arr, centres_arr = _muscle_stl_axes(muscle_dir)
    return _nearest_index(positions, centres_arr)


def muscle_source_orientations(
    fem: FemMesh,
    positions: np.ndarray,
    *,
    tissue_label: str = "muscle",
    muscle_dir=None,
) -> np.ndarray:
    """Unit long-axis (fibre-direction proxy) per source, ``(S, 3)``.

    Fibre direction of each neck muscle is approximated by its long axis (the
    principal axis of the muscle's surface points) — a good proxy for the strap
    and scalene muscles, which run roughly along their length.

    When ``muscle_dir`` is given, each individual muscle STL supplies its own
    axis and every source is oriented by the *nearest* muscle STL. This is the
    faithful per-muscle result. Without it we fall back to the FEM tissue's
    connected components — but CGAL meshing tends to fuse the muscle bodies into
    one region, collapsing this to a single global axis, so prefer ``muscle_dir``.
    Signs are consistent (positive Z) so neighbouring dipoles don't cancel.
    """
    if muscle_dir is not None:
        axes_arr, centres_arr = _muscle_stl_axes(muscle_dir)
        nearest = _nearest_index(positions, centres_arr)
    else:
        labels, centroids = _muscle_components(fem, tissue_label)
        comp_ids = np.unique(labels)
        axes_arr = np.array([_principal_axis(centroids[labels == c]) for c in comp_ids])
        centres_arr = np.array([centroids[labels == c].mean(axis=0) for c in comp_ids])
        nearest = _nearest_index(positions, centres_arr)

    return axes_arr[nearest]
