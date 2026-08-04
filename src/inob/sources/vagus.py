"""Source-dipole sampling along the vagus nerve.

A single canonical implementation, used by both the local forward solver
(``inob.forward.solve``) and the cluster chunk worker (``inob.forward.chunk``).
"""
from __future__ import annotations

import logging

import numpy as np

from inob.config import source_tissue_labels
from inob.io.hdf5 import FemMesh

logger = logging.getLogger(__name__)


def vagus_sources(
    fem: FemMesh, tissue_label: str, *, spacing_mm: float,
) -> np.ndarray:
    """Sample one dipole position per ``spacing_mm`` of axial Z extent of a tissue.

    Steps: select tets with ``tissue_label`` → tet centroids → split by Z-slabs
    of width ``spacing_mm`` → mean centroid per slab. Returns ``(S, 3)`` mm.

    Raises:
        ValueError: if ``tissue_label`` is unknown or has no tets in ``fem``.
    """
    if tissue_label not in fem.tissue_labels:
        raise ValueError(
            f"tissue {tissue_label!r} not in FEM (have {list(fem.tissue_labels)})"
        )
    tissue_id = fem.label_to_id[tissue_label]
    mask = fem.tissue == tissue_id
    if not mask.any():
        raise ValueError(f"no tets with tissue id {tissue_id} ({tissue_label!r})")
    elems = fem.tets[mask]
    centroids = fem.nodes[elems].mean(axis=1)
    z_lo = float(centroids[:, 2].min())
    z_hi = float(centroids[:, 2].max())
    edges = np.arange(z_lo, z_hi + spacing_mm, spacing_mm)
    src: list[np.ndarray] = []
    for k in range(len(edges) - 1):
        sel = (centroids[:, 2] >= edges[k]) & (centroids[:, 2] < edges[k + 1])
        if sel.any():
            src.append(centroids[sel].mean(axis=0))
    pos = np.asarray(src, dtype=np.float64)
    logger.info(
        "%d dipole positions along %s (spacing %g mm, z=%g..%g)",
        len(pos), tissue_label, spacing_mm, z_lo, z_hi,
    )
    return pos


def _sample_one(fem: FemMesh, label: str, *, spacing_mm: float) -> np.ndarray:
    """Sample one tissue, dispatching muscle to its volume-fill sampler."""
    if label == "muscle":
        from inob.sources.muscle import muscle_sources
        return muscle_sources(fem, spacing_mm=spacing_mm)
    return vagus_sources(fem, label, spacing_mm=spacing_mm)


def sample_source_tissues(
    fem: FemMesh, source_tissue: str, *, spacing_mm: float,
) -> np.ndarray:
    """Sample dipoles across one or more comma-separated tissue labels.

    ``source_tissue`` may name a single tissue (``"vagus_left"``) or several
    (``"spinal_cord,vagus_left"``); positions from each are concatenated. This
    is what lets the cluster pipeline target vagus, spine, or both.

    ``muscle`` is sampled differently: it is a bulky bilateral tissue, so the
    Z-slab averaging used for thin midline nerves would place dipoles outside
    the muscle. It is volume-filled inside the tissue instead (see
    :func:`inob.sources.muscle.muscle_sources`).
    """
    labels = source_tissue_labels(source_tissue)
    if not labels:
        raise ValueError(f"source_tissue is empty: {source_tissue!r}")
    parts = [_sample_one(fem, lab, spacing_mm=spacing_mm) for lab in labels]
    if len(parts) == 1:
        return parts[0]
    pos = np.concatenate(parts, axis=0)
    logger.info("combined %d dipole positions across tissues %s", len(pos), labels)
    return pos


def _tets_containing(points: np.ndarray, fem: FemMesh, *, k: int = 64) -> np.ndarray:
    """Boolean mask: is each point inside some tetrahedron of ``fem``?

    Exact barycentric containment, restricted to the ``k`` tets whose centroids
    are nearest each point — an element that contains the point is necessarily
    among its nearest neighbours, so this is exact for any sane mesh while
    staying cheap enough to run per clicked source.
    """
    from scipy.spatial import cKDTree

    verts = fem.nodes[fem.tets]                     # (T, 4, 3)
    centroids = verts.mean(axis=1)
    tree = cKDTree(centroids)
    _, idx = tree.query(points, k=min(k, len(centroids)))
    # query returns (n_points,) when k == 1 and (n_points, k) otherwise; reshape
    # explicitly rather than atleast_2d, which would turn the k == 1 case into a
    # single row of n_points candidates and silently skip every point but one.
    idx = np.asarray(idx).reshape(len(points), -1)

    inside = np.zeros(len(points), dtype=bool)
    for i, cand in enumerate(idx):
        v = verts[cand]                             # (k, 4, 3)
        d = v[:, 3, :]
        # Columns a-d, b-d, c-d; barycentric coords of p relative to that basis.
        t = np.stack([v[:, 0] - d, v[:, 1] - d, v[:, 2] - d], axis=-1)  # (k,3,3)
        rhs = points[i] - d                                             # (k,3)
        try:
            lam = np.linalg.solve(t, rhs[..., None])[..., 0]             # (k,3)
        except np.linalg.LinAlgError:
            continue                                # degenerate tets → not inside
        full = np.concatenate([lam, 1.0 - lam.sum(axis=1, keepdims=True)], axis=1)
        # A small negative tolerance keeps points exactly on a face/edge inside.
        inside[i] = bool((full >= -1e-9).all(axis=1).any())
    return inside


def assert_sources_in_mesh(pos: np.ndarray, fem: FemMesh) -> None:
    """Raise a readable error for any source outside the FEM volume.

    DUNEuro's own failure for this is a bare C++ exception —
    ``Dune::Exception [findEntity:...kdtree.hh]: position ... not contained in
    mesh`` — which surfaces in the GUI as an opaque wall of text several minutes
    into a solve. A clicked point just off the anatomy is an ordinary mistake,
    so it deserves an ordinary message, raised before the solver starts.
    """
    inside = _tets_containing(pos, fem)
    if inside.all():
        return

    from scipy.spatial import cKDTree

    tree = cKDTree(fem.nodes)
    bad = np.flatnonzero(~inside)
    lines = []
    for i in bad:
        dist, j = tree.query(pos[i])
        near = fem.nodes[j]
        lines.append(
            f"  source {i + 1} at ({pos[i][0]:.1f}, {pos[i][1]:.1f}, "
            f"{pos[i][2]:.1f}) mm — {dist:.1f} mm outside; nearest point in the "
            f"model is ({near[0]:.1f}, {near[1]:.1f}, {near[2]:.1f})"
        )
    lo, hi = fem.nodes.min(axis=0), fem.nodes.max(axis=0)
    raise ValueError(
        f"{len(bad)} of {len(pos)} source(s) lie outside the FEM model, so the "
        "forward solve cannot evaluate them:\n" + "\n".join(lines) +
        f"\nThe model spans ({lo[0]:.0f}, {lo[1]:.0f}, {lo[2]:.0f}) to "
        f"({hi[0]:.0f}, {hi[1]:.0f}, {hi[2]:.0f}) mm. Move the source onto the "
        "target structure, or rebuild the mesh with that region included."
    )


def resolve_source_positions(cfg, fem: FemMesh) -> np.ndarray:
    """Dipole positions for the forward solve, ``(S, 3)`` mm.

    If ``cfg.forward.point_sources`` is set (e.g. points clicked in the GUI),
    those explicit positions are used verbatim; otherwise dipoles are sampled
    along ``cfg.forward.source_tissue`` (one or more comma-separated tissues)
    at ``cfg.forward.source_spacing_mm``.
    """
    if cfg.forward.point_sources:
        pos = np.asarray(cfg.forward.point_sources, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"point_sources must be (S, 3); got {pos.shape}")
        # Only explicit sources need this: sampled ones come from the mesh.
        assert_sources_in_mesh(pos, fem)
        logger.info("%d explicit point sources (overriding vagus sampling)", len(pos))
        return pos
    return sample_source_tissues(
        fem, cfg.forward.source_tissue, spacing_mm=cfg.forward.source_spacing_mm,
    )
