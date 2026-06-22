"""Source-dipole sampling along the vagus nerve.

A single canonical implementation, used by both the local forward solver
(``inob.forward.solve``) and the cluster chunk worker
(``inob.forward.chunk``). Replaces three separate copies in the legacy
scripts (``run_fem_duneuro.py``, ``cluster/run_chunk.py``).
"""
from __future__ import annotations

import logging

import numpy as np

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


def resolve_source_positions(cfg, fem: FemMesh) -> np.ndarray:
    """Dipole positions for the forward solve, ``(S, 3)`` mm.

    If ``cfg.forward.point_sources`` is set (e.g. points clicked in the GUI),
    those explicit positions are used verbatim; otherwise dipoles are sampled
    along ``cfg.forward.source_tissue`` at ``cfg.forward.source_spacing_mm``.
    """
    if cfg.forward.point_sources:
        pos = np.asarray(cfg.forward.point_sources, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"point_sources must be (S, 3); got {pos.shape}")
        logger.info("%d explicit point sources (overriding vagus sampling)", len(pos))
        return pos
    return vagus_sources(
        fem, cfg.forward.source_tissue, spacing_mm=cfg.forward.source_spacing_mm,
    )
