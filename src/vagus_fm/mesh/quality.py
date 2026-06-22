"""Tetrahedral mesh quality metrics + validation.

We use iso2mesh's ``meshquality`` when available — it returns the per-element
quality factor (1.0 = regular tetrahedron, ~0 = degenerate). On systems without
iso2mesh we compute an equivalent Joe-Liu metric ourselves.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityStats:
    n_tets: int
    min: float
    p1: float
    p5: float
    mean: float
    max: float

    def as_dict(self) -> dict[str, float]:
        return {"n_tets": self.n_tets, "min": self.min, "p1": self.p1,
                "p5": self.p5, "mean": self.mean, "max": self.max}


class MeshQualityError(ValueError):
    """Raised when a mesh fails the configured quality gates."""


def _joe_liu_quality(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Joe-Liu mean-ratio quality, normalised so a regular tet → 1.

    q = 12 * (3V)^(2/3) / sum(edge_length^2)
    """
    p = nodes[tets]
    a, b, c, d = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    vol = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0
    abs_vol = np.abs(vol)
    edges = np.array([
        np.einsum("ij,ij->i", b - a, b - a),
        np.einsum("ij,ij->i", c - a, c - a),
        np.einsum("ij,ij->i", d - a, d - a),
        np.einsum("ij,ij->i", c - b, c - b),
        np.einsum("ij,ij->i", d - b, d - b),
        np.einsum("ij,ij->i", d - c, d - c),
    ]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 12.0 * np.power(3.0 * abs_vol, 2.0 / 3.0) / edges
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    return q


def compute_quality(nodes: np.ndarray, tets: np.ndarray) -> QualityStats:
    """Return per-element quality stats for a tetrahedral mesh.

    Tries ``iso2mesh.meshquality`` (1-indexed elements expected) and falls back
    to the Joe-Liu metric when iso2mesh is unavailable or raises.
    """
    if len(tets) == 0:
        return QualityStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    q: np.ndarray | None = None
    try:
        import iso2mesh as im
        q = np.asarray(im.meshquality(nodes, np.asarray(tets) + 1)).ravel()
    except Exception as e:
        logger.debug("iso2mesh.meshquality unavailable (%s); using Joe-Liu fallback", e)
        q = _joe_liu_quality(nodes, tets)
    return QualityStats(
        n_tets=len(q),
        min=float(q.min()),
        p1=float(np.percentile(q, 1)),
        p5=float(np.percentile(q, 5)),
        mean=float(q.mean()),
        max=float(q.max()),
    )


def assert_mesh_ok(stats: QualityStats, *, min_quality: float = 0.05) -> None:
    """Raise :class:`MeshQualityError` if min quality below threshold."""
    if stats.n_tets and stats.min < min_quality:
        raise MeshQualityError(
            f"min mesh quality {stats.min:.4f} < threshold {min_quality:.4f} "
            f"(p1={stats.p1:.4f}, p5={stats.p5:.4f}, mean={stats.mean:.4f})"
        )


def assert_units_mm(nodes: np.ndarray, *, lo_mm: float = 1.0, hi_mm: float = 10_000.0) -> None:
    """Reject node arrays whose bbox extent is implausible for millimetres."""
    if len(nodes) == 0:
        return
    extent = float(np.ptp(nodes, axis=0).max())
    if not (lo_mm <= extent <= hi_mm):
        raise MeshQualityError(
            f"node bbox extent {extent:.3g} implausible for mm "
            f"(expected {lo_mm}..{hi_mm} mm)"
        )
