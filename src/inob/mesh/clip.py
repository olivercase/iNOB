"""Clip meshes to a region of interest, leaving them watertight.

A raw plane cut opens the mesh: the boundary loop is a hole, so the result is
neither watertight nor a valid FEM domain. Every function here caps the cut,
keeps the largest resulting body, and reports the topology it produced, so a
clipped compartment is still a closed conductor.

Why clip at all: the BodyParts3D skin is a whole body (z = -78..1641 mm). Meshing
the legs to model a heart source costs elements and buys nothing. Clipping to a
head+thorax band keeps the volume conductor continuous from the sensors down past
the heart, which is the part that carries the cardiac field artefact.
"""
from __future__ import annotations

import logging

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

__all__ = ["keep_largest", "clip_capped", "clip_band", "ClipReport"]


class ClipReport(dict):
    """Topology of a clip, carried alongside the mesh for logging and tests."""

    @property
    def ok(self) -> bool:
        return bool(self["watertight"] and self["winding_consistent"])


def keep_largest(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """The largest connected body by volume (falls back to face count)."""
    bodies = m.split(only_watertight=False)
    if len(bodies) <= 1:
        return m
    def size(b: trimesh.Trimesh) -> float:
        return abs(b.volume) if b.is_watertight else float(len(b.faces))
    best = max(bodies, key=size)
    logger.debug("keep_largest: %d bodies -> %d faces", len(bodies), len(best.faces))
    return best


def clip_capped(
    m: trimesh.Trimesh,
    origin,
    normal,
    *,
    label: str = "mesh",
    largest_only: bool = True,
) -> trimesh.Trimesh:
    """Keep the half of ``m`` on the ``normal`` side of the plane, capped.

    ``normal`` points at the half that is KEPT. Returns an empty mesh if the
    plane misses the geometry entirely, so callers can drop a compartment that
    falls outside the clip band rather than crash.
    """
    origin = np.asarray(origin, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    out = m.slice_plane(origin, normal, cap=True)
    if out is None or len(out.faces) == 0:
        logger.warning("[clip] %s: plane removed everything", label)
        return trimesh.Trimesh()
    if largest_only:
        out = keep_largest(out)
    out.remove_unreferenced_vertices()
    logger.info(
        "[clip] %s: %d->%d F  watertight=%s euler=%d",
        label, len(m.faces), len(out.faces), out.is_watertight, out.euler_number,
    )
    return out


def clip_band(
    m: trimesh.Trimesh,
    *,
    lo: float | None = None,
    hi: float | None = None,
    axis: int = 2,
    label: str = "mesh",
    largest_only: bool = True,
) -> trimesh.Trimesh:
    """Clip to ``lo <= x[axis] <= hi``, capping both cuts.

    Either bound may be None (open on that side). Cuts are applied one plane at
    a time and capped in turn — capping both at once is not well defined when
    the two cuts share a body.
    """
    out = m
    e = np.zeros(3)
    e[axis] = 1.0
    if lo is not None:
        out = clip_capped(out, e * lo, e, label=f"{label}[>{lo:g}]",
                          largest_only=largest_only)
        if len(out.faces) == 0:
            return out
    if hi is not None:
        out = clip_capped(out, e * hi, -e, label=f"{label}[<{hi:g}]",
                          largest_only=largest_only)
    return out


def report(m: trimesh.Trimesh) -> ClipReport:
    """Topology summary of a clipped mesh."""
    wt = bool(m.is_watertight)
    return ClipReport(
        vertices=int(len(m.vertices)),
        faces=int(len(m.faces)),
        bodies=int(len(m.split(only_watertight=False))),
        watertight=wt,
        winding_consistent=bool(m.is_winding_consistent),
        euler=int(m.euler_number),
        volume_mm3=float(m.volume) if wt else float("nan"),
    )
