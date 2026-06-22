"""Surface-mesh repair primitives.

Layered watertightening pipeline used by :mod:`vagus_fm.geometry.builder`:

  1. Cheap trimesh repair (merge verts, drop degenerates, fix normals).
  2. Boolean union of spatially overlapping connected components.
  3. (Voxel shrinkwrap — see :mod:`vagus_fm.mesh.shrinkwrap`.)
  4. pymeshfix as a final hole-closer.

Each helper takes a :class:`trimesh.Trimesh` and returns a new (or in-place)
:class:`trimesh.Trimesh`. None of these functions raise on degenerate input;
they log a warning and return the best they can.
"""
from __future__ import annotations

import logging

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def cheap_repair(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge dup verts, kill degenerates, fix winding/normals.

    Cheap and idempotent — safe to call repeatedly between heavier passes.
    """
    m = trimesh.Trimesh(
        np.asarray(m.vertices, dtype=np.float64),
        np.asarray(m.faces, dtype=np.int64),
        process=True,
    )
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    try:
        m.fix_normals()
    except Exception as e:
        logger.debug("fix_normals failed (continuing): %s", e)
    return m


def is_perfect(m: trimesh.Trimesh) -> bool:
    """True iff watertight, winding-consistent, and Euler==2 (genus 0)."""
    try:
        return (
            bool(m.is_watertight)
            and bool(m.is_winding_consistent)
            and int(m.euler_number) == 2
        )
    except Exception:
        return False


def boolean_union_overlapping(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Boolean-union spatially-overlapping connected components.

    Splits the mesh into components, groups those whose bounding boxes
    intersect (transitively), and runs ``trimesh.boolean.union`` on each
    group. Non-overlapping components are left as separate shells in the
    returned mesh. Useful as a stepping stone before shrinkwrap.
    """
    parts = m.split(only_watertight=False)
    if len(parts) <= 1:
        return m

    n = len(parts)
    overlap_groups: list[set[int]] = []
    used: set[int] = set()
    for i in range(n):
        if i in used:
            continue
        group = {i}
        changed = True
        while changed:
            changed = False
            grp_lo = np.min([parts[k].bounds[0] for k in group], axis=0)
            grp_hi = np.max([parts[k].bounds[1] for k in group], axis=0)
            for j in range(n):
                if j in group or j in used:
                    continue
                bj = parts[j].bounds
                if np.all(bj[1] >= grp_lo) and np.all(bj[0] <= grp_hi):
                    group.add(j)
                    changed = True
        used.update(group)
        overlap_groups.append(group)

    pieces: list[trimesh.Trimesh] = []
    for grp in overlap_groups:
        if len(grp) == 1:
            pieces.append(parts[next(iter(grp))])
            continue
        try:
            merged = trimesh.boolean.union([parts[k] for k in sorted(grp)])
        except Exception as e:
            logger.warning("boolean union failed on group of %d: %s; keeping unmerged",
                           len(grp), e)
            pieces.extend(parts[k] for k in grp)
            continue
        if merged is None or len(merged.faces) == 0:
            logger.warning("boolean union produced empty mesh; keeping unmerged")
            pieces.extend(parts[k] for k in grp)
        else:
            pieces.append(merged)

    return pieces[0] if len(pieces) == 1 else trimesh.util.concatenate(pieces)


def pymeshfix_pass(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Final close-everything-up pass via pymeshfix.

    Raises :class:`RuntimeError` if pymeshfix returns an empty mesh — caller
    is expected to handle the failure rather than silently degrading.
    """
    import pymeshfix

    mf = pymeshfix.MeshFix(
        np.asarray(m.vertices, dtype=np.float64),
        np.asarray(m.faces, dtype=np.int32),
    )
    try:
        mf.repair(verbose=False)
    except TypeError:
        mf.repair()
    v_rep = getattr(mf, "points", getattr(mf, "v", None))
    f_rep = getattr(mf, "faces", getattr(mf, "f", None))
    if v_rep is None or f_rep is None or len(f_rep) == 0:
        raise RuntimeError("pymeshfix returned an empty mesh")
    out = trimesh.Trimesh(
        np.asarray(v_rep, dtype=np.float64),
        np.asarray(f_rep, dtype=np.int64),
        process=True,
    )
    return cheap_repair(out)
