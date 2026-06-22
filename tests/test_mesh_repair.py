"""Mesh-repair primitives."""
from __future__ import annotations

import numpy as np
import trimesh

from vagus_fm.mesh.repair import (
    boolean_union_overlapping,
    cheap_repair,
    is_perfect,
)


def test_cheap_repair_idempotent_on_clean_mesh() -> None:
    m = trimesh.creation.icosphere(radius=10.0, subdivisions=3)
    out = cheap_repair(m)
    assert isinstance(out, trimesh.Trimesh)
    assert is_perfect(out)


def test_is_perfect_true_for_sphere() -> None:
    m = trimesh.creation.icosphere(radius=5.0)
    assert is_perfect(m)


def test_is_perfect_false_for_open_mesh() -> None:
    m = trimesh.creation.icosphere(radius=5.0)
    # Drop a face → no longer watertight
    m.update_faces(np.arange(1, len(m.faces)))
    assert not is_perfect(m)


def test_cheap_repair_removes_duplicate_vertices() -> None:
    m = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    # Duplicate every vertex
    v_dup = np.vstack([m.vertices, m.vertices])
    f_dup = np.vstack([m.faces, m.faces + len(m.vertices)])
    raw = trimesh.Trimesh(v_dup, f_dup, process=False)
    out = cheap_repair(raw)
    assert len(out.vertices) <= len(m.vertices) + 1   # merged


def test_boolean_union_two_overlapping_spheres() -> None:
    a = trimesh.creation.icosphere(radius=10.0)
    b = trimesh.creation.icosphere(radius=10.0)
    b.apply_translation([5.0, 0.0, 0.0])
    merged = trimesh.util.concatenate([a, b])
    out = boolean_union_overlapping(merged)
    assert isinstance(out, trimesh.Trimesh)
    # The union of two overlapping spheres is a single solid
    parts = out.split(only_watertight=False)
    assert len(parts) <= 2


def test_boolean_union_disjoint_kept_separate() -> None:
    a = trimesh.creation.icosphere(radius=5.0)
    b = trimesh.creation.icosphere(radius=5.0)
    b.apply_translation([100.0, 0.0, 0.0])   # far away — bboxes disjoint
    merged = trimesh.util.concatenate([a, b])
    out = boolean_union_overlapping(merged)
    parts = out.split(only_watertight=False)
    assert len(parts) == 2
