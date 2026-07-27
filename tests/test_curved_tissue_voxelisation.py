"""Voxelisation and repair of long curved tissues (the spinal cord case).

A convex hull is only a valid fill for convex bodies. The spinal cord follows
the cervical lordosis and thoracic kyphosis, so a hull chords across the
curvature: it over-fills the volume several-fold and drags the centroid off
the true axis — and the centroid is exactly where source dipoles get sampled.
These tests use a synthetic curved tube so they do not depend on the atlas.
"""
from __future__ import annotations

import numpy as np
import trimesh

from inob.fem.cgal_builder import VOXELISATION
from inob.mesh.repair import cheap_repair, drop_degenerate_components
from inob.mesh.shrinkwrap import occupancy_from_mesh
from inob.mesh.voxelize import voxelize_mesh, voxelize_solid_for_mesh


def _curved_tube(radius: float = 5.0, length: float = 400.0,
                 bow: float = 25.0, n: int = 120, seg: int = 16) -> trimesh.Trimesh:
    """An S-shaped tube: straight in z, bowed in y — a spinal cord in miniature.

    The double bend matters: it mimics the cervical lordosis and thoracic
    kyphosis, and it is what makes a convex hull so much worse than a single
    arc would suggest.

    Built from numpy directly rather than ``sweep_polygon`` so the test suite
    does not need shapely.
    """
    t = np.linspace(0.0, 1.0, n)
    centres = np.column_stack([np.zeros_like(t), bow * np.sin(2 * np.pi * t), t * length])

    tangents = np.gradient(centres, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    # Two normals per ring, orthogonal to the local tangent.
    ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(tangents, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(tangents, u)

    theta = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
    rings = (
        centres[:, None, :]
        + radius * (np.cos(theta)[None, :, None] * u[:, None, :]
                    + np.sin(theta)[None, :, None] * v[:, None, :])
    )
    verts = rings.reshape(-1, 3)

    faces = []
    for i in range(n - 1):
        for j in range(seg):
            a = i * seg + j
            b = i * seg + (j + 1) % seg
            c = (i + 1) * seg + (j + 1) % seg
            d = (i + 1) * seg + j
            faces.extend([[a, b, c], [a, c, d]])
    # Cap both ends so the tube is a closed volume.
    start_c, end_c = len(verts), len(verts) + 1
    verts = np.vstack([verts, centres[0], centres[-1]])
    for j in range(seg):
        faces.append([start_c, (j + 1) % seg, j])
        faces.append([end_c, (n - 1) * seg + j, (n - 1) * seg + (j + 1) % seg])

    return trimesh.Trimesh(verts, np.array(faces), process=True)


def _grid(mesh: trimesh.Trimesh, pitch: float, pad: float = 15.0):
    mn = mesh.bounds[0] - pad
    mx = mesh.bounds[1] + pad
    shape = tuple(((mx - mn) / pitch).astype(int) + 1)
    xs = mn[0] + (np.arange(shape[0]) + 0.5) * pitch
    ys = mn[1] + (np.arange(shape[1]) + 0.5) * pitch
    zs = mn[2] + (np.arange(shape[2]) + 0.5) * pitch
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return X, Y, Z, mn


def test_hull_overfills_a_curved_tube_far_worse_than_surface() -> None:
    """Documents why 'solid' is wrong here: the failure this guards against.

    Measured against the tube's true volume, so the claim is about physical
    accuracy rather than one voxeliser relative to the other.
    """
    tube = _curved_tube()
    pitch = 3.0
    voxel_mm3 = pitch ** 3
    true_cm3 = tube.volume / 1000.0
    X, Y, Z, mn = _grid(tube, pitch)

    hull_cm3 = voxelize_solid_for_mesh(
        tube, X, Y, Z, pitch=pitch, mn=mn).sum() * voxel_mm3 / 1000.0
    surf_cm3 = voxelize_mesh(
        tube, X, pitch=pitch, mn=mn, closing_mm=2.0).sum() * voxel_mm3 / 1000.0

    # The hull inflates the tissue several-fold...
    assert hull_cm3 > 3.0 * true_cm3
    # ...while the surface fill stays within the voxel-quantisation floor for a
    # structure only ~3 voxels across.
    assert surf_cm3 < 2.5 * true_cm3
    assert surf_cm3 < 0.6 * hull_cm3


def test_surface_voxelisation_tracks_the_curve() -> None:
    """The hull's centroid leaves the tube; the surface fill stays on it."""
    tube = _curved_tube()
    pitch = 3.0
    X, Y, Z, mn = _grid(tube, pitch)
    hull = voxelize_solid_for_mesh(tube, X, Y, Z, pitch=pitch, mn=mn)
    surf = voxelize_mesh(tube, X, pitch=pitch, mn=mn, closing_mm=2.0)

    # Slice at the bow's extremum, where the hull's chord is furthest from the
    # tube. (The mid-length slice is the S-curve's zero crossing, where hull
    # and truth coincide and the defect is invisible.)
    V = np.asarray(tube.vertices)
    z_at_bow = V[np.argmax(np.abs(V[:, 1])), 2]
    k = int(np.argmin(np.abs(Z[0, 0, :] - z_at_bow)))

    in_slice = np.abs(V[:, 2] - Z[0, 0, k]) < pitch
    true_y = float(V[in_slice][:, 1].mean())
    hull_y = float(Y[:, :, k][hull[:, :, k]].mean())
    surf_y = float(Y[:, :, k][surf[:, :, k]].mean())

    # The hull pulls the centroid off the true axis by more than the tube's
    # own radius; the surface fill stays within a voxel of it.
    assert abs(hull_y - true_y) > 5.0
    assert abs(surf_y - true_y) < 3.0
    assert abs(surf_y - true_y) < abs(hull_y - true_y)


def test_spinal_cord_is_configured_for_surface_voxelisation() -> None:
    assert VOXELISATION["spinal_cord"]["method"] == "surface"


def test_every_voxelisation_entry_declares_a_method() -> None:
    for label, opts in VOXELISATION.items():
        assert opts.get("method") in ("solid", "surface"), label


# ── repair guards ──────────────────────────────────────────────────────────

def test_drop_degenerate_components_removes_stray_triangles() -> None:
    """The shipped cord STL carries two isolated single triangles; they break
    boolean union and mislead pymeshfix into keeping a fragment."""
    body = trimesh.creation.icosphere(radius=10.0, subdivisions=2)
    stray = trimesh.Trimesh(
        vertices=np.array([[100.0, 0, 0], [100.2, 0, 0], [100.0, 0.1, 0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    merged = trimesh.util.concatenate([body, stray])
    assert len(merged.split(only_watertight=False)) == 2

    cleaned = drop_degenerate_components(cheap_repair(merged))
    assert len(cleaned.split(only_watertight=False)) == 1
    assert len(cleaned.faces) == len(body.faces)


def test_drop_degenerate_components_is_a_noop_on_clean_input() -> None:
    body = trimesh.creation.icosphere(radius=10.0, subdivisions=2)
    assert drop_degenerate_components(body) is body


def test_drop_degenerate_components_keeps_multiple_real_bodies() -> None:
    """Bone is legitimately many separate bodies — none may be dropped."""
    a = trimesh.creation.icosphere(radius=5.0, subdivisions=2)
    b = trimesh.creation.icosphere(radius=5.0, subdivisions=2)
    b.apply_translation([50.0, 0, 0])
    merged = trimesh.util.concatenate([a, b])
    assert len(drop_degenerate_components(merged).split(only_watertight=False)) == 2


# ── reproducibility ────────────────────────────────────────────────────────

def test_surface_sampling_is_seeded() -> None:
    """Unseeded sampling draws from the global RNG, so adding a compartment
    would perturb the geometry of every compartment built after it."""
    # An open tube (caps removed) is not watertight, so occupancy_from_mesh
    # takes the random surface-splat path rather than the deterministic
    # ray-trace fill. That is the path whose seeding we care about.
    tube = _curved_tube()
    open_tube = trimesh.Trimesh(tube.vertices, tube.faces[: -2 * 16], process=True)
    assert not open_tube.is_watertight
    tube = open_tube
    a, _ = occupancy_from_mesh(tube, pitch=2.0, n_samples=20_000, seed=7)
    b, _ = occupancy_from_mesh(tube, pitch=2.0, n_samples=20_000, seed=7)
    c, _ = occupancy_from_mesh(tube, pitch=2.0, n_samples=20_000, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_compartment_seeds_are_independent_of_build_order() -> None:
    from inob.geometry.builder import compartment_seed
    # Same name, same seed, regardless of what else is in the config.
    assert compartment_seed("mesh_spinal_cord", 0) == compartment_seed("mesh_spinal_cord", 0)
    assert compartment_seed("mesh_spinal_cord", 0) != compartment_seed("mesh_bone", 0)
    assert compartment_seed("mesh_bone", 0) != compartment_seed("mesh_bone", 1)
