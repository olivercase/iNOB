"""Triaxial OPM sensor array placement over the skin surface.

Cylindrical raycasting around the head-foot (Z) axis, with a stand-off
``depth`` along the outward surface normal. At each hit position we emit
three coils oriented (Radial, Tangent-1, Tangent-2). The output layout
matches the existing FieldTrip-style ``grad`` group — all R first, then
all T1, then all T2 — so downstream code is unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from vagus_fm.config import Config
from vagus_fm.io.hdf5 import SensorArray, save_sensors, validate_sensors
from vagus_fm.io.stl import load_stl

logger = logging.getLogger(__name__)


def cylindrical_raycast(
    mesh: trimesh.Trimesh,
    *,
    resolution_mm: float,
    z_min: float,
    z_max: float,
    cylinder_radius_factor: float = 0.75,
    angular_margin_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample skin-surface positions + outward normals via radial inward rays.

    Returns ``(positions, normals)`` in mesh coordinates (mm). Glancing hits
    where the surface is nearly parallel to the ray are filtered out.
    """
    bounds = mesh.bounds
    cx = 0.5 * (bounds[0, 0] + bounds[1, 0])
    cy = 0.5 * (bounds[0, 1] + bounds[1, 1])
    x_range = bounds[1, 0] - bounds[0, 0]
    y_range = bounds[1, 1] - bounds[0, 1]
    radius = cylinder_radius_factor * max(x_range, y_range)

    zs = np.arange(z_min, z_max + 0.5 * resolution_mm, resolution_mm)
    n_theta = max(8, round(2 * np.pi * radius / resolution_mm))
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    Z, TH = np.meshgrid(zs, thetas, indexing="ij")
    Z = Z.ravel()
    TH = TH.ravel()
    cos_t = np.cos(TH)
    sin_t = np.sin(TH)
    origins = np.column_stack([cx + radius * cos_t, cy + radius * sin_t, Z])
    directions = np.column_stack([-cos_t, -sin_t, np.zeros_like(TH)])

    logger.info(
        "rays: %d (%d z-slices x %d angles, radius=%.0f mm)",
        len(origins), len(zs), n_theta, radius,
    )

    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    else:
        logger.info("embree not available — falling back to slow ray.ray_triangle")
        intersector = mesh.ray
    locations, ray_idx, tri_idx = intersector.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True,
    )
    if len(locations) == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    dists = np.linalg.norm(locations - origins[ray_idx], axis=1)
    order = np.lexsort((dists, ray_idx))
    locations = locations[order]
    ray_idx_s = ray_idx[order]
    tri_idx_s = tri_idx[order]
    _, first = np.unique(ray_idx_s, return_index=True)
    locations = locations[first]
    rays_hit = ray_idx_s[first]
    tris_hit = tri_idx_s[first]

    normals = mesh.face_normals[tris_hit]
    ray_dirs_hit = directions[rays_hit]
    flip = np.einsum("ij,ij->i", normals, ray_dirs_hit) > 0
    normals = np.where(flip[:, None], -normals, normals)
    cos_ang = -np.einsum("ij,ij->i", normals, ray_dirs_hit)
    angle_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
    keep = angle_deg < angular_margin_deg
    return locations[keep], normals[keep]


def build_triaxial(positions: np.ndarray, normals: np.ndarray) -> SensorArray:
    """Build a triaxial :class:`SensorArray` from positions + outward normals.

    Output channel order: all N radial, then all N tangent-1, then all N
    tangent-2. Tangent vectors come from the SVD null space of the normal.
    """
    n = len(positions)
    coilpos = np.tile(positions, (3, 1))
    coilori = np.zeros((3 * n, 3))
    labels: list[str] = []

    for i in range(n):
        nrm = normals[i]
        nn = float(np.linalg.norm(nrm))
        if nn > 1e-12:
            nrm = nrm / nn
        else:
            nrm = np.array([0.0, 0.0, 1.0])
        _, _, vh = np.linalg.svd(nrm.reshape(1, 3))
        coilori[i] = nrm
        coilori[i + n] = vh[1]
        coilori[i + 2 * n] = vh[2]
        labels.append(f"mag-{i + 1:04d}-R")
    for i in range(n):
        labels.append(f"mag-{i + 1:04d}-T1")
    for i in range(n):
        labels.append(f"mag-{i + 1:04d}-T2")

    coilori /= np.linalg.norm(coilori, axis=1, keepdims=True)
    return SensorArray(
        coilpos=coilpos,
        coilori=coilori,
        labels=tuple(labels),
        chantype=tuple(["megmag"] * (3 * n)),
        chanunit=tuple(["T"] * (3 * n)),
        unit="mm",
    )


def generate_sensor_array(
    cfg: Config,
    *,
    z_min: float | None = None,
    z_max: float | None = None,
) -> Path:
    """Build the full triaxial sensor array per ``cfg`` and save to HDF5."""
    skin_path = cfg.data.torso_skin
    logger.info("Loading skin: %s", skin_path)
    mesh = load_stl(skin_path)
    logger.info("  skin: %d V / %d F", len(mesh.vertices), len(mesh.faces))

    z_lo, z_hi = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    if z_min is None:
        z_min = z_lo + cfg.sensors.z_crop_low_factor * (z_hi - z_lo)
    if z_max is None:
        z_max = z_hi
    logger.info(
        "Head-foot crop: Z in [%.1f, %.1f] mm (bbox Z in [%.1f, %.1f])",
        z_min, z_max, z_lo, z_hi,
    )

    positions, normals = cylindrical_raycast(
        mesh,
        resolution_mm=cfg.sensors.resolution_mm,
        z_min=z_min,
        z_max=z_max,
        cylinder_radius_factor=cfg.sensors.cylinder_radius_factor,
        angular_margin_deg=cfg.sensors.angular_margin_deg,
    )
    logger.info("hit %d surface points", len(positions))
    if len(positions) == 0:
        raise RuntimeError(
            "No surface hits — check sensors.z_crop_low_factor / mesh bounds."
        )

    positions_offset = positions + normals * cfg.sensors.depth_mm
    array = build_triaxial(positions_offset, normals)
    validate_sensors(array)

    out = cfg.outputs.sensors_mat
    save_sensors(out, array)
    logger.info("[saved] %s (%d channels)", out, len(array.coilpos))
    return out
