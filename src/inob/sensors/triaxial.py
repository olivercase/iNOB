"""OPM sensor array placement over the skin surface.

Cylindrical raycasting around the head-foot (Z) axis, with a stand-off
``depth`` along the outward surface normal. At each hit position we emit
three coils oriented (Radial, Tangent-1, Tangent-2). The output layout
matches the existing FieldTrip-style ``grad`` group — all R first, then
all T1, then all T2 — so downstream code is unchanged.

The two tangents are built around one fixed reference axis rather than
per-sensor, so that T1 means the same thing at every position; see
:func:`tangent_frame`. ``sensors.axes: radial`` keeps the R coil alone
(:func:`build_radial`), and ``sensors.from_file`` replaces placement with an
imported array (:mod:`inob.sensors.imported`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from inob.config import DEFAULT_TANGENT_BASIS, Config
from inob.io.hdf5 import SensorArray, save_sensors, validate_sensors
from inob.io.stl import load_stl
from inob.sensors.imported import (
    FRAME_NOTE,
    import_magnetometers,
    load_skin_if_present,
    read_array_file,
    resolve_array_file,
)

logger = logging.getLogger(__name__)

#: Superior axis (+Z; see docs/COORDINATES.md). The tangential frame is built
#: around it, which makes T1 horizontal and T2 head-foot at every sensor whose
#: normal is not parallel to it.
_UP = np.array([0.0, 0.0, 1.0])

#: Stands in for :data:`_UP` where the normal is parallel to it and the cross
#: product is degenerate — the crown of the head, the soles of the feet.
#: Posterior (+Y); any axis off :data:`_UP` would serve.
_UP_FALLBACK = np.array([0.0, 1.0, 0.0])

#: |n . up| above which the global-up frame is too ill-conditioned to use.
#: No tangential frame is continuous over a closed surface, so the choice is
#: only where to put the discontinuity; this confines it to two small caps
#: instead of spreading it over the whole array.
_PARALLEL_TOL = 1e-6


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
        len(origins),
        len(zs),
        n_theta,
        radius,
    )

    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    else:
        logger.info("embree not available — falling back to slow ray.ray_triangle")
        intersector = mesh.ray
    locations, ray_idx, tri_idx = intersector.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True,
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


def _unit_rows(v: np.ndarray) -> np.ndarray:
    """Row-normalise, sending any zero-length row to :data:`_UP`."""
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    degenerate = norms[:, 0] <= 1e-12
    out = np.where(degenerate[:, None], _UP, v / np.where(norms > 0.0, norms, 1.0))
    return out


def tangent_frame(
    normals: np.ndarray, *, basis: str = DEFAULT_TANGENT_BASIS
) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal tangents per outward normal — (N,3) to a pair of (N,3).

    ``global_up`` takes T1 = up x n and T2 = n x T1 about the fixed superior
    axis, so T1 is the horizontal (circumferential) tangent and T2 the
    head-foot one at every sensor. That is what makes a tangential field map
    readable: neighbouring sensors measure the same physical direction, and
    a smooth field looks smooth.

    ``local_svd`` is what the package did before v1.2.0 — the null space of
    each normal taken on its own. It is orthonormal and the radial channel
    and the three-axis norm are identical either way, but the split between
    T1 and T2 is set by LAPACK rather than by anatomy, so it varies (and
    flips sign) between adjacent sensors with near-identical normals. It is
    kept only to reproduce leadfields solved before the change.
    """
    n = _unit_rows(normals)
    if basis == "local_svd":
        t1 = np.empty_like(n)
        t2 = np.empty_like(n)
        for i, nrm in enumerate(n):
            _, _, vh = np.linalg.svd(nrm.reshape(1, 3))
            t1[i] = vh[1]
            t2[i] = vh[2]
        return t1, t2
    if basis != "global_up":
        raise ValueError(f"unknown tangent basis {basis!r}")

    up = np.broadcast_to(_UP, n.shape).copy()
    parallel = np.abs(n @ _UP) > 1.0 - _PARALLEL_TOL
    up[parallel] = _UP_FALLBACK
    if parallel.any():
        logger.debug(
            "%d/%d normals parallel to +Z — tangents there use the +Y reference",
            int(parallel.sum()),
            len(n),
        )
    t1 = np.cross(up, n)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    return t1, t2


def build_triaxial(
    positions: np.ndarray,
    normals: np.ndarray,
    *,
    tangent_basis: str = DEFAULT_TANGENT_BASIS,
) -> SensorArray:
    """Build a triaxial :class:`SensorArray` from positions + outward normals.

    Output channel order: all N radial, then all N tangent-1, then all N
    tangent-2. ``tangent_basis`` picks how T1/T2 are oriented — see
    :func:`tangent_frame`.
    """
    n = len(positions)
    coilpos = np.tile(positions, (3, 1))
    coilori = np.zeros((3 * n, 3))
    labels: list[str] = []

    radial = _unit_rows(normals)
    t1, t2 = tangent_frame(radial, basis=tangent_basis)
    coilori[:n] = radial
    coilori[n : 2 * n] = t1
    coilori[2 * n :] = t2

    for i in range(n):
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


def build_radial(positions: np.ndarray, normals: np.ndarray) -> SensorArray:
    """Build a single-axis :class:`SensorArray`: one R coil per position.

    R is the outward normal, exactly as in :func:`build_triaxial`, so a
    radial array's channels are the first block of the triaxial array on the
    same positions and a leadfield solved on one can be checked against the
    other.
    """
    n = len(positions)
    return SensorArray(
        coilpos=np.asarray(positions, dtype=float).reshape(-1, 3).copy(),
        coilori=_unit_rows(normals),
        labels=tuple(f"mag-{i + 1:04d}-R" for i in range(n)),
        chantype=tuple(["megmag"] * n),
        chanunit=tuple(["T"] * n),
        unit="mm",
    )


def import_sensor_array(cfg: Config, path: Path) -> Path:
    """Read a magnetometer array from ``path`` and save it as ``generate`` would."""
    logger.info("Importing sensor array: %s", path)
    logger.info("  %s", FRAME_NOTE)
    raw = read_array_file(path, kind="meg")
    array = import_magnetometers(raw, skin=load_skin_if_present(cfg))
    validate_sensors(array)
    out = cfg.outputs.sensors_mat
    save_sensors(out, array)
    logger.info("[saved] %s (%d channels)", out, len(array.coilpos))
    return out


def generate_sensor_array(
    cfg: Config,
    *,
    z_min: float | None = None,
    z_max: float | None = None,
    from_file: str | Path | None = None,
) -> Path:
    """Build the sensor array per ``cfg`` and save to HDF5.

    ``from_file`` (else ``sensors.from_file``) imports an array instead of
    placing one; see :mod:`inob.sensors.imported`.
    """
    path = resolve_array_file(cfg, from_file) or resolve_array_file(cfg, cfg.sensors.from_file)
    if path is not None:
        if z_min is not None or z_max is not None:
            logger.warning("--zmin/--zmax are ignored for an imported array")
        return import_sensor_array(cfg, path)

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
        z_min,
        z_max,
        z_lo,
        z_hi,
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
        raise RuntimeError("No surface hits — check sensors.z_crop_low_factor / mesh bounds.")

    positions_offset = positions + normals * cfg.sensors.depth_mm
    if cfg.sensors.axes == "radial":
        logger.info("axes: radial only (one coil per position, outward skin normal)")
        array = build_radial(positions_offset, normals)
    else:
        logger.info("tangent basis: %s", cfg.sensors.tangent_basis)
        array = build_triaxial(positions_offset, normals, tangent_basis=cfg.sensors.tangent_basis)
    validate_sensors(array)

    out = cfg.outputs.sensors_mat
    save_sensors(out, array)
    logger.info("[saved] %s (%d channels)", out, len(array.coilpos))
    return out
