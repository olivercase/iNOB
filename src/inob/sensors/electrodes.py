"""HD surface-electrode array placement on the skin.

For the dual-modality (MEG/EEG) forward model, we co-locate a high-density
PEDOT:PSS-style electrode patch over the imaging target. Which tissue the
patch is centred over follows ``electrodes.target_tissue``, which
``--source-target`` sets from :data:`inob.config.SOURCE_TARGETS` — a spine run
sites the patch over the cord, a vagus run over the cervical vagus. The OPM
array wraps the whole torso and is target-agnostic; this patch is not, so
leaving it on the vagus for a spine solve would compare patch placement rather
than modality. Geometry:

  1. Find a centre pose on the skin near the target (project the mean of the
     target tissue's tet centroids over a Z slab onto the skin surface).
  2. Build a tangent frame at that centre (outward normal + two tangents).
  3. Generate a regular ``rows × cols`` grid of contact positions in that
     local frame, with ``contact_pitch_mm`` spacing.
  4. Project each grid point onto the closest skin face (so contacts sit
     exactly on the skin surface, accommodating local curvature).

The output uses the same ``grad`` HDF5 group as the OPM array but with
``chantype="eeg"`` and ``chanunit="V"``. Coil orientations are the outward
skin normals at each contact (informational only — DUNEuro EEG ignores
orientation).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from inob.anatomy import vertebra_z_band
from inob.config import Config
from inob.io.hdf5 import (
    FemMesh,
    SchemaError,
    SensorArray,
    load_fem,
    save_sensors,
    validate_sensors,
)
from inob.io.stl import load_stl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElectrodeArrayParams:
    rows: int
    cols: int
    contact_pitch_mm: float
    shape: str = "rectangular"           # rectangular / paddle32 / whole_body
    head_offset_mm: float = 15.0         # paddle: distance from grid top to head contact
    foot_offset_mm: float = 15.0         # paddle: distance from grid bottom to foot contact
    target_tissue: str = "vagus_left"
    target_z_low_factor: float = 0.6
    target_z_high_factor: float = 0.9
    # When set (e.g. "c7"), the patch is centred on this vertebra's absolute Z
    # band instead of the fractional body-height slab — so it sits over the
    # source of interest. ``target_z_band_mm`` is the resolved (z_lo, z_hi) mm,
    # filled in by the caller (which has the STL dir); level is kept for logs.
    target_level: str | None = None
    target_z_band_mm: tuple[float, float] | None = None
    label_prefix: str = "elec"
    n_contacts: int = 1000               # whole_body: total contact count
    sample_seed: int = 0                 # whole_body: RNG seed for surface sampling


def _target_centre(fem: FemMesh, target_tissue: str,
                   z_low_factor: float, z_high_factor: float,
                   z_band_mm: tuple[float, float] | None = None) -> np.ndarray:
    """3-D centre = mean of ``target_tissue`` tet centroids within a Z band.

    The band is either an absolute ``z_band_mm`` (mm — e.g. a vertebra's STL
    bounding box, used to place the patch over a chosen level) or, when that is
    ``None``, the ``[z_low, z_high]`` fractional slab over the body Z extent.
    """
    if target_tissue not in fem.tissue_labels:
        raise SchemaError(
            f"electrode target tissue {target_tissue!r} not in FEM "
            f"(have {list(fem.tissue_labels)})"
        )
    tid = fem.label_to_id[target_tissue]
    mask = fem.tissue == tid
    if not mask.any():
        raise SchemaError(f"no tets with tissue id {tid} ({target_tissue!r})")
    centroids = fem.nodes[fem.tets[mask]].mean(axis=1)
    if z_band_mm is not None:
        z_lo, z_hi = z_band_mm
    else:
        body_z_lo = float(fem.nodes[:, 2].min())
        body_z_hi = float(fem.nodes[:, 2].max())
        z_lo = body_z_lo + z_low_factor * (body_z_hi - body_z_lo)
        z_hi = body_z_lo + z_high_factor * (body_z_hi - body_z_lo)
    sel = (centroids[:, 2] >= z_lo) & (centroids[:, 2] <= z_hi)
    if not sel.any():
        logger.warning(
            "no %s centroids in Z [%g, %g]; using full-tissue mean",
            target_tissue, z_lo, z_hi,
        )
        sel = np.ones(len(centroids), dtype=bool)
    return centroids[sel].mean(axis=0)


def _project_to_skin(skin: trimesh.Trimesh, p: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Closest-point projection of ``p`` onto the skin mesh.

    Returns ``(projected_point_mm, outward_normal_unit)``.
    """
    closest, _, tri_idx = trimesh.proximity.closest_point(skin, p[None, :])
    pt = closest[0]
    nrm = skin.face_normals[tri_idx[0]]
    # ensure outward (away from body centroid)
    body_c = skin.centroid
    if np.dot(nrm, pt - body_c) < 0:
        nrm = -nrm
    return pt, nrm / max(float(np.linalg.norm(nrm)), 1e-12)


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal tangents given an outward normal."""
    _, _, vh = np.linalg.svd(normal.reshape(1, 3))
    t1, t2 = vh[1], vh[2]
    return t1, t2


def _local_uv_rectangular(rows: int, cols: int, pitch: float) -> tuple[np.ndarray, list[str]]:
    """Regular ``rows × cols`` grid in the (u, v) tangent plane."""
    u = (np.arange(cols) - (cols - 1) / 2.0) * pitch
    v = (np.arange(rows) - (rows - 1) / 2.0) * pitch
    UU, VV = np.meshgrid(u, v, indexing="ij")
    uv = np.column_stack([UU.ravel(), VV.ravel()])
    labels = [f"elec-{r:02d}-{c:02d}" for r in range(rows) for c in range(cols)]
    return uv, labels


def _local_uv_paddle32(
    pitch: float, head_offset_mm: float, foot_offset_mm: float,
    *, rows: int = 6, cols: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Paddle/figurine layout: 1 head + ``rows × cols`` body + 1 foot = 32 contacts.

    Layout (drawn for rows=6, cols=5):

           o            head  (above the grid)
       o o o o o
       o o o o o
       o o o o o        body  (rows × cols regular grid)
       o o o o o
       o o o o o
       o o o o o
           o            foot  (below the grid)

    The head and foot contacts are aligned to the grid centre column at
    ``head_offset_mm`` above the top row and ``foot_offset_mm`` below the
    bottom row. Total channel count = 1 + rows*cols + 1.
    """
    u = (np.arange(cols) - (cols - 1) / 2.0) * pitch
    v = (np.arange(rows) - (rows - 1) / 2.0) * pitch
    UU, VV = np.meshgrid(u, v, indexing="ij")
    body = np.column_stack([UU.ravel(), VV.ravel()])
    head = np.array([[0.0, v.max() + head_offset_mm]])
    foot = np.array([[0.0, v.min() - foot_offset_mm]])
    uv = np.vstack([head, body, foot])
    labels = ["elec-head-00"] \
        + [f"elec-{r:02d}-{c:02d}" for r in range(rows) for c in range(cols)] \
        + ["elec-foot-00"]
    return uv, labels


def _whole_body_electrodes(
    skin: trimesh.Trimesh, n_contacts: int, *, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample ~``n_contacts`` electrodes uniformly across the entire skin.

    Uses ``trimesh.sample.sample_surface_even`` (Poisson-disk-like) for
    near-uniform spacing; falls back to area-weighted random sampling if
    Poisson-disk underdelivers. Outward normals are taken from the face the
    sample lands on.
    """
    rng = np.random.default_rng(seed)
    try:
        pts, face_idx = trimesh.sample.sample_surface_even(
            skin, count=n_contacts, seed=int(seed),
        )
    except TypeError:
        pts, face_idx = trimesh.sample.sample_surface_even(skin, count=n_contacts)
    if len(pts) < int(0.8 * n_contacts):
        # Poisson-disk under-delivered (mesh too noisy or count too high).
        try:
            pts, face_idx = trimesh.sample.sample_surface(skin, count=n_contacts,
                                                           seed=int(seed))
        except TypeError:
            pts, face_idx = trimesh.sample.sample_surface(skin, count=n_contacts)
    n = len(pts)
    if n == 0:
        raise RuntimeError("whole-body sampling returned 0 points")
    normals = np.asarray(skin.face_normals[face_idx], dtype=np.float64)
    body_c = skin.centroid
    flip = np.einsum("ij,ij->i", normals, pts - body_c) < 0
    normals = np.where(flip[:, None], -normals, normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    width = max(4, int(np.ceil(np.log10(max(n, 10)))))
    labels = [f"wb-{i:0{width}d}" for i in range(n)]
    _ = rng                         # reserved for future jitter / decimation
    return np.asarray(pts, dtype=np.float64), normals, labels


def build_electrode_array(
    skin: trimesh.Trimesh,
    fem: FemMesh,
    params: ElectrodeArrayParams,
) -> SensorArray:
    """Place electrodes on the skin per ``params.shape``.

    ``rectangular`` / ``paddle32`` are patches over the target; ``whole_body``
    is a uniform skin-wide net.
    """
    if params.shape == "whole_body":
        pts, normals, labels = _whole_body_electrodes(
            skin, params.n_contacts, seed=params.sample_seed,
        )
        n = len(pts)
        return SensorArray(
            coilpos=pts, coilori=normals,
            labels=tuple(labels),
            chantype=tuple(["eeg"] * n),
            chanunit=tuple(["V"] * n),
            unit="mm",
        )

    centre_3d = _target_centre(
        fem, params.target_tissue,
        params.target_z_low_factor, params.target_z_high_factor,
        z_band_mm=params.target_z_band_mm,
    )
    if params.target_z_band_mm is not None:
        logger.info("electrode patch centred on level %s (Z %.1f..%.1f mm)",
                    params.target_level, *params.target_z_band_mm)
    centre, normal = _project_to_skin(skin, centre_3d)
    t1, t2 = _tangent_basis(normal)

    # Pick the tangent that points cranially (towards +Z) so "head" of the
    # paddle ends up at the top of the body.
    if t2[2] < 0:
        t2 = -t2
    # Re-orthogonalise t1 to keep right-handedness with the outward normal.
    t1 = np.cross(t2, normal)
    t1 /= max(float(np.linalg.norm(t1)), 1e-12)

    if params.shape == "rectangular":
        uv, labels = _local_uv_rectangular(params.rows, params.cols,
                                           params.contact_pitch_mm)
    elif params.shape == "paddle32":
        uv, labels = _local_uv_paddle32(
            params.contact_pitch_mm,
            params.head_offset_mm, params.foot_offset_mm,
            rows=params.rows, cols=params.cols,
        )
    else:
        raise ValueError(f"unknown electrode shape {params.shape!r}")

    local_grid = uv[:, 0:1] * t1[None, :] + uv[:, 1:2] * t2[None, :]
    candidates = centre[None, :] + local_grid

    # Project every candidate onto the skin to follow curvature
    closest, _, tri_idx = trimesh.proximity.closest_point(skin, candidates)
    normals = skin.face_normals[tri_idx]
    body_c = skin.centroid
    flip = np.einsum("ij,ij->i", normals, closest - body_c) < 0
    normals = np.where(flip[:, None], -normals, normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    n = len(closest)
    return SensorArray(
        coilpos=closest,
        coilori=normals,
        labels=tuple(labels),
        chantype=tuple(["eeg"] * n),
        chanunit=tuple(["V"] * n),
        unit="mm",
    )


def generate_electrode_array(
    cfg: Config,
    *,
    rows: int | None = None,
    cols: int | None = None,
    contact_pitch_mm: float | None = None,
    target_tissue: str | None = None,
    target_level: str | None = None,
    out_path: Path | None = None,
) -> Path:
    """Generate the HD electrode array per ``cfg.electrodes`` and save it."""
    elec = cfg.electrodes
    # A vertebral level (CLI --level, else cfg.electrodes.target_level) centres
    # the patch on that vertebra's absolute Z band so it sits over the source of
    # interest; without one the fractional body-height slab is used as before.
    level = target_level if target_level is not None else getattr(elec, "target_level", None)
    # An explicit empty string is the "full-region / mid-slab" opt-out (e.g. the
    # spine full-cord survey): it forces the fractional slab even though the
    # source-target default set a level. ``None`` (arg absent) falls to the cfg.
    level = level or None
    z_band_mm = vertebra_z_band(cfg.data.bone_dir, level) if level else None
    params = ElectrodeArrayParams(
        rows=rows if rows is not None else elec.rows,
        cols=cols if cols is not None else elec.cols,
        contact_pitch_mm=contact_pitch_mm if contact_pitch_mm is not None
                         else elec.contact_pitch_mm,
        shape=elec.shape,
        head_offset_mm=elec.head_offset_mm,
        foot_offset_mm=elec.foot_offset_mm,
        target_tissue=target_tissue or elec.target_tissue,
        target_z_low_factor=elec.target_z_low_factor,
        target_z_high_factor=elec.target_z_high_factor,
        target_level=level,
        target_z_band_mm=z_band_mm,
        label_prefix=elec.label_prefix,
        n_contacts=getattr(elec, "n_contacts", 1000),
        sample_seed=getattr(elec, "sample_seed", 0),
    )
    skin = load_stl(cfg.data.torso_skin)
    fem = load_fem(cfg.outputs.fem_mat)
    array = build_electrode_array(skin, fem, params)
    validate_sensors(array)
    out = out_path or cfg.outputs.electrodes_mat
    save_sensors(out, array)
    logger.info(
        "[saved] %s (%d HD contacts %dx%d @ %.1fmm)",
        out, len(array.coilpos), params.rows, params.cols, params.contact_pitch_mm,
    )
    return out
