"""Multi-compartment tetrahedral mesh for the head+thorax cardiac conductor.

``inob.fem.cgal_builder`` builds the vagus study's fixed tissue set from config
paths. This one is generic over compartments: it takes an ordered mapping of
label -> watertight mesh and paints them into one label volume, later entries
overwriting earlier ones, then hands that to CGAL.

The paint order is the anatomy's nesting order, and two choices in it are worth
stating because they are not free:

* **brain after bone.** The skull vault and the brain overlap by a voxel or two
  at the inner table at any realistic pitch. Letting the brain win preserves the
  brain volume and thins the skull by <= 1 voxel. For MEG that is the right way
  round -- the external magnetic field is famously insensitive to skull
  conductivity, and very much not insensitive to the source compartment's extent.
* **heart after lungs.** They abut along the mediastinal surface; anatomically the
  heart displaces the lung, so the heart wins the shared voxels.

Everything not painted is the background soft-tissue body.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import trimesh

from inob.io.hdf5 import FemMesh
from inob.mesh.quality import assert_mesh_ok, compute_quality
from inob.mesh.voxelize import (
    fill_internal_cavities,
    keep_largest_component,
    voxelize_watertight,
)

logger = logging.getLogger(__name__)

__all__ = ["FemSpec", "build_conductor_fem", "PAINT_ORDER"]

#: Outermost first. The body compartment must come first: it is the background
#: every other label is painted into.
PAINT_ORDER: tuple[str, ...] = (
    "skin", "muscle", "bone", "lung_left", "lung_right", "heart",
    "brain", "spinal_cord",
)


@dataclass
class FemSpec:
    """Voxel grid and CGAL parameters."""

    #: Voxel pitch for the label volume, mm.
    pitch_mm: float = 3.0
    #: Padding around the anatomy, mm.
    pad_mm: float = 12.0
    #: CGAL surface element size (mm) and target tet volume (mm^3).
    #:
    #: These are given in MILLIMETRES and converted to the voxel units cgalv2m
    #: actually wants (it meshes the label volume in index space). Passing voxel
    #: units by mistake is a silent 1/pitch^3 error in element count -- at a 3 mm
    #: pitch, "maxvol=30" reads as 30 mm^3 and means 810.
    radbound_mm: float = 4.5
    maxvol_mm3: float = 30.0
    #: Reject the mesh below this element quality.
    min_quality: float = 0.05
    #: Compartments to include, in paint order. Defaults to whatever is supplied.
    order: tuple[str, ...] = field(default_factory=lambda: PAINT_ORDER)


def _grid(meshes: dict[str, trimesh.Trimesh], spec: FemSpec):
    v = np.vstack([np.asarray(m.vertices) for m in meshes.values()])
    mn = v.min(0) - spec.pad_mm
    mx = v.max(0) + spec.pad_mm
    shape = tuple(((mx - mn) / spec.pitch_mm).astype(int) + 1)
    xs = mn[0] + (np.arange(shape[0]) + 0.5) * spec.pitch_mm
    X = np.empty(shape, dtype=np.float32)          # only the shape is used
    logger.info("[fem] grid %s = %.1fM voxels at %.1f mm pitch",
                shape, np.prod(shape) / 1e6, spec.pitch_mm)
    return mn, shape, X


def build_conductor_fem(
    meshes: dict[str, trimesh.Trimesh], spec: FemSpec | None = None,
) -> FemMesh:
    """Paint ``meshes`` into a label volume and tetrahedralise it."""
    import iso2mesh as im

    spec = spec or FemSpec()
    order = [k for k in spec.order if k in meshes]
    missing = set(meshes) - set(order)
    if missing:
        raise KeyError(f"compartments not in the paint order: {sorted(missing)}")
    if order[0] != "skin":
        raise ValueError("the body ('skin') must be painted first, as the background")

    mn, shape, X = _grid(meshes, spec)
    labels: list[str] = []
    vol = np.zeros(shape, dtype=np.uint8)

    for name in order:
        occ = voxelize_watertight(meshes[name], X, pitch=spec.pitch_mm, mn=mn)
        if name == "skin":
            occ = fill_internal_cavities(keep_largest_component(occ))
        n = int(occ.sum())
        if n == 0:
            logger.warning("[fem] %s voxelised to nothing at %.1f mm -- skipped",
                           name, spec.pitch_mm)
            continue
        labels.append(name)
        vol[occ] = len(labels)
        logger.info("[fem] %-12s -> id %d, %8d voxels (%8.1f cm^3)",
                    name, len(labels), n, n * spec.pitch_mm ** 3 / 1000.0)

    body = vol > 0
    if not body.any():
        raise RuntimeError("label volume is empty")
    for i, name in enumerate(labels, start=1):
        kept = int((vol == i).sum())
        if kept == 0:
            raise RuntimeError(
                f"compartment {name!r} was painted but fully overwritten by an "
                f"inner compartment -- check the paint order")

    radbound_vox = spec.radbound_mm / spec.pitch_mm
    maxvol_vox = spec.maxvol_mm3 / spec.pitch_mm ** 3
    logger.info("[fem] cgalv2m(radbound=%.3g mm = %.3g vox, maxvol=%.3g mm^3 = "
                "%.3g vox^3) over %d labels...",
                spec.radbound_mm, radbound_vox, spec.maxvol_mm3, maxvol_vox,
                len(labels))
    node, elem, _face = im.cgalv2m(vol, radbound_vox, maxvol_vox)
    nodes_mm = np.asarray(node[:, :3], float) * spec.pitch_mm + mn
    elem = np.asarray(elem, np.int64)
    tets = elem[:, :4] - 1
    regions = elem[:, 4]

    # CGAL renumbers regions by discovery order, not by our label id. Recover the
    # mapping by majority vote of the label volume at each tet's centroid, which
    # is robust to the odd element straddling an interface.
    cen = nodes_mm[tets].mean(1)
    ijk = np.clip(((cen - mn) / spec.pitch_mm).astype(int), 0, np.array(shape) - 1)
    sampled = vol[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    tissue = np.zeros(len(tets), np.int32)
    for r in np.unique(regions):
        sel = regions == r
        vals, counts = np.unique(sampled[sel], return_counts=True)
        vals, counts = vals[vals > 0], counts[vals > 0]
        if not len(vals):
            tissue[sel] = 1                        # degenerate: call it body
            continue
        tissue[sel] = int(vals[counts.argmax()])

    mesh = FemMesh(nodes=nodes_mm, tets=tets.astype(np.int32), tissue=tissue,
                   tissue_labels=tuple(labels))
    logger.info("[fem] %d nodes, %d tets", len(nodes_mm), len(tets))
    for i, name in enumerate(labels, start=1):
        logger.info("[fem]   %-12s %8d tets", name, int((tissue == i).sum()))
    if (tissue == 0).any():
        raise RuntimeError(f"{int((tissue == 0).sum())} tets got no tissue id")

    q = compute_quality(mesh.nodes, mesh.tets)
    logger.info("[fem] quality min=%.4f p1=%.4f mean=%.4f", q.min, q.p1, q.mean)
    assert_mesh_ok(q, min_quality=spec.min_quality)
    return mesh
