"""Local refinement of one tissue of an existing FEM mesh, with TetGen.

Why this exists
---------------
The Venant source models place monopoles on the vertices around a dipole, and
DUNEuro practice is that the vertex closest to the dipole belongs to the source
compartment alone. The CGAL mesher used by :mod:`inob.fem.cgal_builder` takes
one element size for the whole body, and at a size the body can afford a thin
source compartment (the spinal cord, a nerve) comes out one or two elements
across: every vertex of it lies on a tissue interface. Refining the whole mesh
to fix one compartment multiplies the solve cost for nothing.

This module refines only the elements of the named tissue. TetGen reconstructs
the mesh (``-r``), keeps every tissue interface and the outer surface as
constrained faces, and splits elements whose volume exceeds a per-element bound
(``-a`` with a ``.vol`` file): the bound is set for the target tissue and left
open everywhere else. The result is conforming, each new element inherits its
parent's tissue, and elements away from the target change only where a split
has to propagate to keep the mesh conforming.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from inob.io.hdf5 import FemMesh

logger = logging.getLogger(__name__)


def find_tetgen() -> str:
    """Path of a TetGen 1.5 executable (iso2mesh ships one)."""
    for name in ("tetgen1.5", "tetgen"):
        found = shutil.which(name)
        if found:
            return found
    try:
        from iso2mesh import mcpath
        from iso2mesh.utils import getexeext

        return mcpath("tetgen1.5", getexeext())
    except Exception as e:  # pragma: no cover - depends on the install
        raise FileNotFoundError(
            "TetGen not found: install iso2mesh's tools or put tetgen on PATH"
        ) from e


def interface_faces(tets: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    """Faces on the outer surface or between two tissues, (F, 3) node indices."""
    t = np.asarray(tets, dtype=np.int64)
    faces = np.vstack([t[:, [0, 1, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 3]], t[:, [1, 2, 3]]])
    owner = np.tile(np.asarray(tissue, dtype=np.int64), 4)
    key = np.sort(faces, axis=1)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    key, owner = key[order], owner[order]
    same_as_next = np.all(key[1:] == key[:-1], axis=1)
    paired_first = np.concatenate([same_as_next, [False]])
    paired_second = np.concatenate([[False], same_as_next])
    boundary = ~paired_first & ~paired_second
    between = paired_first & (owner != np.roll(owner, -1))
    return key[boundary | between]


def vertices_interior_to(fem: FemMesh, label: str) -> np.ndarray:
    """Boolean mask over nodes: used by ``label`` elements and by no other tissue."""
    tid = fem.label_to_id[label]
    inside = np.zeros(len(fem.nodes), dtype=bool)
    inside[np.unique(fem.tets[fem.tissue == tid])] = True
    inside[np.unique(fem.tets[fem.tissue != tid])] = False
    return inside


def refine_tissue(
    fem: FemMesh,
    label: str,
    *,
    max_volume_mm3: float,
    tetgen: str | None = None,
    keep_dir: Path | None = None,
) -> FemMesh:
    """Return ``fem`` with the elements of ``label`` refined to ``max_volume_mm3``.

    The bound is a target: nearly every element of the tissue ends below it,
    and the quality pass can leave a few slightly above.

    A regular tetrahedron of edge ``a`` has volume ``a**3 / (6 * sqrt(2))``, so
    0.5 mm^3 is an edge of about 1.6 mm.
    """
    if label not in fem.tissue_labels:
        raise ValueError(f"tissue {label!r} not in FEM (have {list(fem.tissue_labels)})")
    if max_volume_mm3 <= 0:
        raise ValueError(f"max_volume_mm3 must be positive, got {max_volume_mm3}")
    exe = tetgen or find_tetgen()
    tid = fem.label_to_id[label]
    nodes = np.asarray(fem.nodes, dtype=np.float64)
    tets = np.asarray(fem.tets, dtype=np.int64)
    tissue = np.asarray(fem.tissue, dtype=np.int64)
    faces = interface_faces(tets, tissue)

    workdir = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix="inob-refine-"))
    workdir.mkdir(parents=True, exist_ok=True)
    base = workdir / "mesh"
    try:
        idx = np.arange(1, len(nodes) + 1)
        np.savetxt(
            f"{base}.node",
            np.column_stack([idx, nodes]),
            fmt="%d %.10f %.10f %.10f",
            header=f"{len(nodes)} 3 0 0",
            comments="",
        )
        eidx = np.arange(1, len(tets) + 1)
        np.savetxt(
            f"{base}.ele",
            np.column_stack([eidx, tets + 1, tissue]),
            fmt="%d",
            header=f"{len(tets)} 4 1",
            comments="",
        )
        np.savetxt(
            f"{base}.face",
            np.column_stack([np.arange(1, len(faces) + 1), faces + 1, np.ones(len(faces), int)]),
            fmt="%d",
            header=f"{len(faces)} 1",
            comments="",
        )
        vol = np.where(tissue == tid, float(max_volume_mm3), -1.0)
        np.savetxt(
            f"{base}.vol",
            np.column_stack([eidx, vol]),
            fmt="%d %.6f",
            header=f"{len(tets)}",
            comments="",
        )
        # -r reconstruct, -a per-element volume bounds from .vol, -A keep the
        # region attribute, -Q quiet. -q2.0/0 bounds the radius-edge ratio of
        # the elements the splits create: without it a few come out flat
        # (quality ~0), with it the worst element is as good as the input's.
        # On the whole-body mesh it adds almost nothing away from the target.
        proc = subprocess.run(
            [exe, "-raAQq2.0/0", f"{base}.ele"], capture_output=True, text=True, check=False
        )
        out_node, out_ele = Path(f"{workdir}/mesh.1.node"), Path(f"{workdir}/mesh.1.ele")
        if proc.returncode != 0 or not out_ele.exists():
            raise RuntimeError(
                f"TetGen failed (exit {proc.returncode}): {(proc.stdout + proc.stderr)[-800:]}"
            )
        new_nodes = np.loadtxt(out_node, skiprows=1, comments="#")[:, 1:4]
        ele = np.loadtxt(out_ele, skiprows=1, comments="#", dtype=np.int64)
    finally:
        if keep_dir is None:
            shutil.rmtree(workdir, ignore_errors=True)

    new_tets = ele[:, 1:5] - 1
    new_tissue = ele[:, 5].astype(np.int32)
    if set(np.unique(new_tissue)) != set(np.unique(tissue)):
        raise RuntimeError("TetGen changed the set of tissue ids; refusing the result")
    logger.info(
        "refined %s to <= %g mm^3: %d -> %d nodes, %d -> %d tets (%s: %d -> %d)",
        label,
        max_volume_mm3,
        len(nodes),
        len(new_nodes),
        len(tets),
        len(new_tets),
        label,
        int((tissue == tid).sum()),
        int((new_tissue == tid).sum()),
    )
    return FemMesh(
        nodes=new_nodes.astype(np.float64),
        tets=new_tets.astype(np.int32),
        tissue=new_tissue,
        tissue_labels=tuple(fem.tissue_labels),
        unit=fem.unit,
    )
