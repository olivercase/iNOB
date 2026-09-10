"""The solved field *inside* the volume conductor, not just at the sensors.

Every other forward path in this package reads the FEM solution only where a
sensor sits — at coils (:mod:`inob.forward.solve`) or at electrodes
(:mod:`inob.forward.eeg`). DUNEuro solves for the potential everywhere in the
mesh, and that solution is what actually explains a topography: where the
current goes, which tissue shunts it, why a nerve two millimetres deeper is
invisible. This module reads it out.

Two routes, because the DUNEuro Python bindings expose two different things:

* **A dipole's own field → VTK.** ``solveEEGForward`` writes the solution into
  an opaque ``FunctionWrapper``; its DOF coefficients are not reachable from
  Python, so the only way out is the volume VTK writer. That is fine — the
  result is a ParaView-ready file with the potential on every vertex and its
  gradient on every cell.
* **A field as numbers at chosen points.** ``evaluateMultipleFunctionsAtPositions``
  takes a matrix whose *rows are DOF coefficient vectors* — which is exactly
  what the rows of the EEG transfer matrix are. Each row is the potential for
  unit current injected at one electrode (that is why DUNEuro's
  ``solveTDCSForward`` is literally ``computeEEGTransferMatrix``). So the
  stimulation field of any electrode montage is a linear combination of transfer
  rows, evaluated anywhere you like, as potential, gradient (E field) or
  ``-sigma * grad`` (current density).

Both are read-outs of the same solve the leadfield already pays for.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from inob.config import Config
from inob.forward.duneuro_driver import (
    build_driver,
    build_source_model_config,
    import_duneuro,
)
from inob.io.hdf5 import FemMesh

logger = logging.getLogger(__name__)

#: What ``evaluateMultipleFunctionsAtPositions`` returns per position.
#: ``direct`` → the potential (1 value); ``gradient`` → ∇u (3 values, the E
#: field up to sign); ``current`` → −σ∇u (3 values, the current density).
EVALUATION_TYPES: tuple[str, ...] = ("direct", "gradient", "current")

#: Values per position for each evaluation type.
_VALUES_PER_POSITION: dict[str, int] = {"direct": 1, "gradient": 3, "current": 3}


@dataclass(frozen=True)
class VolumeField:
    """A scalar or vector field sampled at points inside the volume conductor."""

    positions_mm: np.ndarray  # (P, 3)
    values: np.ndarray  # (P,) for "direct", (P, 3) otherwise
    evaluation_type: str  # one of EVALUATION_TYPES
    tissue_ids: np.ndarray | None = None  # (P,) FEM tissue id per point, if known
    description: str = ""

    @property
    def magnitude(self) -> np.ndarray:
        """One number per point: |value|, so a vector field can be coloured."""
        v = np.asarray(self.values, dtype=float)
        return np.abs(v) if v.ndim == 1 else np.linalg.norm(v, axis=1)


def save_volume_field(path: Path, field: VolumeField) -> None:
    """Atomically save a :class:`VolumeField` to an NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        np.savez(
            tmp_path,
            positions_mm=np.asarray(field.positions_mm, dtype=np.float64),
            values=np.asarray(field.values, dtype=np.float64),
            evaluation_type=np.array(field.evaluation_type),
            tissue_ids=(
                np.asarray(field.tissue_ids, dtype=np.int64)
                if field.tissue_ids is not None
                else np.zeros(0, dtype=np.int64)
            ),
            description=np.array(field.description),
        )
        candidate = (
            tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
        )
        os.replace(candidate, path)
    except BaseException:
        for p in (tmp_path, tmp_path.with_suffix(tmp_path.suffix + ".npz")):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    logger.info("Wrote volume field → %s (%.1f MB)", path, path.stat().st_size / 1e6)


def load_volume_field(path: Path) -> VolumeField:
    """Load a volume field NPZ written by :func:`save_volume_field`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"volume field not found: {path}")
    with np.load(path, allow_pickle=False) as f:
        tissue = f["tissue_ids"] if "tissue_ids" in f.files else np.zeros(0)
        return VolumeField(
            positions_mm=f["positions_mm"],
            values=f["values"],
            evaluation_type=str(f["evaluation_type"]),
            tissue_ids=tissue if tissue.size else None,
            description=str(f["description"]) if "description" in f.files else "",
        )


def sample_points(
    fem: FemMesh,
    *,
    spacing_mm: float = 0.0,
    tissues: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Points strictly inside the mesh, with their tissue id.

    Returns ``(positions_mm, tissue_ids)``. Tetrahedron centroids are used
    rather than a regular grid because a centroid is inside its element by
    construction — a grid point near a boundary lands outside the volume and
    DUNEuro's element search then throws.

    ``tissues`` restricts to named compartments; ``spacing_mm`` thins the result
    to roughly one point per cube of that size, which is what keeps a whole-neck
    field readable (and small enough to ship to a browser).
    """
    centroids = fem.nodes[fem.tets].mean(axis=1)
    tissue_ids = np.asarray(fem.tissue, dtype=np.int64)

    if tissues:
        wanted = {fem.label_to_id[t] for t in tissues}
        keep = np.isin(tissue_ids, list(wanted))
        centroids, tissue_ids = centroids[keep], tissue_ids[keep]

    if spacing_mm > 0 and len(centroids):
        # One representative per occupied voxel: quantise, then take the first
        # centroid in each cell. Deterministic, and no distance matrix.
        cells = np.floor(centroids / spacing_mm).astype(np.int64)
        _, first = np.unique(cells, axis=0, return_index=True)
        first.sort()
        centroids, tissue_ids = centroids[first], tissue_ids[first]

    return centroids, tissue_ids


def evaluate_dof_rows(
    driver: Any,
    dof_rows: np.ndarray,
    positions_mm: np.ndarray,
    *,
    evaluation_type: str = "current",
) -> np.ndarray:
    """Evaluate DOF-coefficient rows at ``positions_mm``.

    ``dof_rows`` is ``(n_functions, n_dofs)`` — each row the coefficients of one
    FE function, e.g. one row of the EEG transfer matrix. Returns
    ``(n_functions, n_positions)`` for ``direct`` and
    ``(n_functions, n_positions, 3)`` for ``gradient`` / ``current``.
    """
    if evaluation_type not in EVALUATION_TYPES:
        raise ValueError(
            f"evaluation_type must be one of {list(EVALUATION_TYPES)}; got {evaluation_type!r}"
        )
    rows = np.atleast_2d(np.asarray(dof_rows, dtype=np.float64))
    positions = [list(map(float, p)) for p in np.asarray(positions_mm, dtype=float)]
    raw, _ = driver.evaluateMultipleFunctionsAtPositions(
        rows,
        positions,
        {"evaluation_return_type": evaluation_type},
    )
    # DUNEuro returns one row per function, with the per-position values laid
    # out consecutively (1 or 3 of them).
    out = np.asarray(raw, dtype=np.float64).reshape(len(rows), len(positions), -1)
    stride = _VALUES_PER_POSITION[evaluation_type]
    if out.shape[2] != stride:
        raise ValueError(
            f"expected {stride} value(s) per position for {evaluation_type!r}, got {out.shape[2]}"
        )
    return out[:, :, 0] if stride == 1 else out


def stimulation_field(
    cfg: Config,
    fem: FemMesh,
    electrode_pos_mm: np.ndarray,
    *,
    anode: int,
    cathode: int,
    current_mA: float = 1.0,
    positions_mm: np.ndarray | None = None,
    evaluation_type: str = "current",
    spacing_mm: float = 5.0,
    tissues: tuple[str, ...] = (),
) -> VolumeField:
    """The field a bipolar stimulation montage drives through the volume.

    Reciprocity, used forwards: row *i* of the EEG transfer matrix is the
    potential for unit current injected at electrode *i*, so a montage driving
    ``current`` in at ``anode`` and out at ``cathode`` has DOF coefficients
    ``I·(T[anode] − T[cathode])``. That is DUNEuro's own definition of the tDCS
    forward problem — ``solveTDCSForward`` is an alias for
    ``computeEEGTransferMatrix``.

    Units follow the pipeline's mm-mode convention (σ in S/mm, lengths in mm),
    so treat the magnitudes as relative unless you have calibrated them the way
    :mod:`inob.analysis.sphere_calibration` calibrates the leadfields.
    """
    from inob.forward.eeg import attach_electrodes

    dp = import_duneuro(cfg)
    driver, driver_cfg, _cond = build_driver(cfg, fem)
    attach_electrodes(driver, dp, electrode_pos_mm)

    logger.info("computing EEG transfer matrix for the stimulation field (slow)…")
    T_raw, _ = driver.computeEEGTransferMatrix(driver_cfg)
    T = np.asarray(T_raw, dtype=np.float64)
    n = T.shape[0]
    for name, idx in (("anode", anode), ("cathode", cathode)):
        if not 0 <= idx < n:
            raise ValueError(f"{name} {idx} out of range (have {n} electrodes)")
    if anode == cathode:
        raise ValueError("anode and cathode must be different electrodes")

    dof = current_mA * (T[anode] - T[cathode])

    if positions_mm is None:
        positions_mm, tissue_ids = sample_points(fem, spacing_mm=spacing_mm, tissues=tissues)
    else:
        positions_mm, tissue_ids = np.asarray(positions_mm, dtype=float), None
    logger.info("evaluating %s at %d points", evaluation_type, len(positions_mm))

    values = evaluate_dof_rows(driver, dof[None, :], positions_mm, evaluation_type=evaluation_type)[
        0
    ]
    return VolumeField(
        positions_mm=positions_mm,
        values=values,
        evaluation_type=evaluation_type,
        tissue_ids=tissue_ids,
        description=(
            f"stimulation {current_mA} mA, anode {anode} → cathode {cathode}, {evaluation_type}"
        ),
    )


def export_source_field_vtk(
    cfg: Config,
    fem: FemMesh,
    *,
    source_pos_mm: np.ndarray,
    moment: np.ndarray,
    out_path: Path,
    subsampling_levels: int = 0,
) -> Path:
    """Write one dipole's potential field over the whole mesh as VTK.

    The potential lands on every vertex and its gradient on every cell, so
    ParaView (or any VTK reader) can show what the source does *inside* the
    body rather than only what reaches the sensors.

    Returns the written path. DUNEuro appends its own extension, so ``out_path``
    is passed without one and the actual file is ``<out_path>.vtu``.

    Note ``addVertexDataGradient`` is unreachable from Python — duneuro-py binds
    it under the name ``addVertexData``, shadowed by the real overload of that
    name — so the gradient is written as cell data instead.
    """
    dp = import_duneuro(cfg)
    driver, driver_cfg, _cond = build_driver(cfg, fem)

    dipole = dp.Dipole3d(np.asarray(source_pos_mm, dtype=float), np.asarray(moment, dtype=float))
    solution = driver.makeDomainFunction()
    solve_cfg = {**driver_cfg, "source_model": build_source_model_config(cfg)}
    logger.info("solving the volume potential for one dipole (slow)…")
    driver.solveEEGForward(dipole, solution, solve_cfg)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = driver.volumeConductorVTKWriter({"anisotropy.enable": "false"})
    writer.addVertexData(solution, "potential")
    writer.addCellDataGradient(solution, "gradient")
    writer.write({"filename": str(out_path), "subsamplingLevels": str(subsampling_levels)})
    written = out_path.with_suffix(".vtu")
    logger.info("Wrote volume potential → %s", written)
    return written
