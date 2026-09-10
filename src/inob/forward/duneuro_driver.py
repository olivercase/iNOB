"""Shared DUNEuro driver setup.

Encapsulates the pieces every forward path needs: import the duneuropy
extension (with an optional ``cfg.forward.duneuro_path`` shim for legacy
absolute-path installs), build the conductivity vector from the YAML, and
construct the ``MEEGDriver3d`` config dict.

Used by:
  * :mod:`inob.forward.solve` — single-machine forward solve
  * :mod:`inob.forward.chunk` — cluster array-job worker
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from inob.config import Config
from inob.io.hdf5 import FemMesh

if TYPE_CHECKING:
    import duneuropy as dp  # noqa: F401

logger = logging.getLogger(__name__)


class DuneuroUnavailableError(ImportError):
    """Raised when ``duneuropy`` cannot be imported and no shim path is set."""


def import_duneuro(cfg: Config) -> Any:
    """Import the ``duneuropy`` extension.

    If ``cfg.forward.duneuro_path`` is set, prepend it to ``sys.path`` first
    (useful for source-built duneuro outside a Python venv). Otherwise relies
    on a normal ``import duneuropy`` resolving via the installed venv.
    """
    if cfg.forward.duneuro_path is not None:
        path = str(cfg.forward.duneuro_path.expanduser())
        if path not in sys.path:
            sys.path.insert(0, path)
            logger.info("prepended duneuropy shim path: %s", path)
    try:
        import duneuropy as dp
    except ImportError as e:
        raise DuneuroUnavailableError(
            "duneuropy could not be imported. Install via cluster/build_duneuro.sh "
            "or set forward.duneuro_path in your config to the duneuro-py src dir."
        ) from e
    return dp


def build_conductivity_vector(
    cfg: Config,
    fem: FemMesh,
) -> np.ndarray:
    """Build the conductivity array (S/mm, post unit-scale) aligned with tissue ids.

    Conductivities come from ``cfg.forward.conductivities_sm`` (S/m) scaled by
    ``cfg.forward.sigma_unit_scale``. The output has length ``max(tissue_id)``,
    indexed at position ``tissue_id - 1``. Raises if any tissue label in the
    FEM lacks a conductivity entry.
    """
    label_to_id = fem.label_to_id
    missing = [lab for lab in fem.tissue_labels if lab not in cfg.forward.conductivities_sm]
    if missing:
        raise KeyError(f"forward.conductivities_sm missing entries for tissues: {missing}")
    cond_size = int(np.unique(fem.tissue).max())
    cond = np.zeros(cond_size, dtype=np.float64)
    for lab, tid in label_to_id.items():
        cond[tid - 1] = cfg.forward.conductivities_sm[lab] * cfg.forward.sigma_unit_scale
    return cond


def build_conductivity_tensors(
    cfg: Config,
    fem: FemMesh,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Per-element conductivity tensors for fibre-aligned muscle anisotropy.

    Returns ``(labels, tensors)`` — ``labels`` (one index per tet) into
    ``tensors`` (a list of 3×3 S/mm matrices) — or ``None`` when the isotropic
    scalar path applies instead.

    Whether anisotropy applies is *derived*, not flagged: under the default
    ``forward.muscle_anisotropy.mode = "auto"`` it is on exactly when ``muscle``
    is one of the ``forward.source_tissue`` labels. So a muscle solve gets the
    tensor and a vagus/spine solve stays byte-identical to the scalar path,
    with nothing to remember to set (see :meth:`MuscleAnisotropyCfg.active_for`).

    Layout: the first ``L = len(cond)`` tensors are the isotropic ``σ·I`` of
    every tissue (positions match the scalar ``build_conductivity_vector``);
    then one anisotropic tensor per muscle STL is appended and every muscle tet
    is relabelled to point at its muscle's tensor. Muscle tensor:
    ``σ_⊥·I + (σ_∥ − σ_⊥)·ê⊗ê`` with ``ê`` the muscle's unit fibre axis.
    duneuro (``volume_conductor.tensors``) then reads ``tensors[labels[e]]``.
    """
    aniso = cfg.forward.muscle_anisotropy
    if "muscle" not in fem.tissue_labels:
        return None
    if not aniso.active_for(cfg.forward.source_tissue):
        return None

    from inob.sources.muscle import muscle_tet_fibre_axes

    scale = cfg.forward.sigma_unit_scale
    cond = build_conductivity_vector(cfg, fem)  # (L,) S/mm, isotropic
    eye = np.eye(3, dtype=np.float64)
    tensors: list[np.ndarray] = [float(c) * eye for c in cond]
    base = len(tensors)

    group, axes, mask = muscle_tet_fibre_axes(fem, muscle_dir=cfg.data.muscle_dir)
    sl, st = aniso.sigma_long_sm * scale, aniso.sigma_trans_sm * scale
    for e in axes:
        e = e / max(float(np.linalg.norm(e)), 1e-12)
        tensors.append(
            np.ascontiguousarray(st * eye + (sl - st) * np.outer(e, e), dtype=np.float64)
        )

    labels = fem.tissue.astype(np.int64) - 1
    labels[mask] = base + group
    return labels, tensors


def build_source_model_config(cfg: Config) -> dict[str, str]:
    """Return the DUNEuro ``source_model`` sub-config from the YAML.

    This is the entry that turns a point dipole into a FEM right-hand side —
    see :class:`inob.config.SourceModelCfg` for what the models are and when
    each is appropriate. It is passed to ``applyEEGTransfer`` /
    ``applyMEGTransfer`` / ``computeMEGPrimaryField``, *not* to the transfer
    matrix computation: the transfer matrix depends only on the solver, the
    volume conductor and the sensors, so one transfer matrix can be re-applied
    with several source models. That makes an A/B between source models cheap
    — only the (fast) apply step repeats.

    Every value is stringified: DUNEuro parses this dict into a
    ``Dune::ParameterTree``, whose typed ``get<T>`` accepts strings, and whose
    ``get<bool>`` accepts ``"true"``/``"false"``. The Venant keys have no
    defaults on the C++ side, so all of them must be present or the driver
    throws.
    """
    s = cfg.forward.source_model
    out: dict[str, str] = {"type": s.type}
    if s.type in ("venant", "multipolar_venant"):
        out.update(
            {
                "numberOfMoments": str(s.number_of_moments),
                "referenceLength": str(s.reference_length_mm),
                "weightingExponent": str(s.weighting_exponent),
                "relaxationFactor": str(s.relaxation_factor),
                "mixedMoments": str(s.mixed_moments).lower(),
                "restrict": str(s.restrict).lower(),
                "initialization": s.initialization,
                "extensions": s.extensions,
                "intorderadd": str(s.intorderadd),
            }
        )
    return out


def build_driver_config(
    cfg: Config,
    fem: FemMesh,
    cond: np.ndarray,
    *,
    aniso_tensors: tuple[np.ndarray, list[np.ndarray]] | None = None,
    limit_threads: bool = False,
) -> dict[str, Any]:
    """Return the MEEGDriver3d configuration dictionary.

    When ``aniso_tensors`` is given the volume conductor is described by full
    per-element 3×3 tensors (``labels`` + ``tensors``); otherwise by the scalar
    isotropic ``labels`` + ``conductivities`` path.
    """
    s = cfg.forward.solver
    tissue0 = fem.tissue.astype(np.int64) - 1
    if aniso_tensors is not None:
        labels, tensor_list = aniso_tensors
        vc_tensors: dict[str, Any] = {
            "labels": labels.astype(np.int64),
            "tensors": [np.asarray(t, dtype=np.float64) for t in tensor_list],
        }
    else:
        vc_tensors = {"labels": tissue0, "conductivities": cond}
    return {
        "type": "fitted",
        "solver_type": s.type,
        "element_type": "tetrahedron",
        "post_process": "false",
        "post_process_meg": str(s.post_process_meg).lower(),
        "subtract_mean": str(s.subtract_mean).lower(),
        "solver": {
            "reduction": str(s.reduction),
            "edge_norm_type": s.edge_norm_type,
            "penalty": str(s.penalty),
            "scheme": s.scheme,
            "weights": s.weights,
        },
        "volume_conductor": {
            "grid": {"nodes": fem.nodes, "elements": fem.tets.astype(np.int64)},
            "tensors": vc_tensors,
        },
        "meg": {"intorderadd": str(s.intorderadd), "type": "physical"},
        # Only set when this process is one of many: DUNEuro reads
        # ``numberOfThreads`` for its TBB arenas and otherwise takes the whole
        # machine, which is right for a serial solve and ruinous for a chunked
        # one. See SolverCfg.threads_per_process.
        **(
            {"numberOfThreads": str(s.threads_per_process)}
            if limit_threads and s.threads_per_process > 0
            else {}
        ),
    }


def build_driver(
    cfg: Config,
    fem: FemMesh,
    *,
    limit_threads: bool = False,
) -> tuple[Any, dict[str, Any], np.ndarray]:
    """Construct a fully-configured ``MEEGDriver3d``.

    Returns ``(driver, driver_cfg, cond)`` so callers can pass ``driver_cfg``
    on to ``computeMEGTransferMatrix`` / ``applyMEGTransfer``.
    """
    dp = import_duneuro(cfg)
    cond = build_conductivity_vector(cfg, fem)
    logger.info("Conductivities (S/mm scaled):")
    for lab, tid in sorted(fem.label_to_id.items(), key=lambda x: x[1]):
        logger.info("  %-11s (id=%d): %.6f", lab, tid, cond[tid - 1])
    aniso_tensors = build_conductivity_tensors(cfg, fem)
    if aniso_tensors is not None:
        a = cfg.forward.muscle_anisotropy
        n_muscle_stl = len(aniso_tensors[1]) - cond.shape[0]
        logger.info(
            "Muscle anisotropy ENABLED: sigma_long=%.3f sigma_trans=%.3f S/m "
            "(%.1f:1) over %d muscle STLs (%d muscle tets)",
            a.sigma_long_sm,
            a.sigma_trans_sm,
            a.sigma_long_sm / max(a.sigma_trans_sm, 1e-12),
            n_muscle_stl,
            int((fem.tissue == fem.label_to_id["muscle"]).sum()),
        )
    driver_cfg = build_driver_config(
        cfg, fem, cond, aniso_tensors=aniso_tensors, limit_threads=limit_threads
    )
    driver = dp.MEEGDriver3d(driver_cfg)
    return driver, driver_cfg, cond


def attach_coils(
    driver: Any,
    dp: Any,
    coilpos: np.ndarray,
    coilori: np.ndarray,
) -> None:
    """Attach coils + projections (one orientation per channel) to ``driver``."""
    coils_du = [dp.FieldVector3D(p) for p in coilpos]
    projs_du = [[dp.FieldVector3D(o)] for o in coilori]
    driver.setCoilsAndProjections(coils_du, projs_du)


def build_orthogonal_dipoles(
    dp: Any,
    src_pos_mm: np.ndarray,
) -> list[Any]:
    """Three orthogonal dipoles (X, Y, Z moments) per source position."""
    eye3 = np.eye(3)
    dipoles: list[Any] = []
    for p in src_pos_mm:
        for k in range(3):
            dipoles.append(dp.Dipole3d(p, eye3[k]))
    return dipoles


# DUNEuro is run in mm-mode (mesh coordinates in mm, conductivities in S/mm).
# Its MEG core computes the raw Biot–Savart kernel from mesh coordinates and
# applies the SI prefactor μ0/4π = 1e-7 T·m/A outside the integral, so the
# magnetic field it returns is a fixed factor larger than SI Tesla per (A·m).
# Because the field scales as length^-2 (the (r)/|r|^3 kernel integrated over a
# fixed dipole moment), the mm/m convention introduces a single geometry- and
# conductivity-independent constant. Measured against the Sarvas analytic sphere
# it is exactly 10.0 to machine precision (see
# inob.analysis.sphere_calibration.calibrate_meg_factor), so the raw field is
# converted to SI by multiplying by 0.1.
MEG_MM_MODE_TO_SI: float = 0.1


def compute_meg_leadfield(
    driver: Any,
    transfer_matrix: np.ndarray,
    dipoles_du: list[Any],
    driver_cfg: dict[str, Any],
) -> np.ndarray:
    """Full MEG leadfield in SI units (Tesla per A·m), (n_coils, n_dipoles).

    The DUNEuro MEG transfer matrix yields ONLY the secondary (volume-current)
    field. The primary Biot–Savart field of the source itself must be added
    separately via ``computeMEGPrimaryField`` — it is not part of the transfer
    result and ``post_process_meg`` does not add it. Omitting it makes the
    leadfield collapse to ~zero for sources radial to a spherical conductor
    (where the entire signal is primary) and wrong everywhere else.

    Both contributions are returned in DUNEuro mm-mode units; their sum is
    rescaled to SI by :data:`MEG_MM_MODE_TO_SI`. ``driver_cfg`` must already
    carry the ``source_model`` entry.
    """
    secondary_raw, _ = driver.applyMEGTransfer(transfer_matrix, dipoles_du, driver_cfg)
    primary_raw = driver.computeMEGPrimaryField(dipoles_du, driver_cfg)
    secondary = np.column_stack([np.asarray(f) for f in secondary_raw])
    primary = np.column_stack([np.asarray(f) for f in primary_raw])
    return (secondary + primary) * MEG_MM_MODE_TO_SI
