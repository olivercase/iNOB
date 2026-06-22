"""Shared DUNEuro driver setup.

Encapsulates the pieces every forward path needs: import the duneuropy
extension (with an optional ``cfg.forward.duneuro_path`` shim for legacy
absolute-path installs), build the conductivity vector from the YAML, and
construct the ``MEEGDriver3d`` config dict — once, in one place.

Used by:
  * :mod:`vagus_fm.forward.solve` — single-machine forward solve
  * :mod:`vagus_fm.forward.chunk` — cluster array-job worker
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from vagus_fm.config import Config
from vagus_fm.io.hdf5 import FemMesh

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
    cfg: Config, fem: FemMesh,
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
        raise KeyError(
            f"forward.conductivities_sm missing entries for tissues: {missing}"
        )
    cond_size = int(np.unique(fem.tissue).max())
    cond = np.zeros(cond_size, dtype=np.float64)
    for lab, tid in label_to_id.items():
        cond[tid - 1] = cfg.forward.conductivities_sm[lab] * cfg.forward.sigma_unit_scale
    return cond


def build_driver_config(
    cfg: Config, fem: FemMesh, cond: np.ndarray,
) -> dict[str, Any]:
    """Return the MEEGDriver3d configuration dictionary."""
    s = cfg.forward.solver
    tissue0 = (fem.tissue.astype(np.int64) - 1)
    return {
        "type":             "fitted",
        "solver_type":      s.type,
        "element_type":     "tetrahedron",
        "post_process":     "false",
        "post_process_meg": str(s.post_process_meg).lower(),
        "subtract_mean":    str(s.subtract_mean).lower(),
        "solver": {
            "reduction":      str(s.reduction),
            "edge_norm_type": s.edge_norm_type,
            "penalty":        str(s.penalty),
            "scheme":         s.scheme,
            "weights":        s.weights,
        },
        "volume_conductor": {
            "grid":    {"nodes": fem.nodes, "elements": fem.tets.astype(np.int64)},
            "tensors": {"labels": tissue0, "conductivities": cond},
        },
        "meg": {"intorderadd": str(s.intorderadd), "type": "physical"},
    }


def build_driver(cfg: Config, fem: FemMesh) -> tuple[Any, dict[str, Any], np.ndarray]:
    """Construct a fully-configured ``MEEGDriver3d``.

    Returns ``(driver, driver_cfg, cond)`` so callers can pass ``driver_cfg``
    on to ``computeMEGTransferMatrix`` / ``applyMEGTransfer``.
    """
    dp = import_duneuro(cfg)
    cond = build_conductivity_vector(cfg, fem)
    logger.info("Conductivities (S/mm scaled):")
    for lab, tid in sorted(fem.label_to_id.items(), key=lambda x: x[1]):
        logger.info("  %-11s (id=%d): %.6f", lab, tid, cond[tid - 1])
    driver_cfg = build_driver_config(cfg, fem, cond)
    driver = dp.MEEGDriver3d(driver_cfg)
    return driver, driver_cfg, cond


def attach_coils(
    driver: Any, dp: Any, coilpos: np.ndarray, coilori: np.ndarray,
) -> None:
    """Attach coils + projections (one orientation per channel) to ``driver``."""
    coils_du = [dp.FieldVector3D(p) for p in coilpos]
    projs_du = [[dp.FieldVector3D(o)] for o in coilori]
    driver.setCoilsAndProjections(coils_du, projs_du)


def build_orthogonal_dipoles(
    dp: Any, src_pos_mm: np.ndarray,
) -> list[Any]:
    """Three orthogonal dipoles (X, Y, Z moments) per source position."""
    eye3 = np.eye(3)
    dipoles: list[Any] = []
    for p in src_pos_mm:
        for k in range(3):
            dipoles.append(dp.Dipole3d(p, eye3[k]))
    return dipoles
