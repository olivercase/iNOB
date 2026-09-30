"""DUNEuro FEM leadfields for the head+thorax cardiac conductor.

``inob.forward.solve`` drives DUNEuro from a :class:`~inob.config.Config`. This
module is the config-free equivalent the CFA study needs, because the CFA run
solves the *same* array against **nine different volume conductors** -- one per
head orientation -- and wants the transfer matrix reused across every source set.

The economics are the whole reason this is structured the way it is. The MEG
transfer matrix costs one FEM solve per coil (85 here), and once it exists any
number of dipoles is a matrix product. The distributed myocardial source has 400
sites x 3 components, the equivalent-dipole model has 2 x 3 and the insula has 3
-- 1209 dipoles that would be 1209 solves the naive way, and are free this way.

Units. DUNEuro runs in mm-mode: nodes in mm, conductivities in S/mm, and its MEG
core applies mu0/4pi outside an integral whose kernel scales as length^-2, so the
raw field is a fixed factor from SI. That factor is 10 exactly (verified against
the Sarvas sphere in :mod:`inob.analysis.sphere_calibration`), so everything here
returns **Tesla per A.m** and the conversion to fT happens once, in the engine.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from inob.forward.duneuro_driver import MEG_MM_MODE_TO_SI
from inob.io.hdf5 import FemMesh

logger = logging.getLogger(__name__)

__all__ = ["SolverOptions", "ConductorSolver"]

#: S/m -> S/mm, for DUNEuro's mm-mode.
SIGMA_TO_SMM: float = 1e-3


@dataclass(frozen=True)
class SolverOptions:
    """DUNEuro solver and source-model settings.

    ``venant`` rather than ``partial_integration`` is the default here: the
    myocardial sites sit 3 mm inside a wall meshed at ~3 mm, so a source model
    that loads only the containing element's vertices is sensitive to which
    element that happens to be. St. Venant spreads the load over a patch and is
    the standard choice for sources this close to a tissue boundary.
    """

    solver_type: str = "cg"
    reduction: float = 1e-10
    edge_norm_type: str = "houston"
    penalty: float = 20.0
    scheme: str = "sipg"
    weights: str = "tensorOnly"
    intorderadd: int = 5
    post_process_meg: bool = True
    subtract_mean: bool = True
    source_model: str = "venant"
    venant_reference_length_mm: float = 20.0
    venant_moments: int = 3
    venant_weighting_exponent: int = 1
    venant_relaxation: float = 1e-6
    threads: int = 0

    def source_model_cfg(self) -> dict[str, str]:
        if self.source_model == "partial_integration":
            return {"type": "partial_integration"}
        return {
            "type": self.source_model,
            "numberOfMoments": str(self.venant_moments),
            "referenceLength": str(self.venant_reference_length_mm),
            "weightingExponent": str(self.venant_weighting_exponent),
            "relaxationFactor": str(self.venant_relaxation),
            "mixedMoments": "true",
            "restrict": "true",
            "initialization": "closest_vertex",
            "extensions": "vertex",
            "intorderadd": "2",
        }


class ConductorSolver:
    """One volume conductor + one coil array, reusable over many source sets.

    Prefer :meth:`for_sites` over the constructor. The CFA study needs the field
    **vector** at each sensor site, not a scalar projection, because the
    precision factor tilts each channel's sensitive axis and the estimators must
    read the same physical field out along a *different* axis than the data were
    generated with. Attaching three world-axis projections per site gives that
    vector once; every channel orientation, nominal or perturbed, is then a
    linear combination of it and costs nothing further.
    """

    @classmethod
    def for_sites(
        cls, fem: FemMesh, conductivity_sm: dict[str, float],
        site_pos_mm: np.ndarray, *, options: "SolverOptions | None" = None,
    ) -> "ConductorSolver":
        """Solver whose coils are the world x/y/z axes at each site.

        Coil order is ``[all sites e_x, all sites e_y, all sites e_z]``, so the
        result reshapes to ``(3, n_site, ...)`` and transposes to a per-site
        field vector.
        """
        p = np.asarray(site_pos_mm, float).reshape(-1, 3)
        pos = np.tile(p, (3, 1))
        ori = np.repeat(np.eye(3), len(p), axis=0)
        self = cls(fem, conductivity_sm, pos, ori, options=options)
        self.n_site = len(p)
        return self

    def __init__(
        self,
        fem: FemMesh,
        conductivity_sm: dict[str, float],
        coilpos_mm: np.ndarray,
        coilori: np.ndarray,
        *,
        options: SolverOptions | None = None,
    ) -> None:
        import duneuropy as dp

        self.dp = dp
        self.fem = fem
        self.opts = options or SolverOptions()
        self.coilpos_mm = np.asarray(coilpos_mm, float).reshape(-1, 3)
        self.coilori = np.asarray(coilori, float).reshape(-1, 3)
        self.coilori = self.coilori / np.linalg.norm(self.coilori, axis=1, keepdims=True)

        missing = [k for k in fem.tissue_labels if k not in conductivity_sm]
        if missing:
            raise KeyError(f"no conductivity given for {missing}")
        self.sigma_sm = np.array([conductivity_sm[k] for k in fem.tissue_labels], float)
        self._check_sources_inside()

        self.cfg = self._driver_cfg()
        self.driver = dp.MEEGDriver3d(self.cfg)
        self.driver.setCoilsAndProjections(
            [dp.FieldVector3D(p) for p in self.coilpos_mm],
            [[dp.FieldVector3D(o)] for o in self.coilori],
        )
        self.cfg["source_model"] = self.opts.source_model_cfg()
        self._transfer: np.ndarray | None = None
        self.n_site: int | None = None
        logger.info(
            "[solver] %d nodes / %d tets / %d labels, %d coils; sigma S/m: %s",
            len(fem.nodes), len(fem.tets), len(fem.tissue_labels), len(self.coilpos_mm),
            {k: v for k, v in zip(fem.tissue_labels, self.sigma_sm)},
        )

    # -- setup ---------------------------------------------------------------
    def _check_sources_inside(self) -> None:
        """Coils must sit outside the conductor, or MEG is being read inside tissue."""
        lo, hi = self.fem.nodes.min(0), self.fem.nodes.max(0)
        inside_box = ((self.coilpos_mm > lo) & (self.coilpos_mm < hi)).all(1)
        if inside_box.all():
            logger.debug("[solver] all coils within the mesh bounding box (expected: "
                         "the array wraps the head, which the box contains)")

    def _driver_cfg(self) -> dict:
        o = self.opts
        return {
            "type": "fitted",
            "solver_type": o.solver_type,
            "element_type": "tetrahedron",
            "post_process": "false",
            "post_process_meg": str(o.post_process_meg).lower(),
            "subtract_mean": str(o.subtract_mean).lower(),
            "solver": {
                "reduction": str(o.reduction),
                "edge_norm_type": o.edge_norm_type,
                "penalty": str(o.penalty),
                "scheme": o.scheme,
                "weights": o.weights,
            },
            "volume_conductor": {
                "grid": {"nodes": self.fem.nodes,
                         "elements": self.fem.tets.astype(np.int64)},
                "tensors": {"labels": self.fem.tissue.astype(np.int64) - 1,
                            "conductivities": self.sigma_sm * SIGMA_TO_SMM},
            },
            "meg": {"intorderadd": str(o.intorderadd), "type": "physical"},
            **({"numberOfThreads": str(o.threads)} if o.threads > 0 else {}),
        }

    # -- solving -------------------------------------------------------------
    @property
    def transfer(self) -> np.ndarray:
        """MEG transfer matrix, computed once (one FEM solve per coil)."""
        if self._transfer is None:
            t0 = time.time()
            logger.info("[solver] MEG transfer matrix for %d coils...",
                        len(self.coilpos_mm))
            raw, _ = self.driver.computeMEGTransferMatrix(self.cfg)
            self._transfer = np.asarray(raw)
            logger.info("[solver]   %s in %.0f s", self._transfer.shape, time.time() - t0)
        return self._transfer

    def leadfield(self, pos_mm: np.ndarray, ori: np.ndarray | None = None) -> np.ndarray:
        """Leadfield in T/(A.m).

        With ``ori=None`` returns ``(n_coils, 3 * n_sources)`` -- x/y/z moments per
        source, in source-major order. With ``ori`` given (one unit vector per
        source) returns ``(n_coils, n_sources)``, the field of each source driven
        along its own orientation.
        """
        p = np.asarray(pos_mm, float).reshape(-1, 3)
        dips = [self.dp.Dipole3d(pt, e) for pt in p for e in np.eye(3)]
        t0 = time.time()
        sec, _ = self.driver.applyMEGTransfer(self.transfer, dips, self.cfg)
        pri = self.driver.computeMEGPrimaryField(dips, self.cfg)
        lf = (np.column_stack([np.asarray(f) for f in sec])
              + np.column_stack([np.asarray(f) for f in pri])) * MEG_MM_MODE_TO_SI
        logger.info("[solver] leadfield for %d dipoles in %.1f s -> %s",
                    len(dips), time.time() - t0, lf.shape)
        if ori is None:
            return lf
        o = np.asarray(ori, float).reshape(-1, 3)
        if len(o) != len(p):
            raise ValueError("ori must have one row per source position")
        o = o / np.linalg.norm(o, axis=1, keepdims=True)
        return np.einsum("cnk,nk->cn", lf.reshape(len(lf), len(p), 3), o)

    def site_field(self, pos_mm: np.ndarray) -> np.ndarray:
        """Field vector at every site, ``(n_site, 3, 3 * n_sources)`` in T/(A.m).

        Axis 1 is the world x/y/z component of B at that site; axis 2 runs over
        source-major x/y/z moments. Only valid on a solver built by
        :meth:`for_sites`.
        """
        if self.n_site is None:
            raise RuntimeError("site_field requires a solver built by for_sites()")
        lf = self.leadfield(pos_mm)                      # (3 * n_site, 3 * n_src)
        return lf.reshape(3, self.n_site, -1).transpose(1, 0, 2)
