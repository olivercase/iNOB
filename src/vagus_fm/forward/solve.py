"""Single-machine DUNEuro forward solve.

Builds the FEM driver from ``cfg``, samples vagus dipoles, computes the MEG
transfer matrix, applies it to three orthogonal dipole moments per source,
and writes ``cfg.outputs.forward_npz`` with a schema validated by
:mod:`vagus_fm.io.npz`.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from vagus_fm.config import Config
from vagus_fm.forward.duneuro_driver import (
    attach_coils,
    build_driver,
    build_orthogonal_dipoles,
    import_duneuro,
)
from vagus_fm.io.hdf5 import load_fem, load_sensors, validate_fem, validate_sensors
from vagus_fm.io.npz import Leadfield, save_leadfield, validate_leadfield
from vagus_fm.sources.vagus import vagus_sources

logger = logging.getLogger(__name__)


def run_forward(cfg: Config) -> Path:
    """Run the local forward solve and write the leadfield NPZ.

    Returns the absolute output path. Validates inputs (FEM + sensor) and
    output (NPZ schema) so a corrupted run never silently produces a bad
    artefact.
    """
    fem = load_fem(cfg.outputs.fem_mat)
    validate_fem(fem)
    logger.info("FEM: %d nodes, %d tets, labels=%s",
                len(fem.nodes), len(fem.tets), list(fem.tissue_labels))

    sensors = load_sensors(cfg.outputs.sensors_mat)
    validate_sensors(sensors)
    logger.info("Sensors: %d channels (%d positions × 3 axes)",
                len(sensors.coilpos), len(sensors.coilpos) // 3)

    src_pos_mm = vagus_sources(
        fem, cfg.forward.source_tissue, spacing_mm=cfg.forward.source_spacing_mm,
    )
    n_src = len(src_pos_mm)
    logger.info("%d sources along %s (spacing %g mm); Z=[%.0f, %.0f]",
                n_src, cfg.forward.source_tissue, cfg.forward.source_spacing_mm,
                src_pos_mm[:, 2].min(), src_pos_mm[:, 2].max())

    dp = import_duneuro(cfg)
    driver, driver_cfg, cond = build_driver(cfg, fem)
    attach_coils(driver, dp, sensors.coilpos, sensors.coilori)

    logger.info("Computing MEG transfer matrix (slow)…")
    t0 = time.time()
    T_raw, _ = driver.computeMEGTransferMatrix(driver_cfg)
    T = np.array(T_raw)
    logger.info("  T: %s (%.0f s)", T.shape, time.time() - t0)

    dipoles_du = build_orthogonal_dipoles(dp, src_pos_mm)
    driver_cfg["source_model"] = {"type": "partial_integration"}

    logger.info("Applying transfer to %d dipoles (3 per source)…", len(dipoles_du))
    t0 = time.time()
    fields_raw, _ = driver.applyMEGTransfer(T, dipoles_du, driver_cfg)
    L = np.column_stack([np.asarray(f) for f in fields_raw])
    logger.info("  L: %s (%.0f s)", L.shape, time.time() - t0)

    L_fT_per_nAm = L * 1e6
    logger.info("fT/nAm: min=%.3e max=%.3e rms=%.3e",
                L_fT_per_nAm.min(), L_fT_per_nAm.max(),
                np.sqrt(np.mean(L_fT_per_nAm ** 2)))

    lf = Leadfield(
        L=L,
        L_fT_per_nAm=L_fT_per_nAm,
        source_pos=src_pos_mm,
        coil_pos=sensors.coilpos,
        coil_orient=sensors.coilori,
        channel_names=tuple(sensors.labels),
        conductivities=cond,
        tissue_labels=tuple(fem.tissue_labels),
        seed=cfg.reproducibility.seed,
    )
    validate_leadfield(
        lf,
        require_finite=cfg.forward.validate.require_finite,
        max_abs_fT_per_nAm=cfg.forward.validate.max_abs_fT_per_nAm,
    )
    save_leadfield(cfg.outputs.forward_npz, lf)
    return cfg.outputs.forward_npz
