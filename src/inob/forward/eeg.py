"""DUNEuro EEG forward solve (surface-potential leadfield).

The companion of :mod:`inob.forward.solve` (MEG). Same FEM, same source
dipoles — different driver path. DUNEuro's ``MEEGDriver3d`` exposes an EEG
transfer-matrix API (``computeEEGTransferMatrix`` / ``applyEEGTransfer``)
that yields the surface-potential leadfield in V/(A·m).

We use a common-average reference: the per-channel mean across electrodes
is subtracted from each leadfield column. This matches the behaviour of an
HD-EMG-style PEDOT:PSS array with a wire-shorted reference contact, and
side-steps the ill-defined choice of a single reference electrode.

Note that this choice is invisible to any *differential* readout: a potential
difference between two contacts is unchanged by subtracting a common per-source
constant from every channel. So the bipolar amplitudes reported by
:func:`inob.analysis.snr.per_source_best_bipolar` are reference-free, and can
be read off an existing leadfield without re-solving.

Outputs ``cfg.outputs.forward_eeg_npz``, schema-identical to the MEG NPZ
produced by :func:`inob.forward.solve.run_forward` with the magnetic
fields replaced by surface potentials (units V/(A·m), µV/(nA·m) for the
human-friendly variant).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from inob.config import Config
from inob.forward.duneuro_driver import (
    build_driver,
    build_orthogonal_dipoles,
    build_source_model_config,
    import_duneuro,
)
from inob.io.hdf5 import load_fem, load_sensors, validate_fem, validate_sensors
from inob.io.npz import Leadfield, save_leadfield, validate_leadfield
from inob.sources.vagus import resolve_source_positions

logger = logging.getLogger(__name__)


# Empirical conversion factor that turns DUNEuro mm-mode raw EEG output
# (V at internal mm-units) into µV per (1 nA·m source moment). Calibrated
# against the Berg-Scherg analytic series on a homogeneous-sphere FEM by
# :mod:`inob.analysis.sphere_calibration`. See ``outputs/calibration/
# calibration.json`` and ``docs/VALIDATION.md`` §2 for the derivation.
#
# Median over 27-case (R, depth, σ) sweep is 0.622 with CV = 6.8%, and is
# σ-invariant (Berg-Scherg's 1/σ and DUNEuro's σ scaling cancel exactly).
EEG_CALIBRATION_FACTOR: float = 0.622


def attach_electrodes(
    driver, dp, electrode_pos: np.ndarray, *, electrode_type: str = "closest_subentity_center",
) -> None:
    """Attach surface electrodes to a DUNEuro driver.

    DUNEuro's ``setElectrodes`` expects a config dict alongside the positions
    selecting how the contacts are realised on the FEM mesh. ``"closest_subentity_center"``
    snaps each position to the centre of the nearest boundary face — robust
    for HD-EMG-style contacts that need to sit ON the skin.
    """
    elec_du = [dp.FieldVector3D(p) for p in electrode_pos]
    driver.setElectrodes(elec_du, {"type": electrode_type, "codims": "1"})


def compute_eeg_leadfield(
    driver, dp, driver_cfg: dict, src_pos_mm: np.ndarray,
    source_model_cfg: dict | None = None,
) -> np.ndarray:
    """Compute the EEG leadfield using DUNEuro's transfer-matrix path.

    Returns ``(n_electrodes, 3*n_src)`` of V per A·m, common-average
    referenced.

    ``source_model_cfg`` is the DUNEuro ``source_model`` sub-config (see
    :func:`inob.forward.duneuro_driver.build_source_model_config`); it is
    needed only for the *apply* step, not for the transfer matrix. Defaults to
    partial integration so callers without a :class:`Config` keep the historic
    behaviour.
    """
    source_model_cfg = source_model_cfg or {"type": "partial_integration"}
    logger.info("Computing EEG transfer matrix (slow)…")
    t0 = time.time()
    T_raw, _ = driver.computeEEGTransferMatrix(driver_cfg)
    T = np.array(T_raw)
    logger.info("  T: %s (%.0f s)", T.shape, time.time() - t0)

    dipoles_du = build_orthogonal_dipoles(dp, src_pos_mm)
    driver_cfg = {**driver_cfg, "source_model": source_model_cfg}

    logger.info("Applying transfer to %d dipoles (3 per source)…", len(dipoles_du))
    t0 = time.time()
    fields_raw, _ = driver.applyEEGTransfer(T, dipoles_du, driver_cfg)
    L = np.column_stack([np.asarray(f) for f in fields_raw])
    logger.info("  L: %s (%.0f s)", L.shape, time.time() - t0)

    # Common-average reference: subtract per-column mean across electrodes.
    L = L - L.mean(axis=0, keepdims=True)
    return L


def run_eeg_forward(cfg: Config) -> Path:
    """Run the local EEG forward solve and write the leadfield NPZ."""
    fem = load_fem(cfg.outputs.fem_mat)
    validate_fem(fem)
    logger.info("FEM: %d nodes, %d tets, labels=%s",
                len(fem.nodes), len(fem.tets), list(fem.tissue_labels))

    electrodes = load_sensors(cfg.outputs.electrodes_mat)
    validate_sensors(electrodes)
    logger.info("Electrodes: %d HD contacts", len(electrodes.coilpos))

    src_pos_mm = resolve_source_positions(cfg, fem)
    logger.info("%d sources (z=%.0f..%.0f)",
                len(src_pos_mm), src_pos_mm[:, 2].min(), src_pos_mm[:, 2].max())

    dp = import_duneuro(cfg)
    driver, driver_cfg, cond = build_driver(cfg, fem)
    attach_electrodes(driver, dp, electrodes.coilpos)

    logger.info("Source model: %s", cfg.forward.source_model.type)
    L = compute_eeg_leadfield(driver, dp, driver_cfg, src_pos_mm,
                              build_source_model_config(cfg))
    # DUNEuro mm-mode EEG → µV/(nA·m): ×EEG_CALIBRATION_FACTOR (see constant
    # above; derivation in docs/VALIDATION.md §2).
    L_uV_per_nAm = L * EEG_CALIBRATION_FACTOR
    logger.info(
        "µV/(nA·m): min=%.3e max=%.3e rms=%.3e",
        L_uV_per_nAm.min(), L_uV_per_nAm.max(),
        np.sqrt(np.mean(L_uV_per_nAm ** 2)),
    )

    lf = Leadfield(
        L=L,
        L_fT_per_nAm=L_uV_per_nAm,    # reuse field; semantics noted below
        source_pos=src_pos_mm,
        coil_pos=electrodes.coilpos,
        coil_orient=electrodes.coilori,
        channel_names=tuple(electrodes.labels),
        conductivities=cond,
        tissue_labels=tuple(fem.tissue_labels),
        seed=cfg.reproducibility.seed,
    )
    # NB: ``L_fT_per_nAm`` is re-used as the human-friendly leadfield slot;
    # for EEG outputs the units are µV/(nA·m). The schema validator only
    # checks finiteness + amplitude bound, so this is safe.
    validate_leadfield(
        lf,
        require_finite=cfg.forward.validate.require_finite,
        max_abs_fT_per_nAm=cfg.forward.validate.max_abs_fT_per_nAm,
    )
    save_leadfield(cfg.outputs.forward_eeg_npz, lf)
    return cfg.outputs.forward_eeg_npz
