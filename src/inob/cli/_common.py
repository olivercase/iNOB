"""Shared CLI plumbing: argparse fragments, config loading, logging bootstrap."""
from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from inob.config import (
    SOLVER_TYPES,
    SOURCE_MODEL_TYPES,
    SOURCE_TARGETS,
    Config,
    load_config,
    tag_path,
)
from inob.logging_setup import configure_logging, run_log_path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/default.yaml"


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Attach ``--config / --set / --project-root / --log-level`` to a parser."""
    p.add_argument(
        "--config", type=Path, default=Path(DEFAULT_CONFIG),
        help=f"Path to YAML config (default {DEFAULT_CONFIG}).",
    )
    p.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VAL",
        help="Override a config field, e.g. --set forward.source_spacing_mm=3.0. "
             "Repeatable.",
    )
    p.add_argument(
        "--source-target", default=None, metavar="TAG", choices=list(SOURCE_TARGETS),
        help="Target a specific source region. Repoints outputs.forward_npz / "
             "forward_eeg_npz to duneuro_leadfield_<TAG>.npz and sets "
             "forward.source_tissue to that region's tissue(s), so solve and "
             f"analysis stay consistent. One of: {', '.join(SOURCE_TARGETS)}. "
             "Mirrors the cluster SOURCE_TARGET.",
    )
    p.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="CPU workers for the local forward solve (0 = every core, the "
             "default). The sensor array is split into N chunks solved in "
             "parallel, one process each. Same control the GUI's Compute "
             "panel offers; equivalent to --set forward.local_workers=N.",
    )
    p.add_argument(
        "--source-model", default=None, metavar="MODEL",
        choices=list(SOURCE_MODEL_TYPES),
        help="How a point dipole becomes a FEM right-hand side. "
             "'partial_integration' (default) loads only the containing "
             "element's nodes; 'venant' / 'multipolar_venant' spread it over a "
             "patch of neighbouring nodes fitted to the dipole moment (St. "
             "Venant), which behaves better near a conductivity jump. "
             "Equivalent to --set forward.source_model.type=MODEL; the fit "
             "parameters stay on --set. "
             f"One of: {', '.join(SOURCE_MODEL_TYPES)}.",
    )
    p.add_argument(
        "--solver-type", default=None, metavar="TYPE", choices=list(SOLVER_TYPES),
        help="FEM discretisation: 'cg' (default, continuous — what every "
             "leadfield here was solved with) or 'dg' (discontinuous Galerkin, "
             "which represents a conductivity jump as a jump instead of "
             "smearing it across the elements either side, at ~4x the degrees "
             "of freedom). DG works only with the partial-integration source "
             "model. Equivalent to --set forward.solver.type=TYPE.",
    )
    p.add_argument(
        "--muscle-anisotropy", default=None, metavar="MODE",
        choices=("auto", "on", "off"),
        help="Fibre-aligned muscle conductivity tensor. 'auto' (default) turns "
             "it on exactly when muscle is a source tissue, keeping vagus and "
             "spine runs comparable with ones already solved; force 'on'/'off' "
             "for a like-for-like A/B. Equivalent to "
             "--set forward.muscle_anisotropy.mode=MODE.",
    )
    p.add_argument(
        "--project-root", type=Path, default=None,
        help="Override the project root used to resolve relative paths.",
    )
    p.add_argument(
        "--log-level", default=os.environ.get("INOB_LOG", "INFO"),
        help="Logging level (DEBUG/INFO/WARNING/ERROR; default INFO).",
    )
    p.add_argument(
        "--log-file", type=Path, default=None,
        help="Log file path (default: outputs/logs/<stage>-<UTC>.log).",
    )


def apply_source_target(cfg: Config, target: str) -> Config:
    """Repoint every per-target output at ``target`` and set its source tissue.

    Lives on its own so that anything reading the same run's artefacts —
    ``setup`` for the stages that write them, ``inob status`` for the report
    that lists them — resolves the identical set of paths. When this was inline
    in ``setup``, ``status`` had no way to reach it, so a spine run was reported
    against the vagus filenames and looked unbuilt.
    """

    # argparse ``choices`` already rejects unknown slugs, but guard here too
    # so programmatic callers get the same clear error.
    if target not in SOURCE_TARGETS:
        raise SystemExit(
            f"--source-target {target!r} is not recognised. "
            f"Use one of: {', '.join(SOURCE_TARGETS)}"
        )
    spec = SOURCE_TARGETS[target]
    tissues = spec["tissues"]
    elec_tissue = spec["electrodes"]
    # Optional vertebral level to centre the electrode patch on (spine → c7),
    # so the paddle sits over the source of interest rather than the mid-cord
    # slab mean. None for targets without a vertebral level (vagus, muscle).
    elec_level = spec.get("level")
    # Repoint both leadfields to the tagged files (same dir as the config
    # defaults, e.g. outputs/forward/duneuro_leadfield_<target>.npz) AND set
    # the source tissue, so a solve writes the region it names and an
    # analysis reads the region it labels — no manual --set to keep in sync.
    fwd_dir = cfg.outputs.forward_npz.parent
    # The HD electrode patch is target-specific too (see SOURCE_TARGETS), so
    # its array file is tagged the same way. Without the tag a spine patch
    # would overwrite the vagus one at the shared default path and the next
    # EEG solve would silently use whichever array was written last.
    elec_tagged = tag_path(cfg.outputs.electrodes_mat, target)
    elec_png_tagged = tag_path(cfg.outputs.electrodes_png, target)
    # Chunks and the conductivity-sensitivity sweep are per-target too: both
    # are derived from this target's sources, so sharing a directory would
    # let one target's run consume or overwrite another's intermediates.
    # Geometry, FEM and the OPM array are deliberately NOT tagged — they
    # contain every tissue and are shared by every target by design.
    cfg = replace(
        cfg,
        outputs=replace(
            cfg.outputs,
            forward_npz=fwd_dir / f"duneuro_leadfield_{target}.npz",
            forward_eeg_npz=fwd_dir / f"duneuro_eeg_leadfield_{target}.npz",
            electrodes_mat=elec_tagged,
            electrodes_png=elec_png_tagged,
            forward_chunks_dir=tag_path(cfg.outputs.forward_chunks_dir, target),
            sensitivity_dir=tag_path(cfg.outputs.sensitivity_dir, target),
        ),
        forward=replace(cfg.forward, source_tissue=tissues),
        electrodes=replace(cfg.electrodes, target_tissue=elec_tissue,
                           target_level=elec_level),
    )
    logger.info("source-target=%s → tissues=%s → leadfield %s",
                target, tissues, cfg.outputs.forward_npz)
    logger.info("source-target=%s → electrode patch over %s → %s",
                target, elec_tissue, cfg.outputs.electrodes_mat)
    return cfg


def setup(
    args: argparse.Namespace, *, log_prefix: str = "run",
) -> Config:
    """Load + validate the config and configure logging.

    Picks a default log file under ``cfg.outputs.logs_dir`` if the user did
    not pass ``--log-file``. Seeds numpy + random with ``cfg.reproducibility.seed``.
    """
    # --workers is sugar for the config field, applied as an override so it
    # follows exactly the same validation path as --set.
    overrides = list(args.overrides or [])
    workers = getattr(args, "workers", None)
    if workers is not None:
        if workers < 0:
            raise SystemExit("--workers must be 0 (all cores) or a positive count")
        overrides.append(f"forward.local_workers={workers}")
    # Same sugar for the two forward-physics choices. Both go through --set so
    # the config loader validates them exactly once, in one place.
    source_model = getattr(args, "source_model", None)
    if source_model:
        overrides.append(f"forward.source_model.type={source_model}")
    solver_type = getattr(args, "solver_type", None)
    if solver_type:
        overrides.append(f"forward.solver.type={solver_type}")
    muscle_aniso = getattr(args, "muscle_anisotropy", None)
    if muscle_aniso:
        overrides.append(f"forward.muscle_anisotropy.mode={muscle_aniso}")

    cfg = load_config(
        args.config, overrides=overrides, project_root=args.project_root,
    )
    target = getattr(args, "source_target", None)
    if target:
        cfg = apply_source_target(cfg, target)
    log_path = args.log_file or run_log_path(cfg.outputs.logs_dir, prefix=log_prefix)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_level, log_path)
    logger.info("inob %s | config=%s | project_root=%s | log=%s",
                log_prefix, args.config, cfg.project_root, log_path)
    np.random.seed(cfg.reproducibility.seed)
    random.seed(cfg.reproducibility.seed)
    return cfg
