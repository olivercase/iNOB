"""Shared CLI plumbing: argparse fragments, config loading, logging bootstrap."""
from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from inob.config import SOURCE_TARGETS, Config, load_config, tag_path
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


def setup(
    args: argparse.Namespace, *, log_prefix: str = "run",
) -> Config:
    """Load + validate the config and configure logging.

    Picks a default log file under ``cfg.outputs.logs_dir`` if the user did
    not pass ``--log-file``. Seeds numpy + random with ``cfg.reproducibility.seed``.
    """
    cfg = load_config(
        args.config, overrides=args.overrides, project_root=args.project_root,
    )
    target = getattr(args, "source_target", None)
    if target:
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
    log_path = args.log_file or run_log_path(cfg.outputs.logs_dir, prefix=log_prefix)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_level, log_path)
    logger.info("inob %s | config=%s | project_root=%s | log=%s",
                log_prefix, args.config, cfg.project_root, log_path)
    np.random.seed(cfg.reproducibility.seed)
    random.seed(cfg.reproducibility.seed)
    return cfg
