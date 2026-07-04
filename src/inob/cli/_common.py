"""Shared CLI plumbing: argparse fragments, config loading, logging bootstrap."""
from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np

from inob.config import Config, load_config
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
    log_path = args.log_file or run_log_path(cfg.outputs.logs_dir, prefix=log_prefix)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_level, log_path)
    logger.info("inob %s | config=%s | project_root=%s | log=%s",
                log_prefix, args.config, cfg.project_root, log_path)
    np.random.seed(cfg.reproducibility.seed)
    random.seed(cfg.reproducibility.seed)
    return cfg
