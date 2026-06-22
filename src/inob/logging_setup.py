"""Centralised logging configuration.

Use ``configure_logging()`` once at the start of every CLI entry point.
Modules use ``logger = logging.getLogger(__name__)`` and never call ``print()``.
"""
from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(
    level: int | str = "INFO",
    log_path: Path | None = None,
    *,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
) -> logging.Logger:
    """Install a stream handler and (optionally) a file handler on the root logger.

    Idempotent: removes any handlers we previously installed, leaving foreign ones alone.
    """
    root = logging.getLogger()
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    root.setLevel(level)

    for h in [h for h in root.handlers if getattr(h, "_inob", False)]:
        root.removeHandler(h)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(formatter)
    sh.setLevel(level)
    sh._inob = True   # type: ignore[attr-defined]
    root.addHandler(sh)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        fh._inob = True   # type: ignore[attr-defined]
        root.addHandler(fh)

    return root


def run_log_path(logs_dir: Path, prefix: str = "run") -> Path:
    """Generate a timestamped log file path inside ``logs_dir``."""
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return logs_dir / f"{prefix}-{ts}.log"
