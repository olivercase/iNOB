"""Centralised logging configuration.

Use ``configure_logging()`` once at the start of every CLI entry point.
Modules use ``logger = logging.getLogger(__name__)`` and never call ``print()``.

Loading the config is itself a source of warnings, and it has to happen before
``configure_logging`` can run — the config is what says where the log file
goes. Call :func:`begin_capture` first: it puts the console handler in place
immediately, so those early warnings are formatted like every other line, and
buffers them until the file handler exists so they are in the log too.
"""
from __future__ import annotations

import logging
import logging.handlers
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

    pending = _take_buffer(root)
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
        # Replay whatever begin_capture() held onto. The console has already
        # shown these; this is what puts them in the file, so a log is a
        # complete record of the run rather than one that starts after the
        # config was read.
        for record in pending:
            if record.levelno >= fh.level:
                fh.handle(record)

    return root


def begin_capture(level: int | str = "INFO") -> None:
    """Start logging to the console now, and hold the records for the log file.

    Idempotent, and safe to call when logging is already configured: it does
    nothing once a file handler of ours is installed.
    """
    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) and getattr(h, "_inob", False)
           for h in root.handlers):
        return
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    root.setLevel(level)
    for h in [h for h in root.handlers if getattr(h, "_inob", False)]:
        root.removeHandler(h)

    formatter = logging.Formatter(fmt=DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(formatter)
    sh.setLevel(level)
    sh._inob = True   # type: ignore[attr-defined]
    root.addHandler(sh)

    # capacity high enough that config loading never flushes it, and no target,
    # so nothing is emitted twice.
    buf = logging.handlers.MemoryHandler(capacity=10_000, flushLevel=logging.CRITICAL + 1)
    buf.setLevel(level)
    buf._inob = True        # type: ignore[attr-defined]
    buf._inob_buffer = True  # type: ignore[attr-defined]
    root.addHandler(buf)


def _take_buffer(root: logging.Logger) -> list[logging.LogRecord]:
    """Drain and detach the handler begin_capture() left behind."""
    records: list[logging.LogRecord] = []
    for h in [h for h in root.handlers if getattr(h, "_inob_buffer", False)]:
        records.extend(getattr(h, "buffer", []))
        h.buffer = []       # type: ignore[attr-defined]
        root.removeHandler(h)
        h.close()
    return records


def run_log_path(logs_dir: Path, prefix: str = "run") -> Path:
    """Generate a timestamped log file path inside ``logs_dir``."""
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return logs_dir / f"{prefix}-{ts}.log"
