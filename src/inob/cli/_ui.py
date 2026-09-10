"""Terminal presentation helpers for the ``inob`` command.

Deliberately dependency-free: the package ships no CLI libraries, so this is a
small ANSI layer that degrades to plain text when stdout is not a terminal
(pipes, CI logs, the GUI backend). Honours the ``NO_COLOR`` convention.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sys
import textwrap
import time
from pathlib import Path

_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "cyan": "36",
}


def supports_colour(stream=None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def paint(text: str, *styles: str, stream=None) -> str:
    """Wrap ``text`` in ANSI styles, or return it unchanged without a TTY."""
    if not styles or not supports_colour(stream):
        return text
    codes = ";".join(_CODES[s] for s in styles if s in _CODES)
    return f"\033[{codes}m{text}\033[0m" if codes else text


# Status glyphs. ASCII fallbacks keep Windows terminals and log files readable.
def _glyphs() -> dict[str, str]:
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    if "utf" in encoding:
        return {"ok": "✓", "miss": "·", "fail": "✗", "warn": "!", "arrow": "→"}
    return {"ok": "+", "miss": ".", "fail": "x", "warn": "!", "arrow": "->"}


def mark(kind: str) -> str:
    """A coloured status glyph: ``ok`` / ``miss`` / ``fail`` / ``warn``."""
    glyph = _glyphs()[kind]
    colour = {"ok": "green", "miss": "dim", "fail": "red", "warn": "yellow"}[kind]
    return paint(glyph, colour)


def arrow() -> str:
    return _glyphs()["arrow"]


def heading(text: str) -> str:
    return paint(text, "bold")


def hint(text: str) -> str:
    """A suggested next command, indented and highlighted."""
    return f"  {arrow()} {paint(text, 'cyan')}"


def width(default: int = 80) -> int:
    return shutil.get_terminal_size((default, 24)).columns


def rel(path: Path, root: Path) -> str:
    """Show ``path`` relative to the project root when it sits underneath it."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def human_age(mtime: float, *, now: float | None = None) -> str:
    """'3 minutes ago' style age for a file mtime."""
    delta = max(0.0, (now if now is not None else time.time()) - mtime)
    for limit, divisor, label in (
        (60, 1, "second"),
        (3600, 60, "minute"),
        (86400, 3600, "hour"),
        (604800, 86400, "day"),
    ):
        if delta < limit:
            value = int(delta // divisor)
            if value == 0:
                return "just now"
            return f"1 {label} ago" if value == 1 else f"{value} {label}s ago"
    weeks = int(delta // 604800)
    return "1 week ago" if weeks == 1 else f"{weeks} weeks ago"


def wrap(text: str, indent: str = "", width_hint: int | None = None) -> list[str]:
    """Fold ``text`` to the terminal, indenting every line including the first.

    Report commands lay out fixed-width tables, so a long sentence dropped in
    beside them has to be folded by hand — nothing else wraps it.
    """
    limit = max(40, (width_hint or width()) - len(indent))
    return [indent + line for line in textwrap.wrap(text, limit)] or [indent.rstrip()]


@contextlib.contextmanager
def collect_warnings(source: str = "inob.config"):
    """Yield a list that fills with warnings logged by ``source`` in the block.

    ``doctor`` and ``status`` print a curated report, and a warning logged
    while the config loads would otherwise land on stderr ahead of it, bare and
    unwrapped. Catching them here lets each command place them where they
    belong in its own layout.
    """
    messages: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Collector(level=logging.WARNING)
    logger = logging.getLogger(source)
    logger.addHandler(handler)
    # Stop them reaching the root handlers (or logging's lastResort) as well:
    # the caller has taken responsibility for showing them.
    propagate, logger.propagate = logger.propagate, False
    try:
        yield messages
    finally:
        logger.propagate = propagate
        logger.removeHandler(handler)
