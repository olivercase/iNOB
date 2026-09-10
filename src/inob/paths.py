"""Project-root resolution and output-path helpers.

Lookup order for the project root, in priority:
1. Explicit argument (CLI ``--project-root`` or function param)
2. ``INOB_ROOT`` environment variable
3. Walk up from cwd looking for a marker (``configs/default.yaml`` or ``pyproject.toml``)
4. Fallback to cwd
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_MARKERS = ("pyproject.toml", "configs/default.yaml")


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` (or cwd) until a marker file is found."""
    cur = (start or Path.cwd()).resolve()
    for parent in (cur, *cur.parents):
        if any((parent / m).exists() for m in PROJECT_MARKERS):
            return parent
    return cur


def resolve_project_root(explicit: Path | str | None = None) -> Path:
    """Resolve the project root, honoring CLI/env overrides."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("INOB_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return find_project_root()


def resolve_path(p: str | Path, root: Path) -> Path:
    """Resolve a path relative to ``root`` if it isn't absolute."""
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (root / pp).resolve()
