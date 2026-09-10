"""STL loaders with friendly errors and a unit-of-measure guard.

Loaders accept paths or globs; on ambiguous / missing inputs they raise
:class:`STLLoadError` with the user-facing path so the pipeline log points
straight at the problem (instead of bubbling up ``StopIteration`` from a
``next(glob)`` deep in the call stack).
"""

from __future__ import annotations

import glob as _glob
import logging
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


class STLLoadError(IOError):
    """Raised for any user-facing STL loading problem."""


def _check_unit_mm(mesh: trimesh.Trimesh, name: str) -> None:
    extent = float(mesh.extents.max()) if mesh.extents.size else 0.0
    if extent and not (1.0 <= extent <= 10_000.0):
        raise STLLoadError(
            f"{name}: bbox extent {extent:.3g} implausible for mm "
            f"(expected 1..10000 mm) — check input units"
        )


def load_stl(path: Path | str, *, check_units_mm: bool = True) -> trimesh.Trimesh:
    """Load a single STL into a :class:`trimesh.Trimesh`.

    Raises :class:`STLLoadError` for missing / non-mesh files.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise STLLoadError(f"STL not found: {p}")
    try:
        mesh = trimesh.load_mesh(str(p), process=False)
    except Exception as e:
        raise STLLoadError(f"failed to load {p}: {e}") from e
    if not isinstance(mesh, trimesh.Trimesh):
        # could be a Scene; concatenate to a single mesh
        if hasattr(mesh, "geometry"):
            try:
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            except Exception as e:
                raise STLLoadError(f"{p} contains a scene that could not be merged: {e}") from e
        else:
            raise STLLoadError(f"{p} did not produce a Trimesh (got {type(mesh).__name__})")
    if mesh.vertices.size == 0:
        raise STLLoadError(f"{p}: empty mesh (0 vertices)")
    if check_units_mm:
        _check_unit_mm(mesh, p.name)
    logger.debug("loaded %s: %d V / %d F", p.name, len(mesh.vertices), len(mesh.faces))
    return mesh


def load_stl_glob(
    pattern: str, *, must_exist: bool = True, check_units_mm: bool = True
) -> list[Path]:
    """Resolve a glob; return *sorted* paths.

    With ``must_exist=True`` (the default), an empty match raises
    :class:`STLLoadError` rather than letting downstream code crash with
    ``StopIteration`` or a confusing ``IndexError``.
    """
    paths = sorted(Path(p) for p in _glob.glob(pattern))
    if must_exist and not paths:
        raise STLLoadError(f"no STL files matched glob: {pattern!r}")
    return paths


def load_first_stl(pattern: str, *, check_units_mm: bool = True) -> trimesh.Trimesh:
    """Load the first STL matching a glob (sorted)."""
    paths = load_stl_glob(pattern, must_exist=True)
    return load_stl(paths[0], check_units_mm=check_units_mm)


def concat_stls(paths: list[Path], *, check_units_mm: bool = True) -> trimesh.Trimesh:
    """Load and concatenate multiple STLs (e.g. all bones into one mesh)."""
    if not paths:
        raise STLLoadError("concat_stls: empty path list")
    meshes: list[trimesh.Trimesh] = []
    for p in paths:
        meshes.append(load_stl(p, check_units_mm=False))
    out = trimesh.util.concatenate(meshes)
    if check_units_mm:
        _check_unit_mm(out, f"<concat of {len(paths)} STLs>")
    return out


def bbox_extent(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return ``(extent_x, extent_y, extent_z)`` of the bounding box."""
    return np.asarray(mesh.extents, dtype=np.float64)
