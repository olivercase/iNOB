"""Assemble a named anatomical structure from its BodyParts3D components.

BodyParts3D ships geometry per *model component* (an ``FJ`` id), not per anatomical
concept. Whole organs are composites: there is no ``Heart.obj``. The heart is
FMA7088, which the version-4.3 manifest ``FMA2Obj.txt`` expands to 92 FJ parts
(chambers, walls, valves, papillary muscles); each lung is 245-318 parts. Grepping
filenames for "heart" or "lung" finds nothing, which is why they look absent.

This module reads that manifest, indexes ``internal_meshes/`` by FJ id, and
concatenates the parts. Every component is in the same atlas frame, so the result
needs no inter-mesh registration — it is a multi-body surface that the caller
shrinkwraps or unions into one conductor.
"""
from __future__ import annotations

import glob
import logging
import os
from functools import lru_cache
from pathlib import Path

import trimesh

logger = logging.getLogger(__name__)

__all__ = ["FMA", "load_manifest", "index_meshes", "component_ids", "load_composite"]

#: Anatomical concepts this project needs, by FMA id. Names are the FMA labels.
FMA: dict[str, str] = {
    "heart": "FMA7088",
    "myocardium": "FMA7101",
    "lung_right": "FMA7309",
    "lung_left": "FMA7310",
    # Head compartments for the cardiac-field work. "brain" is FMA50801, the
    # neuraxis: 89 components covering both hemispheres, cerebellum and
    # brainstem, i.e. the conducting brain volume rather than a cortical sheet.
    "brain": "FMA50801",
    "insula_right": "FMA72977",
    "insula_left": "FMA72978",
}

_MANIFEST_REL = Path("data/bodyparts3d/raw_4.3/metadata/FMA2Obj.txt")


@lru_cache(maxsize=4)
def load_manifest(path: Path) -> dict[str, tuple[str, ...]]:
    """FMA id -> tuple of FJ component ids, from ``FMA2Obj.txt``.

    The file is CR-delimited (classic Mac line endings), so a plain readlines()
    returns one 1.2 MB line; splitting on universal newlines is deliberate.
    """
    text = Path(path).read_text(encoding="utf8", errors="replace")
    out: dict[str, tuple[str, ...]] = {}
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.startswith("FMA"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        out[parts[0]] = tuple(parts[2].split("+"))
    if not out:
        raise ValueError(f"no FMA rows parsed from {path}")
    logger.info("manifest: %d FMA concepts from %s", len(out), path)
    return out


@lru_cache(maxsize=4)
def index_meshes(mesh_root: Path) -> dict[str, Path]:
    """FJ id -> mesh path, over ``internal_meshes/<system>/FJ..._BP..._FMA..._name.obj``."""
    idx: dict[str, Path] = {}
    for p in glob.glob(str(Path(mesh_root) / "*" / "*.obj")):
        fj = os.path.basename(p).split("_", 1)[0]
        idx.setdefault(fj, Path(p))
    if not idx:
        raise FileNotFoundError(f"no .obj under {mesh_root}/*/")
    logger.info("mesh index: %d FJ components under %s", len(idx), mesh_root)
    return idx


def component_ids(name: str, manifest: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """FJ ids for a concept name in :data:`FMA` (or a raw ``FMA…`` id)."""
    fma = FMA.get(name, name)
    if fma not in manifest:
        raise KeyError(f"{name!r} ({fma}) not in the BodyParts3D 4.3 manifest")
    return manifest[fma]


def load_composite(
    name: str,
    *,
    mesh_root: Path,
    manifest_path: Path,
    require_all: bool = True,
) -> trimesh.Trimesh:
    """Concatenate every component mesh of ``name`` into one multi-body surface.

    Raises if any component is missing and ``require_all`` — a silently partial
    heart is a wrong conductor, not a coarse one.
    """
    manifest = load_manifest(Path(manifest_path))
    index = index_meshes(Path(mesh_root))
    ids = component_ids(name, manifest)
    missing = [i for i in ids if i not in index]
    if missing and require_all:
        raise FileNotFoundError(
            f"{name}: {len(missing)}/{len(ids)} components missing from {mesh_root} "
            f"(first few: {missing[:5]})"
        )
    have = [i for i in ids if i in index]
    meshes = [trimesh.load(index[i], process=False, force="mesh") for i in have]
    merged = trimesh.util.concatenate(meshes)
    logger.info(
        "[composite] %s (%s): %d/%d components -> %d V / %d F",
        name, FMA.get(name, name), len(have), len(ids),
        len(merged.vertices), len(merged.faces),
    )
    return merged


def default_manifest_path(project_root: Path) -> Path:
    return Path(project_root) / _MANIFEST_REL
