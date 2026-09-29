"""Anatomy lookups shared across sensors/viz — vertebral levels from segmented STLs.

The leadfield carries no vertebral labels, only ``source_pos`` (mm). To locate a
level (e.g. C7) we read its own segmented vertebra STL and take its Z bounding
box. Reproducible and anatomy-driven (no hard-coded coordinates), and generalises
to any labelled vertebra. This lives here — not in ``viz`` — so both electrode
placement (``inob.sensors.electrodes``) and the sensor-field figures
(``inob.viz.sensor_field``) can share it without a viz→sensors import.
"""

from __future__ import annotations

import functools
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inob.io.stl import STLLoadError, load_first_stl

logger = logging.getLogger(__name__)

# level code → substring of the vertebra STL filename in ``cfg.data.bone_dir``.
_VERTEBRA_STL: dict[str, str] = {
    "c1": "Atlas",
    "c2": "Axis",
    "c3": "Third cervical vertebra",
    "c4": "Fourth cervical vertebra",
    "c5": "Fifth cervical vertebra",
    "c6": "Sixth cervical vertebra",
    "c7": "Seventh cervical vertebra",
    "t1": "First thoracic vertebra",
    "t2": "Second thoracic vertebra",
    "t3": "Third thoracic vertebra",
    "t4": "Fourth thoracic vertebra",
    "t5": "Fifth thoracic vertebra",
    "t6": "Sixth thoracic vertebra",
    "t7": "Seventh thoracic vertebra",
    "t8": "Eighth thoracic vertebra",
    "t9": "Ninth thoracic vertebra",
    "t10": "Tenth thoracic vertebra",
    "t11": "Eleventh thoracic vertebra",
    "t12": "Twelfth thoracic vertebra",
    "l1": "First lumbar vertebra",
    "l2": "Second lumbar vertebra",
    "l3": "Third lumbar vertebra",
    "l4": "Fourth lumbar vertebra",
    "l5": "Fifth lumbar vertebra",
}

VERTEBRA_LEVELS: tuple[str, ...] = tuple(_VERTEBRA_STL)


def vertebra_z_band(bone_dir: Path, level: str) -> tuple[float, float]:
    """Z bounding-box (z_lo, z_hi) mm of a vertebra from its segmented STL."""
    key = level.lower()
    if key not in _VERTEBRA_STL:
        raise ValueError(f"unknown vertebral level {level!r} (have {', '.join(VERTEBRA_LEVELS)})")
    pattern = str(bone_dir / f"*{_VERTEBRA_STL[key]}*.stl")
    mesh = load_first_stl(pattern)
    z_lo, z_hi = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    logger.info("%s vertebra Z band: %.1f..%.1f mm", key.upper(), z_lo, z_hi)
    return z_lo, z_hi


# ── z → vertebral level ────────────────────────────────────────────────────
#
# Reviewers read "z = 1419 mm" as a number with no anatomy behind it. The
# figures name the vertebral level instead, and that name has to come from the
# same segmented bones the FEM was built from, not from a table of typical
# heights for some other body.
#
# The level of an axial plane is set by the vertebral *body*, not the whole
# bone: a spinous process slopes down past the next vertebra (C7's reaches
# below T1's upper endplate), so whole-bone Z bounds overlap by a full level
# and cannot tell a body from a disc. The body is the anterior part of the
# bone, i.e. the most negative Y (docs/COORDINATES.md), so each level is
# summarised by the Z extent of its anterior slice.

#: Fraction of a vertebra's anterior–posterior depth taken as its body. A
#: quarter keeps the body and drops the arch and processes on every level from
#: C3 to L5; the atlas has no body and its anterior arch stands in for one.
_BODY_DEPTH_FRACTION = 0.25

#: Half-height of the band around a level boundary that is called a disc. The
#: BodyParts3D bodies nearly abut at their anterior rims (the disc is not
#: segmented), so the measured gap alone would make a disc a fraction of a
#: millimetre tall; 2.5 mm is roughly half a cervical disc.
_GAP_HALF_MIN_MM = 2.5

#: How far beyond the first/last segmented body a Z is still given a level.
#: Further than this and the plane is in the skull or the pelvis, where a
#: vertebral name would be a guess.
_LEVEL_MARGIN_MM = 15.0


@dataclass(frozen=True)
class VertebralBody:
    """Z extent of one vertebral body, mm, body frame."""

    level: str  # "c7"
    z_lo: float  # inferior endplate
    z_hi: float  # superior endplate


@dataclass(frozen=True)
class VertebralLevel:
    """Where an axial plane cuts the spine.

    ``upper``/``lower`` are the same level for a plane through a body, and the
    two neighbours for a plane through the disc between them.
    """

    upper: str
    lower: str

    @property
    def is_gap(self) -> bool:
        return self.upper != self.lower

    @property
    def short(self) -> str:
        """``"C6/C7"`` or ``"C7"`` — for a figure title."""
        if self.is_gap:
            return f"{self.upper.upper()}/{self.lower.upper()}"
        return self.upper.upper()

    @property
    def long(self) -> str:
        """``"C6/C7 intervertebral gap"`` or ``"body of C7"`` — for prose."""
        if self.is_gap:
            return f"{self.short} intervertebral gap"
        return f"body of {self.short}"


def _body_band(mesh_vertices: np.ndarray) -> tuple[float, float]:
    """Z extent (2nd–98th percentile) of the anterior slice of one vertebra."""
    y = mesh_vertices[:, 1]
    y0, y1 = float(y.min()), float(y.max())
    body = mesh_vertices[y <= y0 + _BODY_DEPTH_FRACTION * (y1 - y0)]
    # Percentiles, not min/max: a single stray vertex on an osteophyte-like
    # spike would otherwise push an endplate by millimetres.
    lo, hi = np.percentile(body[:, 2], [2.0, 98.0])
    return float(lo), float(hi)


@functools.lru_cache(maxsize=4)
def _vertebral_bodies_cached(bone_dir: str) -> tuple[VertebralBody, ...]:
    bodies: list[VertebralBody] = []
    for level, name in _VERTEBRA_STL.items():
        try:
            mesh = load_first_stl(str(Path(bone_dir) / f"*{name}*.stl"))
        except STLLoadError:
            # Missing, or an un-fetched LFS pointer. The other levels still
            # locate a plane; only the neighbours of this one lose their gap.
            logger.debug("no usable %s vertebra mesh in %s", level.upper(), bone_dir, exc_info=True)
            continue
        z_lo, z_hi = _body_band(np.asarray(mesh.vertices, dtype=float))
        bodies.append(VertebralBody(level, z_lo, z_hi))
    return tuple(bodies)


def vertebral_bodies(bone_dir: Path) -> tuple[VertebralBody, ...]:
    """The segmented vertebral bodies found in ``bone_dir``, C1 first.

    Cached per directory: 24 STLs take a couple of seconds to parse, and a
    figure may ask for several levels.
    """
    return _vertebral_bodies_cached(str(Path(bone_dir).expanduser().resolve()))


def level_from_bodies(z_mm: float, bodies: tuple[VertebralBody, ...]) -> VertebralLevel | None:
    """The vertebral level of the axial plane at ``z_mm``, or ``None``.

    A plane within ``_GAP_HALF_MIN_MM`` of the boundary between two
    *anatomically adjacent* segmented bodies from C2/C3 down (or inside the
    measured space between them, if that is wider) is the disc between them.
    Otherwise it is the body whose Z extent is nearest. A plane more than
    ``_LEVEL_MARGIN_MM`` beyond the segmented column gets no level.
    """
    if not bodies:
        return None
    z = float(z_mm)
    top = max(b.z_hi for b in bodies)
    bottom = min(b.z_lo for b in bodies)
    if z > top + _LEVEL_MARGIN_MM or z < bottom - _LEVEL_MARGIN_MM:
        return None

    order = {lv: i for i, lv in enumerate(VERTEBRA_LEVELS)}
    ranked = sorted(bodies, key=lambda b: order[b.level])
    for up, lo in itertools.pairwise(ranked):
        if order[lo.level] != order[up.level] + 1:
            continue  # a missing level between them: no disc to name
        if up.level == "c1":
            # Atlanto-axial: a synovial joint round the dens, not a disc. The
            # dens also rises inside C1's ring, so the "gap" would be negative.
            continue
        centre = 0.5 * (up.z_lo + lo.z_hi)
        half = max(0.5 * (up.z_lo - lo.z_hi), _GAP_HALF_MIN_MM)
        if abs(z - centre) <= half:
            return VertebralLevel(up.level, lo.level)

    def _dist(b: VertebralBody) -> float:
        return max(b.z_lo - z, z - b.z_hi, 0.0)

    best = min(ranked, key=lambda b: (_dist(b), abs(z - 0.5 * (b.z_lo + b.z_hi))))
    return VertebralLevel(best.level, best.level)


def vertebral_level(bone_dir: Path | None, z_mm: float) -> VertebralLevel | None:
    """The vertebral level at body-frame height ``z_mm``, or ``None``.

    ``None`` whenever the answer is not known — no bone directory, no
    vertebra meshes in it (a checkout without Git LFS data), or a height
    outside the segmented column — so a figure title can simply drop the
    landmark rather than fail.
    """
    if bone_dir is None:
        return None
    try:
        bodies = vertebral_bodies(Path(bone_dir))
    except OSError:
        logger.debug("cannot read vertebra meshes from %s", bone_dir, exc_info=True)
        return None
    level = level_from_bodies(z_mm, bodies)
    if level is None:
        logger.debug("z = %.1f mm: no vertebral level from %s", z_mm, bone_dir)
    return level


def level_suffix(bone_dir: Path | None, z_mm: float) -> str:
    """``" (C6/C7)"`` for a title, or ``""`` when the level is unknown."""
    level = vertebral_level(bone_dir, z_mm)
    return f" ({level.short})" if level is not None else ""
