"""Anatomy lookups shared across sensors/viz — vertebral levels from segmented STLs.

The leadfield carries no vertebral labels, only ``source_pos`` (mm). To locate a
level (e.g. C7) we read its own segmented vertebra STL and take its Z bounding
box. Reproducible and anatomy-driven (no hard-coded coordinates), and generalises
to any labelled vertebra. This lives here — not in ``viz`` — so both electrode
placement (``inob.sensors.electrodes``) and the sensor-field figures
(``inob.viz.sensor_field``) can share it without a viz→sensors import.
"""
from __future__ import annotations

import logging
from pathlib import Path

from inob.io.stl import load_first_stl

logger = logging.getLogger(__name__)

# level code → substring of the vertebra STL filename in ``cfg.data.bone_dir``.
_VERTEBRA_STL: dict[str, str] = {
    "c1": "Atlas", "c2": "Axis",
    "c3": "Third cervical vertebra", "c4": "Fourth cervical vertebra",
    "c5": "Fifth cervical vertebra", "c6": "Sixth cervical vertebra",
    "c7": "Seventh cervical vertebra",
    "t1": "First thoracic vertebra", "t2": "Second thoracic vertebra",
    "t3": "Third thoracic vertebra", "t4": "Fourth thoracic vertebra",
    "t5": "Fifth thoracic vertebra", "t6": "Sixth thoracic vertebra",
    "t7": "Seventh thoracic vertebra", "t8": "Eighth thoracic vertebra",
    "t9": "Ninth thoracic vertebra", "t10": "Tenth thoracic vertebra",
    "t11": "Eleventh thoracic vertebra", "t12": "Twelfth thoracic vertebra",
    "l1": "First lumbar vertebra", "l2": "Second lumbar vertebra",
    "l3": "Third lumbar vertebra", "l4": "Fourth lumbar vertebra",
    "l5": "Fifth lumbar vertebra",
}

VERTEBRA_LEVELS: tuple[str, ...] = tuple(_VERTEBRA_STL)


def vertebra_z_band(bone_dir: Path, level: str) -> tuple[float, float]:
    """Z bounding-box (z_lo, z_hi) mm of a vertebra from its segmented STL."""
    key = level.lower()
    if key not in _VERTEBRA_STL:
        raise ValueError(
            f"unknown vertebral level {level!r} (have {', '.join(VERTEBRA_LEVELS)})"
        )
    pattern = str(bone_dir / f"*{_VERTEBRA_STL[key]}*.stl")
    mesh = load_first_stl(pattern)
    z_lo, z_hi = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    logger.info("%s vertebra Z band: %.1f..%.1f mm", key.upper(), z_lo, z_hi)
    return z_lo, z_hi
