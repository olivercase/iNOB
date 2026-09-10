"""NPZ I/O for the DUNEuro leadfield artefact.

Local solve and cluster-reduce paths produce the SAME schema:

    L              (C, 3*S) float64    raw DUNEuro forward output
                                         MEG: T per A·m
                                         EEG: raw V at DUNEuro mm-mode units
    L_fT_per_nAm   (C, 3*S) float64    "human" units, modality-dependent:
                                         MEG: L × 1e6                  → fT per nA·m
                                         EEG: L × EEG_CALIBRATION_FACTOR (=0.622) → µV per nA·m
                                       The field name is historical (the
                                       schema started life as MEG-only); for
                                       EEG outputs read it as µV per nA·m.
                                       Modality is inferred from
                                       ``channel_names`` ("mag-…" → MEG;
                                       "elec-…" → EEG) or ``chantype`` of
                                       the corresponding sensor file.
    source_pos     (S, 3)   float64    mm — REQUIRED in both local and cluster outputs
                                         (the legacy cluster reduce.py was missing it)
    coil_pos       (C, 3)   float64
    coil_orient    (C, 3)   float64
    channel_names  (C,)     bytes/str
    conductivities (K,)     float64    S/mm (post unit-scale)
    tissue_labels  (K,)     bytes/str
    seed           ()       int        reproducibility seed used to build inputs
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_KEYS: tuple[str, ...] = (
    "L",
    "L_fT_per_nAm",
    "source_pos",
    "coil_pos",
    "coil_orient",
    "channel_names",
    "conductivities",
    "tissue_labels",
)


class SchemaError(ValueError):
    """Raised when an .npz leadfield file does not match the expected schema."""


@dataclass(frozen=True)
class Leadfield:
    L: np.ndarray  # (C, 3S) Tesla per A·m
    L_fT_per_nAm: np.ndarray  # (C, 3S) fT/nAm
    source_pos: np.ndarray  # (S, 3) mm
    coil_pos: np.ndarray  # (C, 3) mm
    coil_orient: np.ndarray  # (C, 3) unit-norm
    channel_names: tuple[str, ...]
    conductivities: np.ndarray  # (K,) S/mm
    tissue_labels: tuple[str, ...]
    seed: int | None = None


def save_leadfield(path: Path, lf: Leadfield) -> None:
    """Atomically save a :class:`Leadfield` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        np.savez(
            tmp_path,
            L=np.asarray(lf.L, dtype=np.float64),
            L_fT_per_nAm=np.asarray(lf.L_fT_per_nAm, dtype=np.float64),
            source_pos=np.asarray(lf.source_pos, dtype=np.float64),
            coil_pos=np.asarray(lf.coil_pos, dtype=np.float64),
            coil_orient=np.asarray(lf.coil_orient, dtype=np.float64),
            channel_names=np.array(list(lf.channel_names)),
            conductivities=np.asarray(lf.conductivities, dtype=np.float64),
            tissue_labels=np.array(list(lf.tissue_labels)),
            seed=np.array(-1 if lf.seed is None else int(lf.seed), dtype=np.int64),
        )
        # numpy added a default .npz suffix, keep it consistent
        candidate = (
            tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
        )
        os.replace(candidate, path)
    except BaseException:
        for p in (tmp_path, tmp_path.with_suffix(tmp_path.suffix + ".npz")):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    logger.info("Wrote leadfield → %s (%.1f MB)", path, path.stat().st_size / 1e6)


def load_leadfield(path: Path) -> Leadfield:
    """Load a leadfield NPZ produced by :func:`save_leadfield` (or the legacy scripts)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"leadfield not found: {path}")
    with np.load(path, allow_pickle=False) as f:
        keys = set(f.files)
        missing = [k for k in REQUIRED_KEYS if k not in keys]
        if missing:
            raise SchemaError(f"{path}: missing keys {missing}")
        L = f["L"]
        L_fT = f["L_fT_per_nAm"]
        sp = f["source_pos"]
        cp = f["coil_pos"]
        co = f["coil_orient"]
        cn = f["channel_names"]
        cond = f["conductivities"]
        labels = f["tissue_labels"]
        seed = int(f["seed"]) if "seed" in keys else None
    if seed is not None and seed < 0:
        seed = None
    channel_names = tuple(s.decode() if isinstance(s, bytes) else str(s) for s in cn)
    tissue_labels = tuple(s.decode() if isinstance(s, bytes) else str(s) for s in labels)
    return Leadfield(
        L=L,
        L_fT_per_nAm=L_fT,
        source_pos=sp,
        coil_pos=cp,
        coil_orient=co,
        channel_names=channel_names,
        conductivities=cond,
        tissue_labels=tissue_labels,
        seed=seed,
    )


def validate_leadfield(
    lf: Leadfield,
    *,
    require_finite: bool = True,
    max_abs_fT_per_nAm: float = 1.0e6,
) -> None:
    """Validate shapes, finiteness, and amplitude bound on a leadfield."""
    C = lf.coil_pos.shape[0]
    S = lf.source_pos.shape[0]
    if lf.L.shape != (C, 3 * S):
        raise SchemaError(f"L shape {lf.L.shape} != (C={C}, 3*S={3 * S})")
    if lf.L_fT_per_nAm.shape != lf.L.shape:
        raise SchemaError("L_fT_per_nAm shape != L shape")
    if lf.coil_orient.shape != lf.coil_pos.shape:
        raise SchemaError("coil_orient shape != coil_pos shape")
    if len(lf.channel_names) != C:
        raise SchemaError(f"channel_names length {len(lf.channel_names)} != C={C}")
    if len(lf.tissue_labels) != lf.conductivities.shape[0]:
        raise SchemaError("tissue_labels length != conductivities length")
    if require_finite and not np.isfinite(lf.L).all():
        raise SchemaError("L contains non-finite values")
    if C and S:
        amax = float(np.max(np.abs(lf.L_fT_per_nAm)))
        if amax > max_abs_fT_per_nAm:
            raise SchemaError(
                f"|L_fT_per_nAm| max {amax:.3g} exceeds bound {max_abs_fT_per_nAm:.3g}"
            )
