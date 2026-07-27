"""HDF5 I/O for geometry, FEM, and sensor arrays.

All writes are atomic (write to a sibling tempfile, then ``os.replace``).
Loaders return typed dataclasses; ``validate_*`` functions raise
:class:`SchemaError` with a precise message on any malformed file.

On-disk schemas (kept stable for backward compat with existing artefacts):

  geometry.mat
    /<compartment>/vertices    (3, Nv) float64    — MATLAB-style transpose
    /<compartment>/faces       (3, Nf) float64    — 1-indexed (MATLAB)
    /<compartment>/unit        (2, 1)  uint8      — b"mm" packed

  fem.mat
    /pos                       (N, 3)  float64    — node coordinates [mm]
    /tet                       (M, 4)  int32      — 0-indexed
    /tissue                    (M,)    int32      — 1..K
    /tissue_labels             (K,)    SN bytes   — ASCII
    /unit                      (2, 1)  uint8      — b"mm" packed

  sensor_array.mat   (FieldTrip ``grad`` layout)
    /grad/coilpos              (C, 3)  float64
    /grad/coilori              (C, 3)  float64
    /grad/label                (C,)    bytes
    /grad/chantype             (C,)    bytes
    /grad/chanunit             (C,)    bytes
    /grad/unit                 ()      bytes      — scalar b"mm"
"""
from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

UNIT_MM = np.array([[ord("m")], [ord("m")]], dtype=np.uint8)


class SchemaError(ValueError):
    """Raised when an HDF5 artefact does not match the expected schema."""


# ── atomic writes ──────────────────────────────────────────────────────────

@contextmanager
def atomic_write_hdf5(path: Path) -> Iterator[h5py.File]:
    """Yield an open h5py.File on a sibling tempfile; rename into ``path`` on success.

    A partial / crashed write leaves ``path`` untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        with h5py.File(str(tmp_path), "w") as h5:
            yield h5
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _decode_bytes(arr: np.ndarray) -> list[str]:
    """Decode an array of HDF5 byte strings to a list of stripped Python strs."""
    return [bytes(np.asarray(s).tobytes()).rstrip(b"\x00").decode() for s in arr.ravel()]


def _encode_strings(strs: list[str], width: int | None = None) -> np.ndarray:
    """Encode strings as fixed-width ASCII bytes (S<width>) for HDF5."""
    if width is None:
        width = max(1, max((len(s) for s in strs), default=1))
    return np.array(strs, dtype=f"S{width}")


# ── geometry ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompartmentMesh:
    name: str
    vertices: np.ndarray   # (N, 3) float64, mm
    faces: np.ndarray      # (M, 3) int64, 0-indexed


@dataclass(frozen=True)
class Geometry:
    compartments: dict[str, CompartmentMesh]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.compartments)


def save_geometry(path: Path, geom: Geometry) -> None:
    """Write a multi-compartment geometry to HDF5 (MATLAB v7.3-compatible layout)."""
    with atomic_write_hdf5(path) as f:
        for name, comp in geom.compartments.items():
            v = np.asarray(comp.vertices, dtype=np.float64)
            faces = np.asarray(comp.faces, dtype=np.int64) + 1   # 1-indexed for MATLAB
            g = f.create_group(name)
            g.create_dataset("vertices", data=v.T)               # (3, N)
            g.create_dataset("faces",    data=faces.astype(np.float64).T)  # (3, M)
            g.create_dataset("unit",     data=UNIT_MM)
    logger.info("Wrote geometry → %s (%d compartments)", path, len(geom.compartments))


def load_geometry(path: Path) -> Geometry:
    """Load a geometry HDF5 file produced by :func:`save_geometry`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"geometry file not found: {path}")
    comps: dict[str, CompartmentMesh] = {}
    with h5py.File(str(path), "r") as f:
        for name, g in f.items():
            if not isinstance(g, h5py.Group):
                continue
            if "vertices" not in g or "faces" not in g:
                raise SchemaError(f"compartment {name!r} missing vertices/faces in {path}")
            v_raw = g["vertices"][...]
            f_raw = g["faces"][...]
            # (3, N) → (N, 3); detect orientation
            v = v_raw.T if v_raw.shape[0] == 3 and v_raw.shape[1] != 3 else v_raw
            faces = (f_raw.T if f_raw.shape[0] == 3 and f_raw.shape[1] != 3 else f_raw)
            faces = np.asarray(faces, dtype=np.int64) - 1        # back to 0-indexed
            comps[name] = CompartmentMesh(name=name, vertices=np.asarray(v, dtype=np.float64),
                                          faces=faces)
    return Geometry(compartments=comps)


def validate_geometry(geom: Geometry, *, require_units_mm: bool = True) -> None:
    """Sanity-check a geometry: shapes, indices in range, finite, plausible mm extent."""
    if not geom.compartments:
        raise SchemaError("geometry has no compartments")
    for name, c in geom.compartments.items():
        if c.vertices.ndim != 2 or c.vertices.shape[1] != 3:
            raise SchemaError(f"{name}: vertices must be (N, 3); got {c.vertices.shape}")
        if c.faces.ndim != 2 or c.faces.shape[1] != 3:
            raise SchemaError(f"{name}: faces must be (M, 3); got {c.faces.shape}")
        if not np.isfinite(c.vertices).all():
            raise SchemaError(f"{name}: vertices contain non-finite values")
        if c.faces.max() >= len(c.vertices) or c.faces.min() < 0:
            raise SchemaError(
                f"{name}: face index out of range [0, {len(c.vertices)})"
            )
        if require_units_mm:
            extent = float(np.ptp(c.vertices, axis=0).max())
            if not (1.0 <= extent <= 10_000.0):
                raise SchemaError(
                    f"{name}: bbox extent {extent:.3g} is implausible for mm "
                    f"(expected 1..10000 mm)"
                )


# ── FEM ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FemMesh:
    nodes: np.ndarray            # (N, 3) float64 mm
    tets: np.ndarray             # (M, 4) int32 0-indexed
    tissue: np.ndarray           # (M,) int32 1..K
    tissue_labels: tuple[str, ...]
    unit: str = "mm"

    @property
    def label_to_id(self) -> dict[str, int]:
        return {lab: i + 1 for i, lab in enumerate(self.tissue_labels)}


def save_fem(path: Path, mesh: FemMesh) -> None:
    """Write a tetrahedral FEM mesh to HDF5."""
    nodes = np.ascontiguousarray(mesh.nodes, dtype=np.float64)
    tets = np.ascontiguousarray(mesh.tets, dtype=np.int32)
    tissue = np.ascontiguousarray(mesh.tissue, dtype=np.int32)
    labels = list(mesh.tissue_labels)
    with atomic_write_hdf5(path) as f:
        f.create_dataset("pos", data=nodes)
        f.create_dataset("tet", data=tets)
        f.create_dataset("tissue", data=tissue)
        f.create_dataset("tissue_labels", data=_encode_strings(labels))
        f.create_dataset("unit", data=UNIT_MM)
    logger.info(
        "Wrote FEM → %s (%d nodes, %d tets, %d tissues)",
        path, len(nodes), len(tets), len(labels),
    )


def load_fem(path: Path) -> FemMesh:
    """Load a FEM mesh produced by :func:`save_fem` (or the legacy scripts)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FEM file not found: {path}")
    required = ("pos", "tet", "tissue", "tissue_labels")
    with h5py.File(str(path), "r") as f:
        missing = [k for k in required if k not in f]
        if missing:
            raise SchemaError(f"{path}: missing keys {missing}")
        nodes = np.asarray(f["pos"][...], dtype=np.float64)
        tets = np.asarray(f["tet"][...], dtype=np.int64)
        tissue = np.asarray(f["tissue"][...], dtype=np.int64).ravel()
        labels = tuple(_decode_bytes(f["tissue_labels"][...]))
        unit = "mm"
        if "unit" in f:
            try:
                unit = bytes(f["unit"][...].flatten().tolist()).decode().strip("\x00") or "mm"
            except Exception:
                unit = "mm"
    return FemMesh(nodes=nodes, tets=tets.astype(np.int32),
                   tissue=tissue.astype(np.int32),
                   tissue_labels=labels, unit=unit)


def validate_fem(
    mesh: FemMesh,
    *,
    require_contiguous_tissue_ids: bool = True,
    require_units_mm: bool = True,
) -> None:
    """Validate a FEM mesh: shapes, ranges, contiguous tissue IDs, finite coords."""
    if mesh.nodes.ndim != 2 or mesh.nodes.shape[1] != 3:
        raise SchemaError(f"nodes must be (N, 3); got {mesh.nodes.shape}")
    if mesh.tets.ndim != 2 or mesh.tets.shape[1] != 4:
        raise SchemaError(f"tets must be (M, 4); got {mesh.tets.shape}")
    if mesh.tissue.shape != (mesh.tets.shape[0],):
        raise SchemaError(
            f"tissue length {mesh.tissue.shape} != n_tets {mesh.tets.shape[0]}"
        )
    if not np.isfinite(mesh.nodes).all():
        raise SchemaError("nodes contain non-finite values")
    n = len(mesh.nodes)
    if mesh.tets.size and (mesh.tets.max() >= n or mesh.tets.min() < 0):
        raise SchemaError(
            f"tet index out of range [0, {n}); got [{mesh.tets.min()}, {mesh.tets.max()}]"
        )
    if mesh.tissue.size:
        unique = np.unique(mesh.tissue)
        if unique.min() < 1:
            raise SchemaError(f"tissue ids must start at 1; got min={unique.min()}")
        if require_contiguous_tissue_ids:
            expected = np.arange(1, len(mesh.tissue_labels) + 1, dtype=mesh.tissue.dtype)
            if not np.array_equal(unique, expected):
                raise SchemaError(
                    f"tissue ids {unique.tolist()} not contiguous 1..{len(mesh.tissue_labels)}"
                )
    if require_units_mm:
        extent = float(np.ptp(mesh.nodes, axis=0).max()) if len(mesh.nodes) else 0.0
        if extent and not (1.0 <= extent <= 10_000.0):
            raise SchemaError(
                f"node bbox extent {extent:.3g} implausible for mm (expected 1..10000)"
            )


# ── sensor array (FieldTrip grad layout) ───────────────────────────────────

@dataclass(frozen=True)
class SensorArray:
    coilpos: np.ndarray        # (C, 3) float64 mm
    coilori: np.ndarray        # (C, 3) float64 unit-norm
    labels: tuple[str, ...]    # length C
    chantype: tuple[str, ...]  # length C, default "megmag"
    chanunit: tuple[str, ...]  # length C, default "T"
    unit: str = "mm"


def save_sensors(path: Path, sensors: SensorArray) -> None:
    """Write a FieldTrip-style triaxial sensor array."""
    pos = np.ascontiguousarray(sensors.coilpos, dtype=np.float64)
    ori = np.ascontiguousarray(sensors.coilori, dtype=np.float64)
    with atomic_write_hdf5(path) as f:
        g = f.create_group("grad")
        g.create_dataset("coilpos", data=pos)
        g.create_dataset("coilori", data=ori)
        g.create_dataset("label",    data=_encode_strings(list(sensors.labels)))
        g.create_dataset("chantype", data=_encode_strings(list(sensors.chantype)))
        g.create_dataset("chanunit", data=_encode_strings(list(sensors.chanunit)))
        g.create_dataset("unit",     data=np.bytes_(sensors.unit.encode("ascii")))
    logger.info("Wrote sensors → %s (%d channels)", path, len(pos))


def load_sensors(path: Path) -> SensorArray:
    """Load a sensor array produced by :func:`save_sensors` (or the legacy script)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"sensor file not found: {path}")
    with h5py.File(str(path), "r") as f:
        if "grad" not in f:
            raise SchemaError(f"{path}: missing /grad group")
        g = f["grad"]
        for k in ("coilpos", "coilori", "label"):
            if k not in g:
                raise SchemaError(f"{path}: missing /grad/{k}")
        pos = np.asarray(g["coilpos"][...], dtype=np.float64)
        ori = np.asarray(g["coilori"][...], dtype=np.float64)
        labels = tuple(_decode_bytes(g["label"][...]))
        chantype = tuple(_decode_bytes(g["chantype"][...])) if "chantype" in g \
                   else tuple(["megmag"] * len(pos))
        chanunit = tuple(_decode_bytes(g["chanunit"][...])) if "chanunit" in g \
                   else tuple(["T"] * len(pos))
        unit = "mm"
        if "unit" in g:
            try:
                unit = bytes(np.asarray(g["unit"][()]).tobytes()).rstrip(b"\x00").decode() or "mm"
            except Exception:
                unit = "mm"
    n = ori.shape[0]
    if n and not np.allclose(np.linalg.norm(ori, axis=1), 1.0, atol=1e-6):
        ori = ori / np.linalg.norm(ori, axis=1, keepdims=True)
    return SensorArray(coilpos=pos, coilori=ori, labels=labels,
                       chantype=chantype, chanunit=chanunit, unit=unit)


def validate_sensors(sensors: SensorArray) -> None:
    """Validate sensor shapes + orientation unit-norm; raise on mismatch."""
    if sensors.coilpos.ndim != 2 or sensors.coilpos.shape[1] != 3:
        raise SchemaError(f"coilpos must be (C, 3); got {sensors.coilpos.shape}")
    if sensors.coilori.shape != sensors.coilpos.shape:
        raise SchemaError("coilori must match coilpos shape")
    n = len(sensors.coilpos)
    for name, seq in (("labels", sensors.labels),
                      ("chantype", sensors.chantype),
                      ("chanunit", sensors.chanunit)):
        if len(seq) != n:
            raise SchemaError(f"{name} length {len(seq)} != n_channels {n}")
    if n and not np.isfinite(sensors.coilpos).all():
        raise SchemaError("coilpos contains non-finite values")
    if n:
        norms = np.linalg.norm(sensors.coilori, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise SchemaError(
                f"coilori not unit-norm: range [{norms.min():.6f}, {norms.max():.6f}]"
            )


# ── convenience ────────────────────────────────────────────────────────────

def write_with_validation(
    path: Path, write_fn: Callable[[Path], None], validate_fn: Callable[[Path], None]
) -> None:
    """Helper: ``write_fn(path)`` then ``validate_fn(path)``; remove on validation failure."""
    write_fn(path)
    try:
        validate_fn(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
