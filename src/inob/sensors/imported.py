"""Read a sensor or electrode array from a file instead of generating one.

This is how a real array — the one on the bench, or a vendor layout — gets
leadfields: the file's coil positions and orientations are written to the
same ``grad`` group ``inob sensors`` writes, so ``inob forward`` and
everything downstream consume them unchanged.

Formats
-------
``.csv`` / ``.txt``
    One coil per row, ``x y z nx ny nz`` (comma, tab or space separated; a
    header line and ``#`` comments are skipped). Positions in mm. For
    electrodes ``x y z`` is enough.
``.npz``
    ``coil_pos`` (C, 3) and ``coil_orient`` (C, 3), as a leadfield ``.npz``
    stores them, with an optional ``unit`` (default mm). Electrodes take
    ``coil_pos`` alone, or ``elec_pos``.
``.mat``
    A FieldTrip ``grad`` (``coilpos``/``coilori``, else
    ``chanpos``/``chanori``) or ``elec`` (``elecpos``, else ``chanpos``)
    struct, with its ``unit`` (m, cm or mm) converted to mm. MATLAB v5/v7
    files are read with SciPy and v7.3 files with h5py.

The frame is the file's problem, not ours: nothing here can tell a
mis-registered array from a strange one, so positions are taken as body
frame (docs/COORDINATES.md; +X left, +Y posterior, +Z superior — anterior is
−Y). What *is* checked, when the skin surface is on disk, is that the array
sits outside the body and near it; an array in metres, or in a head frame,
fails that check loudly.

Layout
------
Coils are grouped by position. When every position carries three mutually
orthogonal coils the array is written in the package's triaxial block layout
(all R, then all T1, then all T2) so that the triaxial analyses work on it;
``R`` is the coil nearest the outward skin normal there, and T1/T2 are the
other two in file order — not the ``global_up`` convention a generated array
uses. When every position carries one coil it is a single-axis array and its
coils are labelled ``R``. Either way a coil more than
:data:`RADIAL_TOLERANCE_DEG` off the skin normal is reported, because a map of
"R" channels will then show that coil's component, not the radial one. Any
other grouping keeps the file order with ``-A1``, ``-A2``… labels, and the
analyses that need three coils per position refuse it with a message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import trimesh

from inob.config import Config
from inob.io.hdf5 import SchemaError, SensorArray
from inob.io.stl import load_stl

logger = logging.getLogger(__name__)

#: FieldTrip ``unit`` strings this reader converts, as mm per unit.
UNIT_TO_MM: dict[str, float] = {"m": 1000.0, "cm": 10.0, "mm": 1.0}

#: A coil further than this from the skin normal is reported as not radial.
RADIAL_TOLERANCE_DEG = 30.0

#: Coils at positions closer than this (mm) are one sensor.
_SAME_POSITION_MM = 1e-3

#: |cos| below which three coils at one position count as orthogonal (~1.1°).
_ORTHOGONAL_TOL = 0.02

#: How far outside the skin a coil may sit before it is reported. OPMs stand
#: off 6-20 mm and a SQUID dewar's warm-cold gap adds a few centimetres, so
#: 10 cm is past any real array — in practice a unit or frame error.
MEG_MAX_STANDOFF_MM = 100.0

#: An electrode is on the skin; more than 1 cm off it is a registration error.
EEG_MAX_STANDOFF_MM = 10.0

#: How far inside the skin a coil may sit before it is reported (mesh
#: faceting puts a coil placed "on" the skin a fraction of a mm either side).
_INSIDE_TOL_MM = 1.0

FRAME_NOTE = (
    "positions are taken as body frame in mm (docs/COORDINATES.md: +X left, "
    "+Y posterior, +Z superior); registering the array to the model is the "
    "user's responsibility"
)


@dataclass(frozen=True)
class RawArray:
    """Coil positions (mm) and, when the file has them, orientations."""

    positions: np.ndarray
    orientations: np.ndarray | None
    source: str


def resolve_array_file(cfg: Config, path: str | Path | None) -> Path | None:
    """``path`` as given, else relative to the project root; ``None``/"" is none."""
    if path is None or str(path).strip() == "":
        return None
    p = Path(path).expanduser()
    if not p.is_absolute() and not p.exists():
        p = cfg.project_root / p
    return p


def load_skin_if_present(cfg: Config) -> trimesh.Trimesh | None:
    """The skin surface for the import checks, or ``None`` if it is not on disk."""
    skin_path = cfg.data.torso_skin
    if not Path(skin_path).exists():
        return None
    return load_stl(skin_path)


# ── readers ────────────────────────────────────────────────────────────────


def read_array_file(path: Path, *, kind: str = "meg") -> RawArray:
    """Read positions (mm) and orientations from ``path``.

    ``kind`` is ``"meg"`` (orientations required) or ``"eeg"`` (optional).
    """
    if kind not in ("meg", "eeg"):
        raise ValueError(f"kind must be 'meg' or 'eeg', got {kind!r}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"array file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt", ".tsv"):
        pos, ori = _read_table(path, kind)
        scale = 1.0
    elif suffix == ".npz":
        pos, ori, scale = _read_npz(path, kind)
    elif suffix == ".mat":
        pos, ori, scale = _read_fieldtrip(path, kind)
    else:
        raise SchemaError(
            f"{path}: unrecognised array format {suffix!r}; expected .csv, .txt, .npz or .mat"
        )
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3) * scale
    if ori is not None:
        ori = np.asarray(ori, dtype=np.float64).reshape(-1, 3)
        if ori.shape != pos.shape:
            raise SchemaError(f"{path}: {len(pos)} positions but {len(ori)} orientations")
    if kind == "meg" and ori is None:
        raise SchemaError(f"{path}: a magnetometer array needs coil orientations")
    if len(pos) == 0:
        raise SchemaError(f"{path}: no positions found")
    if not np.isfinite(pos).all():
        raise SchemaError(f"{path}: positions contain NaN or infinite values")
    if ori is not None and not np.isfinite(ori).all():
        raise SchemaError(f"{path}: orientations contain NaN or infinite values")
    return RawArray(positions=pos, orientations=ori, source=str(path))


def _read_table(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray | None]:
    rows: list[list[float]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.replace(",", " ").replace(";", " ").split()
        try:
            values = [float(t) for t in tokens]
        except ValueError:
            if not rows:  # a header line
                continue
            raise SchemaError(f"{path}:{lineno}: not a row of numbers: {raw!r}") from None
        rows.append(values)
    if not rows:
        raise SchemaError(f"{path}: no numeric rows")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise SchemaError(f"{path}: rows have differing column counts {sorted(widths)}")
    table = np.asarray(rows, dtype=np.float64)
    ncol = table.shape[1]
    if ncol == 6:
        return table[:, :3], table[:, 3:]
    if ncol == 3 and kind == "eeg":
        return table, None
    want = "x y z [nx ny nz]" if kind == "eeg" else "x y z nx ny nz"
    raise SchemaError(f"{path}: expected columns {want}, found {ncol} columns")


def _unit_scale(unit: Any, where: str) -> float:
    unit_s = str(unit).strip().lower()
    if unit_s not in UNIT_TO_MM:
        raise SchemaError(
            f"{where}: unit {unit_s!r} not understood; expected one of {sorted(UNIT_TO_MM)}"
        )
    return UNIT_TO_MM[unit_s]


def _read_npz(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray | None, float]:
    with np.load(path, allow_pickle=False) as f:
        keys = set(f.files)
        pos_key = next((k for k in ("coil_pos", "elec_pos") if k in keys), None)
        if pos_key is None:
            raise SchemaError(f"{path}: needs a coil_pos array (has {sorted(keys)})")
        pos = np.asarray(f[pos_key])
        ori = np.asarray(f["coil_orient"]) if "coil_orient" in keys else None
        scale = _unit_scale(f["unit"][()], str(path)) if "unit" in keys else 1.0
    return pos, ori, scale


def _read_fieldtrip(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Positions, orientations and the mm scale from a FieldTrip struct."""
    fields = _load_mat_struct(path, prefer=("grad",) if kind == "meg" else ("elec",))
    where = str(path)
    if "unit" not in fields:
        raise SchemaError(
            f"{where}: the struct has no 'unit' field, so its scale is unknown; "
            "set it in MATLAB (e.g. grad = ft_convert_units(grad, 'mm')) and re-save"
        )
    scale = _unit_scale(fields["unit"], where)

    if kind == "meg":
        pairs = (("coilpos", "coilori"), ("chanpos", "chanori"))
        for pos_k, ori_k in pairs:
            if pos_k in fields and ori_k in fields:
                pos, ori = fields[pos_k], fields[ori_k]
                break
        else:
            raise SchemaError(
                f"{where}: a grad struct needs coilpos/coilori or chanpos/chanori "
                f"(has {sorted(fields)})"
            )
        if pos_k == "coilpos" and "tra" in fields:
            _check_magnetometers(np.atleast_2d(np.asarray(fields["tra"], dtype=float)), where)
        return pos, ori, scale

    for pos_k in ("elecpos", "chanpos"):
        if pos_k in fields:
            return fields[pos_k], None, scale
    raise SchemaError(f"{where}: an elec struct needs elecpos or chanpos (has {sorted(fields)})")


def _check_magnetometers(tra: np.ndarray, where: str) -> None:
    """Refuse a gradiometer ``tra`` (channels built from several coils)."""
    per_channel = np.count_nonzero(tra, axis=1)
    if tra.shape[0] != tra.shape[1] or (per_channel != 1).any():
        raise SchemaError(
            f"{where}: grad.tra combines coils into channels ({tra.shape[1]} coils, "
            f"{tra.shape[0]} channels) — a gradiometer array. Only magnetometers "
            "are modelled; model each coil as its own channel by removing 'tra'"
        )


def _load_mat_struct(path: Path, *, prefer: tuple[str, ...]) -> dict[str, Any]:
    """The fields of the FieldTrip struct in ``path``, as a flat dict."""
    if h5py.is_hdf5(str(path)):
        return _load_mat73_struct(path, prefer)
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    variables = {k: v for k, v in mat.items() if not k.startswith("__")}
    structs = {k: v for k, v in variables.items() if hasattr(v, "_fieldnames")}
    name = next((p for p in prefer if p in structs), None)
    if name is None and len(structs) == 1:
        name = next(iter(structs))
    if name is not None:
        s = structs[name]
        return {f: getattr(s, f) for f in s._fieldnames}
    if structs:
        raise SchemaError(
            f"{path}: holds several structs {sorted(structs)} and none is named "
            f"{' or '.join(prefer)}"
        )
    # Saved with `save(..., '-struct', 'grad')`: the fields are top level.
    return variables


def _load_mat73_struct(path: Path, prefer: tuple[str, ...]) -> dict[str, Any]:
    """Read a v7.3 (HDF5) MATLAB struct. MATLAB stores arrays transposed."""
    out: dict[str, Any] = {}
    with h5py.File(str(path), "r") as f:
        groups = [k for k in f if isinstance(f[k], h5py.Group) and not k.startswith("#")]
        name = next((p for p in prefer if p in groups), None)
        if name is None and len(groups) == 1:
            name = groups[0]
        if name is None:
            raise SchemaError(
                f"{path}: cannot find a {' or '.join(prefer)} struct (groups: {sorted(groups)})"
            )
        g = f[name]
        for key, ds in g.items():
            if not isinstance(ds, h5py.Dataset):
                continue
            cls = ds.attrs.get("MATLAB_class", b"")
            cls = cls.decode() if isinstance(cls, bytes) else str(cls)
            if cls == "cell":
                continue  # labels and chantypes; the importer names channels itself
            data = ds[()]
            if cls == "char":
                out[key] = "".join(chr(c) for c in np.asarray(data).ravel())
            elif isinstance(data, np.ndarray) and data.ndim == 2 and cls:
                out[key] = data.T
            else:
                out[key] = data
    return out


# ── validation against the skin ───────────────────────────────────────────


def _unit_orientations(ori: np.ndarray, where: str) -> np.ndarray:
    """Normalise orientations, refusing zero vectors and noting large fixes."""
    norms = np.linalg.norm(ori, axis=1)
    zero = norms <= 1e-9
    if zero.any():
        raise SchemaError(
            f"{where}: {int(zero.sum())} coil orientation(s) have zero length "
            f"(first at row {int(np.argmax(zero)) + 1}); a coil needs a direction"
        )
    off = np.abs(norms - 1.0) > 1e-3
    if off.any():
        logger.warning(
            "%s: %d orientation(s) were not unit length (|n| in [%.3g, %.3g]); normalised",
            where,
            int(off.sum()),
            float(norms.min()),
            float(norms.max()),
        )
    return ori / norms[:, None]


def skin_offsets(skin: trimesh.Trimesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance (mm, + outside) and outward normal at the nearest skin face."""
    closest, dist, tri = trimesh.proximity.closest_point(skin, np.asarray(points, dtype=float))
    normals = np.asarray(skin.face_normals[tri], dtype=np.float64)
    # A consistent winding fixes outward everywhere up to one global sign, even
    # on a skin that is not closed (the shipped one is open at the cut ends);
    # take that sign from the majority of faces. Orienting each face away from
    # the centroid instead is wrong wherever the surface faces back towards it
    # — under the chin, a 10 mm stand-off read as 10 mm inside.
    if skin.is_winding_consistent:
        if not (skin.is_watertight and skin.volume > 0):
            away = np.einsum("ij,ij->i", skin.face_normals, skin.triangles_center - skin.centroid)
            if np.mean(away > 0) < 0.5:
                normals = -normals
    else:
        flip = np.einsum("ij,ij->i", normals, closest - skin.centroid) < 0
        normals = np.where(flip[:, None], -normals, normals)
    side = np.einsum("ij,ij->i", np.asarray(points) - closest, normals)
    signed = np.where(side < 0, -dist, dist)
    return signed, normals


def check_standoff(
    signed_mm: np.ndarray, *, max_standoff_mm: float, what: str, source: str
) -> None:
    """Warn when coils sit inside the skin or implausibly far outside it."""
    inside = signed_mm < -_INSIDE_TOL_MM
    far = signed_mm > max_standoff_mm
    logger.info(
        "%s: distance to skin %.1f..%.1f mm (median %.1f)",
        what,
        float(signed_mm.min()),
        float(signed_mm.max()),
        float(np.median(signed_mm)),
    )
    if inside.any():
        logger.warning(
            "%s from %s: %d of %d sit inside the skin (deepest %.1f mm). Check the "
            "file's unit and that it is in the model frame (docs/COORDINATES.md)",
            what,
            source,
            int(inside.sum()),
            len(signed_mm),
            float(-signed_mm.min()),
        )
    if far.any():
        logger.warning(
            "%s from %s: %d of %d are more than %.0f mm from the skin (furthest "
            "%.0f mm). Check the file's unit and that it is in the model frame "
            "(docs/COORDINATES.md)",
            what,
            source,
            int(far.sum()),
            len(signed_mm),
            max_standoff_mm,
            float(signed_mm.max()),
        )


# ── building the stored arrays ────────────────────────────────────────────


def _group_by_position(pos: np.ndarray) -> list[np.ndarray]:
    """Row indices sharing a position, in order of first appearance."""
    key = np.round(pos / _SAME_POSITION_MM).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    order = np.argsort(first)
    return [np.flatnonzero(inverse == g) for g in order]


def _is_orthogonal_triplet(ori: np.ndarray) -> bool:
    gram = np.abs(ori @ ori.T)
    return bool((gram[~np.eye(3, dtype=bool)] < _ORTHOGONAL_TOL).all())


def import_magnetometers(raw: RawArray, *, skin: trimesh.Trimesh | None = None) -> SensorArray:
    """Build the stored :class:`SensorArray` from a read magnetometer file."""
    if raw.orientations is None:
        raise SchemaError(f"{raw.source}: a magnetometer array needs coil orientations")
    pos = raw.positions
    ori = _unit_orientations(raw.orientations, raw.source)
    normals = None
    if skin is not None:
        signed, normals = skin_offsets(skin, pos)
        check_standoff(signed, max_standoff_mm=MEG_MAX_STANDOFF_MM, what="coils", source=raw.source)
    else:
        logger.warning("no skin surface available: coil stand-off and radial checks skipped")

    groups = _group_by_position(pos)
    sizes = {len(g) for g in groups}
    width = max(4, len(str(len(groups))))
    cos_tol = np.cos(np.radians(RADIAL_TOLERANCE_DEG))

    if sizes == {3} and all(_is_orthogonal_triplet(ori[g]) for g in groups):
        layout = "triaxial"
        blocks: list[list[int]] = [[], [], []]
        for g in groups:
            if normals is not None:
                align = np.abs(ori[g] @ normals[g[0]])
                r = int(np.argmax(align))
            else:
                r = 0
            rest = [k for k in range(3) if k != r]
            for block, k in zip(blocks, (r, *rest), strict=True):
                block.append(int(g[k]))
        order = np.array(blocks[0] + blocks[1] + blocks[2], dtype=int)
        n = len(groups)
        labels = [f"mag-{i + 1:0{width}d}-{axis}" for axis in ("R", "T1", "T2") for i in range(n)]
        radial_rows = order[:n]
    elif sizes == {1}:
        layout = "single-axis"
        order = np.array([g[0] for g in groups], dtype=int)
        labels = [f"mag-{i + 1:0{width}d}-R" for i in range(len(groups))]
        radial_rows = order
    else:
        layout = "mixed"
        order = np.arange(len(pos))
        labels = [""] * len(pos)
        for i, g in enumerate(groups):
            for k, row in enumerate(g, start=1):
                labels[row] = f"mag-{i + 1:0{width}d}-A{k}"
        radial_rows = np.empty(0, dtype=int)
        logger.warning(
            "%s: positions carry %s coils, not one or three orthogonal; channels "
            "keep the file order and analyses that need three coils per position "
            "will refuse this array",
            raw.source,
            "/".join(str(s) for s in sorted(sizes)),
        )

    if normals is not None and radial_rows.size:
        cos = np.abs(np.einsum("ij,ij->i", ori[radial_rows], normals[radial_rows]))
        off = cos < cos_tol
        if off.any():
            logger.warning(
                "%d of %d coils labelled R are more than %.0f° off the skin "
                "normal (worst %.0f°); maps of R channels show their component, "
                "not the radial field",
                int(off.sum()),
                len(radial_rows),
                RADIAL_TOLERANCE_DEG,
                float(np.degrees(np.arccos(np.clip(cos.min(), 0.0, 1.0)))),
            )
    if layout == "triaxial":
        logger.info("T1/T2 of an imported triaxial array follow the file, not global_up")

    logger.info(
        "imported %d coils at %d positions (%s) from %s",
        len(pos),
        len(groups),
        layout,
        raw.source,
    )
    c = len(order)
    return SensorArray(
        coilpos=pos[order].copy(),
        coilori=ori[order].copy(),
        labels=tuple(labels),
        chantype=tuple(["megmag"] * c),
        chanunit=tuple(["T"] * c),
        unit="mm",
    )


def import_electrodes(raw: RawArray, *, skin: trimesh.Trimesh | None = None) -> SensorArray:
    """Build the stored electrode :class:`SensorArray` from a read file.

    Orientation is informational for EEG (the solver ignores it); the stored
    one is the outward skin normal when the skin is available, else the
    file's, else the direction out from the array's centroid.
    """
    pos = raw.positions
    ori = None
    if skin is not None:
        signed, ori = skin_offsets(skin, pos)
        check_standoff(
            signed, max_standoff_mm=EEG_MAX_STANDOFF_MM, what="electrodes", source=raw.source
        )
    else:
        logger.warning("no skin surface available: electrode-on-skin check skipped")
    if ori is None and raw.orientations is not None:
        ori = _unit_orientations(raw.orientations, raw.source)
    if ori is None:
        ori = pos - pos.mean(axis=0)
        norms = np.linalg.norm(ori, axis=1, keepdims=True)
        ori = np.where(norms > 1e-9, ori / np.where(norms > 0, norms, 1.0), [0.0, 0.0, 1.0])
    ori = ori / np.linalg.norm(ori, axis=1, keepdims=True)
    n = len(pos)
    width = max(4, len(str(n)))
    logger.info("imported %d electrodes from %s", n, raw.source)
    return SensorArray(
        coilpos=pos.copy(),
        coilori=ori,
        labels=tuple(f"elec-{i + 1:0{width}d}" for i in range(n)),
        chantype=tuple(["eeg"] * n),
        chanunit=tuple(["V"] * n),
        unit="mm",
    )
