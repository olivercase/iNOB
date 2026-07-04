"""Typed YAML configuration loader.

The full pipeline is parameterised by a single ``Config`` constructed from
``configs/default.yaml`` (or any file passed via ``--config``). Frozen
dataclasses give us autocompletion and immutability; CLI overrides go through
``apply_overrides`` *before* construction so we never mutate a frozen instance.

Conductivities, mesh sizes, sensor params, and solver settings live here once.
Any code that needs them imports the loaded ``Config`` rather than redefining.
"""
from __future__ import annotations

import logging
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from inob.paths import find_project_root, resolve_path

logger = logging.getLogger(__name__)


# ── leaf configs ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataPaths:
    bone_dir: Path
    torso_skin: Path
    vagus_left_glob: str
    vagus_right_glob: str
    muscle_dir: Path
    vessel_dir: Path
    spinal_cord_dir: Path


@dataclass(frozen=True)
class OutputPaths:
    base: Path
    geometry_mat: Path
    geometry_png: Path
    fem_mat: Path
    fem_png: Path
    sensors_mat: Path
    sensors_png: Path
    electrodes_mat: Path
    electrodes_png: Path
    forward_npz: Path
    forward_chunks_dir: Path
    forward_eeg_npz: Path
    sensitivity_dir: Path
    intermediate_bone_clean: Path
    intermediate_bone_union: Path
    logs_dir: Path


@dataclass(frozen=True)
class ShrinkwrapParams:
    pitch: float
    n_samples: int
    close_iter: int
    decimate_target: int
    smooth_iter: int


@dataclass(frozen=True)
class GeometryValidate:
    require_watertight: bool = True
    require_winding_consistent: bool = True
    require_euler_2: bool = True


@dataclass(frozen=True)
class GeometryCfg:
    shrinkwrap: dict[str, ShrinkwrapParams]
    validate: GeometryValidate


@dataclass(frozen=True)
class FemValidate:
    min_mesh_quality: float = 0.05
    require_contiguous_tissue_ids: bool = True
    require_units_mm: bool = True


@dataclass(frozen=True)
class FemCfg:
    pitch_mm: float
    pad_mm: float
    radbound: float
    maxvol: float
    vagus_dilate_voxels: int
    bone_closing_mm: float
    tissues: tuple[str, ...]
    validate: FemValidate


@dataclass(frozen=True)
class SensorCfg:
    resolution_mm: float
    depth_mm: float
    angular_margin_deg: float
    cylinder_radius_factor: float
    z_crop_low_factor: float


@dataclass(frozen=True)
class SolverCfg:
    type: str = "cg"
    reduction: float = 1.0e-10
    edge_norm_type: str = "houston"
    penalty: int = 20
    scheme: str = "sipg"
    weights: str = "tensorOnly"
    intorderadd: int = 5
    post_process_meg: bool = True
    subtract_mean: bool = True


@dataclass(frozen=True)
class ForwardValidate:
    require_finite: bool = True
    max_abs_fT_per_nAm: float = 1.0e6


@dataclass(frozen=True)
class ForwardCfg:
    conductivities_sm: dict[str, float]
    sigma_unit_scale: float
    source_spacing_mm: float
    source_tissue: str
    duneuro_path: Path | None
    solver: SolverCfg
    validate: ForwardValidate
    # Optional explicit dipole positions (mm). When non-empty, the forward
    # solve uses these instead of geometry-derived ``vagus_sources`` sampling
    # — this is how the GUI's clicked source points reach the solver.
    point_sources: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class ElectrodeCfg:
    rows: int = 6
    cols: int = 5
    contact_pitch_mm: float = 5.0
    shape: str = "paddle32"          # "rectangular" / "paddle32" / "whole_body"
    head_offset_mm: float = 15.0
    foot_offset_mm: float = 15.0
    target_tissue: str = "vagus_left"
    target_z_low_factor: float = 0.65
    target_z_high_factor: float = 0.95
    label_prefix: str = "elec"
    n_contacts: int = 1000           # whole_body: total contact count
    sample_seed: int = 0             # whole_body: RNG seed for skin sampling


@dataclass(frozen=True)
class NoiseCfg:
    # Defaults match QuSpin Gen-3 OPM and Malliaras-group PEDOT:PSS textile
    # electrode hardware. See ``configs/default.yaml`` for citations.
    opm_intrinsic_fT_sqrtHz: float = 7.0
    eeg_amplifier_uV_sqrtHz: float = 0.5
    eeg_electrode_skin_kohm: float = 10.0
    bandwidth_hz: float = 1000.0


@dataclass(frozen=True)
class ClusterCfg:
    n_chunks: int = 32


@dataclass(frozen=True)
class SensitivityCfg:
    perturbations: tuple[float, ...] = (0.8, 1.0, 1.25)
    tissues: tuple[str, ...] = ("bone", "skin")


@dataclass(frozen=True)
class ReproCfg:
    seed: int = 0


@dataclass(frozen=True)
class Config:
    project_root: Path
    data: DataPaths
    outputs: OutputPaths
    geometry: GeometryCfg
    fem: FemCfg
    sensors: SensorCfg
    electrodes: ElectrodeCfg
    noise: NoiseCfg
    forward: ForwardCfg
    cluster: ClusterCfg
    sensitivity: SensitivityCfg
    reproducibility: ReproCfg
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


# ── overrides ──────────────────────────────────────────────────────────────

def parse_override(spec: str) -> tuple[list[str], Any]:
    """Parse ``forward.source_spacing_mm=3.0`` → (['forward','source_spacing_mm'], 3.0)."""
    if "=" not in spec:
        raise ValueError(f"override must look like 'key.path=value', got {spec!r}")
    path_str, raw_val = spec.split("=", 1)
    path = [p for p in path_str.split(".") if p]
    if not path:
        raise ValueError(f"empty override key in {spec!r}")
    try:
        val = yaml.safe_load(raw_val)
    except yaml.YAMLError as e:
        raise ValueError(f"could not parse override value {raw_val!r}: {e}") from e
    return path, val


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.path=value`` strings to a nested dict in place; return it."""
    if not overrides:
        return data
    for spec in overrides:
        path, val = parse_override(spec)
        cursor = data
        for key in path[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[path[-1]] = val
    return data


# ── construction ───────────────────────────────────────────────────────────

class ConfigError(ValueError):
    """Raised when config construction fails (missing/extra keys, bad types)."""


def _check_keys(name: str, expected: set[str], got: set[str]) -> None:
    missing = expected - got
    extra = got - expected
    msgs = []
    if missing:
        msgs.append(f"missing keys: {sorted(missing)}")
    if extra:
        msgs.append(f"unknown keys: {sorted(extra)}")
    if msgs:
        raise ConfigError(f"[{name}] " + "; ".join(msgs))


def _build_dataclass(
    cls: type, data: dict[str, Any], name: str, *, allow_defaults: bool = True
) -> Any:
    """Construct a dataclass from a dict, raising on missing/extra keys.

    With ``allow_defaults=True`` only fields without defaults are required;
    with ``allow_defaults=False`` every field must be present.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    all_fields = {f.name: f for f in fields(cls)}
    got = set(data.keys())
    extra = got - set(all_fields)
    if extra:
        raise ConfigError(f"[{name}] unknown keys: {sorted(extra)}")
    if allow_defaults:
        required = {
            n for n, f in all_fields.items()
            if f.default is MISSING and f.default_factory is MISSING
        }
    else:
        required = set(all_fields)
    missing = required - got
    if missing:
        raise ConfigError(f"[{name}] missing keys: {sorted(missing)}")
    return cls(**{k: data[k] for k in data})


def _build_shrinkwrap(d: dict[str, Any]) -> dict[str, ShrinkwrapParams]:
    out: dict[str, ShrinkwrapParams] = {}
    for k, v in d.items():
        if not isinstance(v, dict):
            raise ConfigError(f"geometry.shrinkwrap.{k} must be a mapping")
        out[k] = _build_dataclass(ShrinkwrapParams, v, f"geometry.shrinkwrap.{k}",
                                  allow_defaults=False)
    return out


def _build_geometry(d: dict[str, Any]) -> GeometryCfg:
    sw = _build_shrinkwrap(d.get("shrinkwrap", {}))
    val = _build_dataclass(GeometryValidate, d.get("validate", {}), "geometry.validate")
    return GeometryCfg(shrinkwrap=sw, validate=val)


def _build_fem(d: dict[str, Any]) -> FemCfg:
    val = _build_dataclass(FemValidate, d.get("validate", {}), "fem.validate")
    body = {k: v for k, v in d.items() if k != "validate"}
    body["tissues"] = tuple(body.get("tissues", ()))
    return _build_dataclass(
        FemCfg, {**body, "validate": val}, "fem", allow_defaults=False
    )


def _build_forward(d: dict[str, Any]) -> ForwardCfg:
    solver = _build_dataclass(SolverCfg, d.get("solver", {}), "forward.solver")
    val = _build_dataclass(ForwardValidate, d.get("validate", {}), "forward.validate")
    body = {k: v for k, v in d.items() if k not in ("solver", "validate")}
    duneuro_path = body.get("duneuro_path")
    body["duneuro_path"] = Path(duneuro_path).expanduser() if duneuro_path else None
    if body.get("point_sources"):
        body["point_sources"] = tuple(
            tuple(float(c) for c in p) for p in body["point_sources"]
        )
    # allow_defaults=True so the optional point_sources field may be omitted;
    # the other forward fields have no defaults and so remain required.
    return _build_dataclass(
        ForwardCfg, {**body, "solver": solver, "validate": val},
        "forward", allow_defaults=True,
    )


def _build_outputs(d: dict[str, Any], root: Path) -> OutputPaths:
    expected = {f.name for f in fields(OutputPaths)}
    _check_keys("outputs", expected, set(d.keys()))
    return OutputPaths(**{k: resolve_path(d[k], root) for k in expected})


def _build_data(d: dict[str, Any], root: Path) -> DataPaths:
    expected = {f.name for f in fields(DataPaths)}
    _check_keys("data", expected, set(d.keys()))
    return DataPaths(
        bone_dir=resolve_path(d["bone_dir"], root),
        torso_skin=resolve_path(d["torso_skin"], root),
        vagus_left_glob=str(resolve_path(d["vagus_left_glob"], root)),
        vagus_right_glob=str(resolve_path(d["vagus_right_glob"], root)),
        muscle_dir=resolve_path(d["muscle_dir"], root),
        vessel_dir=resolve_path(d["vessel_dir"], root),
        spinal_cord_dir=resolve_path(d["spinal_cord_dir"], root),
    )


def load_config(
    path: str | Path,
    *,
    overrides: list[str] | None = None,
    project_root: str | Path | None = None,
) -> Config:
    """Load a YAML config and construct a frozen :class:`Config`.

    ``overrides`` is a list of ``key.path=value`` strings; values are parsed as
    YAML scalars (so ``foo=true``, ``foo=3.14``, ``foo=null`` all work).
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"top-level YAML must be a mapping (got {type(raw).__name__})")

    apply_overrides(raw, overrides)

    # Project-root resolution priority:
    #   1. explicit ``project_root`` arg (CLI)
    #   2. absolute path in YAML
    #   3. ``find_project_root`` walking up from the YAML's directory
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
    elif (yaml_root := raw.get("project_root")) and Path(yaml_root).is_absolute():
        root = Path(yaml_root).expanduser().resolve()
    else:
        root = find_project_root(path.parent)

    expected_top = {
        "project_root", "data", "outputs", "geometry", "fem",
        "sensors", "electrodes", "noise", "forward", "cluster",
        "sensitivity", "reproducibility",
    }
    _check_keys("(top level)", expected_top, set(raw.keys()))

    sens_raw = dict(raw.get("sensitivity", {}))
    if "perturbations" in sens_raw:
        sens_raw["perturbations"] = tuple(sens_raw["perturbations"])
    if "tissues" in sens_raw:
        sens_raw["tissues"] = tuple(sens_raw["tissues"])

    cfg = Config(
        project_root=root,
        data=_build_data(raw["data"], root),
        outputs=_build_outputs(raw["outputs"], root),
        geometry=_build_geometry(raw["geometry"]),
        fem=_build_fem(raw["fem"]),
        sensors=_build_dataclass(SensorCfg, raw["sensors"], "sensors",
                                  allow_defaults=False),
        electrodes=_build_dataclass(
            ElectrodeCfg, raw.get("electrodes", {}), "electrodes",
        ),
        noise=_build_dataclass(NoiseCfg, raw.get("noise", {}), "noise"),
        forward=_build_forward(raw["forward"]),
        cluster=_build_dataclass(ClusterCfg, raw.get("cluster", {}), "cluster"),
        sensitivity=_build_dataclass(
            SensitivityCfg, sens_raw, "sensitivity",
        ),
        reproducibility=_build_dataclass(
            ReproCfg, raw.get("reproducibility", {}), "reproducibility"
        ),
        raw=raw,
    )
    logger.debug("Loaded config from %s with project_root=%s", path, root)
    return cfg


def ensure_output_dirs(cfg: Config) -> None:
    """Create every parent dir for the configured output files."""
    o = cfg.outputs
    for p in (
        o.base, o.logs_dir, o.intermediate_bone_clean,
        o.geometry_mat.parent, o.fem_mat.parent,
        o.sensors_mat.parent, o.forward_npz.parent, o.forward_chunks_dir,
    ):
        Path(p).mkdir(parents=True, exist_ok=True)
