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
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from inob.paths import find_project_root, resolve_path

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised when config construction fails (missing/extra keys, bad types)."""


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


#: FEM discretisations DUNEuro compiles for tetrahedra.
SOLVER_TYPES: tuple[str, ...] = ("cg", "dg")


@dataclass(frozen=True)
class SolverCfg:
    """The FEM discretisation and the linear solve on top of it.

    ``type`` picks continuous (``cg``) or discontinuous (``dg``) Galerkin.
    CG is the default and what every leadfield here was solved with: one
    unknown per node, the potential continuous across element faces. DG puts
    the unknowns inside each element and couples neighbours through numerical
    fluxes, so a conductivity jump is represented *as* a jump instead of being
    smeared across the elements either side of it — the standard argument for
    DG in a mesh with a thin, high-contrast compartment, which is exactly what
    a nerve inside muscle is. It costs roughly 4x the degrees of freedom of CG
    on the same mesh.

    ``edge_norm_type``, ``penalty``, ``scheme`` and ``weights`` configure the
    DG interior-penalty formulation; CG ignores them (they are sent either way,
    which is harmless).
    """
    type: str = "cg"
    reduction: float = 1.0e-10
    edge_norm_type: str = "houston"
    penalty: int = 20
    scheme: str = "sipg"
    weights: str = "tensorOnly"
    intorderadd: int = 5
    post_process_meg: bool = True
    subtract_mean: bool = True
    # TBB threads each *worker process* may use. DUNEuro defaults to
    # tbb::task_arena::automatic, i.e. a full-machine thread pool per process —
    # so the local multi-core path (one process per coil chunk) had every one of
    # its workers claiming every core, ~100 threads each. On a 12-core machine
    # that is ~1200 threads for 12 cores: each worker got half a core and the
    # rest went to contention. 1 is right whenever the parallelism is already
    # across processes; raise it to the cores-per-task the scheduler granted
    # (Myriad requests 4 with -pe smp 4). 0 restores DUNEuro's automatic.
    threads_per_process: int = 1

    def __post_init__(self) -> None:
        if self.threads_per_process < 0:
            raise ConfigError(
                "[forward.solver] threads_per_process must be >= 0 "
                "(0 = let DUNEuro decide)"
            )
        if self.type not in SOLVER_TYPES:
            raise ConfigError(
                f"[forward.solver] type must be one of {list(SOLVER_TYPES)}; "
                f"got {self.type!r}"
            )


#: DUNEuro CG source models this pipeline knows how to configure. DUNEuro's
#: ``CGSourceModelFactory`` also accepts ``subtraction``, ``local_subtraction``,
#: ``whitney``, ``patch_based_venant``, ``spatial_venant`` and
#: ``truncated_spatial_venant``; those need parameters this config does not
#: carry, so they are rejected here rather than failing deep inside C++.
SOURCE_MODEL_TYPES: tuple[str, ...] = (
    "partial_integration", "venant", "multipolar_venant",
)

#: Of those, the ones DUNEuro's *DG* source-model factory also accepts. Its
#: full list is ``partial_integration``, ``patch_based_venant``,
#: ``subtraction``, ``local_subtraction`` and ``truncated_spatial_venant`` —
#: notably not the vertex-based ``venant``, whose monopoles live on vertices
#: that a DG space has no unknowns on.
DG_SOURCE_MODELS: tuple[str, ...] = ("partial_integration",)


@dataclass(frozen=True)
class SourceModelCfg:
    """How a point dipole is realised as a FEM right-hand side.

    A mathematical point dipole is a singularity: it has no representation in
    a finite-dimensional FE space, so every FEM forward solver needs a rule for
    turning it into a load vector. Two are wired up here:

      * ``partial_integration`` (default) — integrate the divergence by parts
        onto the test functions, giving a load only on the vertices of the
        element containing the dipole. Cheapest; the historical default of this
        pipeline and the model every leadfield in ``outputs/`` was solved with.
      * ``venant`` — the St. Venant condition (Buchner et al. 1997): replace
        the dipole by monopoles on the vertices of an element patch around it,
        with strengths fitted (least squares, distance-weighted regularisation)
        so the patch reproduces the dipole's moments up to
        ``number_of_moments``. Spreads the source over more nodes, which is
        better behaved for sources sitting close to a conductivity jump.
        ``multipolar_venant`` is the same machinery with a multipolar rather
        than monopolar load.

    The Venant fields below map onto DUNEuro's parameter names
    (``numberOfMoments``, ``referenceLength``, …) and are ignored by
    ``partial_integration``. ``reference_length_mm`` is in millimetres because
    DUNEuro runs in mm-mode here (see ``sigma_unit_scale``), as are the patch
    extents implied by ``initialization``/``extensions``.

    ``restrict: true`` keeps the monopole patch inside the tissue compartment
    the dipole sits in, so monopoles never leak across a conductivity jump.
    That is usually what you want — but for a *thin* compartment (a vagus nerve
    only one or two elements across) it can leave too few vertices to fit the
    moments well; if a Venant run looks noisy on nerve sources, that is the
    first knob to check.
    """
    type: str = "partial_integration"
    number_of_moments: int = 3
    reference_length_mm: float = 20.0
    weighting_exponent: int = 1
    relaxation_factor: float = 1.0e-6
    mixed_moments: bool = True
    restrict: bool = True
    initialization: str = "closest_vertex"   # or "single_element"
    extensions: str = "vertex"               # "vertex", "intersection", or ""
    intorderadd: int = 2

    def __post_init__(self) -> None:
        if self.type not in SOURCE_MODEL_TYPES:
            raise ConfigError(
                f"[forward.source_model] type must be one of "
                f"{list(SOURCE_MODEL_TYPES)}; got {self.type!r}"
            )
        if self.initialization not in ("closest_vertex", "single_element"):
            raise ConfigError(
                "[forward.source_model] initialization must be 'closest_vertex' "
                f"or 'single_element'; got {self.initialization!r}"
            )
        # An empty `extensions:` in YAML (and `--set ...extensions=`) arrives as
        # None, which the str coercer renders "None" — spell every way of
        # saying "no patch extension" as the empty string DUNEuro expects.
        if self.extensions is None or str(self.extensions).strip() in ("", "None", "none", "null"):
            object.__setattr__(self, "extensions", "")
        for ext in self.extensions.split():
            if ext not in ("vertex", "intersection"):
                raise ConfigError(
                    "[forward.source_model] extensions must be a space-separated "
                    "list of 'vertex'/'intersection' (or empty); got "
                    f"{self.extensions!r}"
                )
        if self.number_of_moments < 1:
            raise ConfigError("[forward.source_model] number_of_moments must be >= 1")
        if self.weighting_exponent >= self.number_of_moments:
            # DUNEuro asserts this internally; catching it here gives a message
            # instead of an abort inside C++.
            raise ConfigError(
                "[forward.source_model] weighting_exponent must be < "
                f"number_of_moments; got {self.weighting_exponent} >= "
                f"{self.number_of_moments}"
            )
        if self.reference_length_mm <= 0:
            raise ConfigError("[forward.source_model] reference_length_mm must be > 0")


@dataclass(frozen=True)
class ForwardValidate:
    require_finite: bool = True
    max_abs_fT_per_nAm: float = 1.0e6


def source_tissue_labels(source_tissue: str) -> list[str]:
    """Split a comma-separated ``forward.source_tissue`` into tissue labels.

    Shared by the source sampler and the conductivity builder so the tissues
    that get *sampled* and the tissues that drive the conductivity model can
    never disagree.
    """
    return [t.strip() for t in source_tissue.split(",") if t.strip()]


@dataclass(frozen=True)
class MuscleAnisotropyCfg:
    """Anisotropic (fibre-aligned) conductivity for the muscle compartment.

    Skeletal muscle conducts several-fold better *along* its fibres than
    *across* them. When active, the muscle tissue is given a per-element
    conductivity tensor ``σ = σ_⊥·I + (σ_∥ − σ_⊥)·ê⊗ê`` with ``ê`` the local
    fibre axis (per-muscle long axis, the same fibre proxy used for source
    orientation). All other tissues stay isotropic.

    ``mode`` decides when it applies — there is no flag to remember:

      * ``"auto"`` (default) — on iff ``muscle`` is one of the forward
        ``source_tissue`` labels. Muscle runs get the tensor; vagus/spine runs
        take the scalar path, byte-identical to before.
      * ``"on"`` / ``"off"`` — force it, for an isotropic-vs-anisotropic A/B.

    Note ``auto`` is a *reproducibility* rule, not a physical one: muscle sits
    in the volume conductor of every run, so a physics purist would enable it
    always. Tying it to the source keeps previously-solved vagus/spine
    leadfields comparable; use ``"on"`` if you want it everywhere.

    Defaults: σ_∥ = 0.40, σ_⊥ = 0.10 S/m (≈4:1), bracketing the isotropic
    mean 0.35 S/m used before (Gabriel et al. 1996; Rush et al. 1963).
    """
    mode: str = "auto"
    sigma_long_sm: float = 0.40
    sigma_trans_sm: float = 0.10

    def __post_init__(self) -> None:
        # YAML 1.1 parses bare ``on``/``off`` as booleans, so both ``mode: on``
        # in a config file and ``--set ...mode=on`` on the command line arrive
        # as True/False rather than the strings the user typed — and the scalar
        # coercer, seeing this field annotated ``str``, may have already turned
        # those into ``"True"``/``"False"``. Normalise every spelling to the
        # canonical trio rather than rejecting what the docs tell people to write.
        mode = self.mode
        if isinstance(mode, bool):
            mode = "on" if mode else "off"
        elif isinstance(mode, str):
            mode = {"true": "on", "false": "off"}.get(mode.strip().lower(),
                                                       mode.strip().lower())
        object.__setattr__(self, "mode", mode)
        if mode not in ("auto", "on", "off"):
            raise ConfigError(
                "[forward.muscle_anisotropy] mode must be 'auto', 'on' or 'off'; "
                f"got {self.mode!r}"
            )

    def active_for(self, source_tissue: str) -> bool:
        """Whether the fibre-aligned tensor applies to this source target."""
        if self.mode != "auto":
            return self.mode == "on"
        return "muscle" in source_tissue_labels(source_tissue)


@dataclass(frozen=True)
class ForwardCfg:
    conductivities_sm: dict[str, float]
    sigma_unit_scale: float
    source_spacing_mm: float
    source_tissue: str
    duneuro_path: Path | None
    solver: SolverCfg
    validate: ForwardValidate
    muscle_anisotropy: MuscleAnisotropyCfg = field(default_factory=MuscleAnisotropyCfg)
    source_model: SourceModelCfg = field(default_factory=SourceModelCfg)

    def __post_init__(self) -> None:
        # CG and DG have *different* source-model factories in DUNEuro, and the
        # DG one has no vertex-based Venant (its unknowns are not on vertices).
        # Caught here because otherwise it is a C++ throw — after the mesh is
        # loaded and the transfer matrix has started, which on the cluster means
        # an hour of an array job to find out.
        if (self.solver.type == "dg"
                and self.source_model.type not in DG_SOURCE_MODELS):
            raise ConfigError(
                f"[forward] solver.type='dg' cannot use "
                f"source_model.type={self.source_model.type!r}. DUNEuro's DG "
                f"source-model factory supports {list(DG_SOURCE_MODELS)} of the "
                "models this pipeline configures (the vertex-based Venant models "
                "are CG-only). Use solver.type='cg', or "
                "source_model.type='partial_integration'."
            )
    # Optional explicit dipole positions (mm). When non-empty, the forward
    # solve uses these instead of geometry-derived ``vagus_sources`` sampling
    # — this is how the GUI's clicked source points reach the solver.
    point_sources: tuple[tuple[float, float, float], ...] = ()
    # Optional vertebral level restricting where dipoles are sampled: sampling
    # still runs at ``source_spacing_mm``, but only the positions inside that
    # band are kept. Accepts one level ("c7") or an inclusive range spanning
    # several ("c1-c7" for the cervical vagus, whose 70 nA·m reference and
    # analytic ladder are cervical-specific while the nerve mesh runs from
    # below T12 to above C1). A range's band is the union of its endpoints'
    # own measured bands, so it follows this anatomy rather than an assumed
    # proportion, exactly as a single level does. This is
    # deliberately NOT ``electrodes.target_level``, which every spine run
    # already sets via SOURCE_TARGETS to place the electrode patch — reusing it
    # here would silently narrow every existing whole-cord solve to C7. Default
    # None = sample the whole tissue, exactly as before.
    source_level: str | None = None
    # Local parallelism for the forward solve. The coil array is split into
    # this many independent chunks, each solved in its own process (mirroring
    # the cluster array-job path) and stitched back together. 0 = use all
    # available CPU cores; 1 = serial (single process). Default 0.
    local_workers: int = 0


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
    # Optional vertebral level (e.g. "c7") to centre the patch on, overriding the
    # fractional slab above. Set per source-target in SOURCE_TARGETS (spine → c7).
    target_level: str | None = None
    label_prefix: str = "elec"
    n_contacts: int = 1000           # whole_body: total contact count
    sample_seed: int = 0             # whole_body: RNG seed for skin sampling


@dataclass(frozen=True)
class NoiseCfg:
    # Defaults match QuSpin Gen-3 OPM and Malliaras-group PEDOT:PSS textile
    # electrode hardware. See ``configs/default.yaml`` for citations.
    opm_intrinsic_fT_sqrtHz: float = 7.0
    eeg_amplifier_uV_sqrtHz: float = 0.1
    eeg_electrode_skin_kohm: float = 10.0
    # Recording passband. Noise integrates over ``band_hi - band_lo``, so the
    # filter settings are part of the detectability answer rather than an
    # afterthought: a broadband 0-1000 Hz figure overstates σ by ~1.5x against
    # the 30-500 Hz band evoked-potential recording actually uses.
    # ``bandwidth_hz``, when set, overrides the pair (legacy single-number
    # form, kept so an existing config or --set keeps working).
    band_lo_hz: float = 30.0
    band_hi_hz: float = 500.0
    bandwidth_hz: float | None = None

    @property
    def effective_bandwidth_hz(self) -> float:
        """Noise-integration bandwidth in Hz."""
        if self.bandwidth_hz is not None:
            return float(self.bandwidth_hz)
        bw = float(self.band_hi_hz) - float(self.band_lo_hz)
        if bw <= 0:
            raise ConfigError(
                f"[noise] band_hi_hz ({self.band_hi_hz}) must exceed "
                f"band_lo_hz ({self.band_lo_hz})"
            )
        return bw

    @property
    def band_label(self) -> str:
        """Human-readable passband for figure captions."""
        if self.bandwidth_hz is not None:
            return f"BW {self.bandwidth_hz:g} Hz"
        return f"{self.band_lo_hz:g}–{self.band_hi_hz:g} Hz"


@dataclass(frozen=True)
class ClusterCfg:
    n_chunks: int = 32


@dataclass(frozen=True)
class SensitivityCfg:
    perturbations: tuple[float, ...] = (0.8, 1.0, 1.25)
    tissues: tuple[str, ...] = ("bone", "skin")


@dataclass(frozen=True)
class AnalyticCfg:
    """Analytic forward-model ladder (Biot–Savart → sphere → FEM).

    Target-agnostic: these describe the concentric-circle approximation of
    whatever elongated structure ``forward.source_tissue`` selects, so they
    must be set per target rather than assumed. The defaults happen to match
    the cervical-vagus literature (Bu et al. 2024) but carry no special
    status — change them in YAML when the target changes.
    """

    # Nominal distances from the structure's central axis.
    source_axis_mm: float = 40.0
    sensor_axis_mm: float = 58.5
    # Half-width of the band around those distances within which the
    # single-sphere solution is treated as a faithful baseline.
    band_tolerance_mm: float = 30.0
    # A source/coil pair counts as "sphere-silent" when the sphere solution
    # falls below this fraction of the Biot–Savart value.
    silent_rel_threshold: float = 0.01
    # Bootstrap settings for the CI on the FEM/sphere peak ratio.
    bootstrap_n: int = 2000
    bootstrap_seed: int = 0


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
    analytic: AnalyticCfg
    reproducibility: ReproCfg
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


# ── source targets ──────────────────────────────────────────────────────────
#
# Canonical definition of the source regions the pipeline can target. This is
# the single source of truth for the ``--source-target`` CLI flag; it MUST stay
# in sync with ``inob__source_target`` in ``cluster/lib.sh`` (same keys, same
# tissue mappings). Each entry maps a slug to:
#
#   tissues    the FEM ``source_tissue`` string whose tets are sampled;
#   label      the human-readable name used in figure titles;
#   electrodes the tissue the HD surface-electrode patch is centred over;
#   level      optional vertebral level to centre the patch on (e.g. "c7").
#
# ``electrodes`` matters because the EEG patch is small (32 contacts over a few
# cm) and directional: a patch sited over the cervical vagus reads a spinal-cord
# source badly, and the resulting MEG-vs-EEG comparison measures patch placement
# rather than modality. For combined targets the patch follows the deeper/larger
# structure, which is the one the array must be sited for.
SOURCE_TARGETS: dict[str, dict[str, str]] = {
    "vagus":       {"tissues": "vagus_left",             "label": "vagus",
                    "electrodes": "vagus_left"},
    "spine":       {"tissues": "spinal_cord",            "label": "spine",
                    "electrodes": "spinal_cord",         "level": "c7"},
    "spine_vagus": {"tissues": "spinal_cord,vagus_left", "label": "spine + vagus",
                    "electrodes": "spinal_cord",         "level": "c7"},
    "muscle":      {"tissues": "muscle",                 "label": "muscle",
                    "electrodes": "muscle"},
    "spine_muscle": {"tissues": "spinal_cord,muscle",    "label": "spine + muscle",
                    "electrodes": "spinal_cord",         "level": "c7"},
}

# Leadfield filename prefixes stripped to recover the ``--source-target`` slug.
# Longest first so ``duneuro_eeg_leadfield_`` wins over ``duneuro_leadfield_``.
_LEADFIELD_PREFIXES = (
    "duneuro_eeg_leadfield_", "duneuro_leadfield_",
    "duneuro_eeg_leadfield", "duneuro_leadfield",
)


def source_target_tag(cfg: Config) -> str:
    """The ``--source-target`` slug carried by the forward leadfield filename.

    ``outputs/forward/duneuro_leadfield_spine_vagus.npz`` → ``"spine_vagus"``.
    Returns ``""`` for an untagged/plain ``duneuro_leadfield.npz``.
    """
    stem = cfg.outputs.forward_npz.stem            # e.g. duneuro_leadfield_spine
    for prefix in _LEADFIELD_PREFIXES:
        if stem.startswith(prefix):
            return stem[len(prefix):].lstrip("_")
    return ""


def replace_source_model(cfg: Config, **changes: Any) -> Config:
    """Return ``cfg`` with ``forward.source_model`` fields replaced.

    Lets a caller sweep source models over one loaded config — an A/B on the
    same mesh, same sensors, same conductivities — without re-reading YAML or
    reaching into two levels of frozen dataclass by hand.
    """
    sm = replace(cfg.forward.source_model, **changes)
    return replace(cfg, forward=replace(cfg.forward, source_model=sm))


def tag_path(path: Path, tag: str) -> Path:
    """Insert ``_<tag>`` into a filename before its suffix; append for dirs.

    ``outputs/forward/chunks`` + ``spine`` → ``outputs/forward/chunks_spine``
    ``outputs/detectability.png`` + ``spine`` → ``outputs/detectability_spine.png``

    An empty tag returns ``path`` unchanged, so untagged legacy runs keep the
    paths they have always used. Idempotent: re-tagging with the same tag is a
    no-op, so this is safe to apply to a path that already carries it.
    """
    if not tag:
        return path
    if path.stem.endswith(f"_{tag}") or path.name.endswith(f"_{tag}"):
        return path
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def target_output(cfg: Config, filename: str) -> Path:
    """Default path under ``outputs/`` for a figure that depends on the target.

    Every analysis artefact derived from a leadfield must carry the target slug
    or a spine run silently overwrites the vagus figure of the same name. This
    is the single place that decides how, so the naming cannot drift between
    the modules that produce those artefacts.
    """
    return cfg.outputs.base / tag_path(Path(filename), source_target_tag(cfg)).name


def source_region_label(cfg: Config) -> str:
    """Human-readable label for the analysed source region.

    Single source of truth for figure titles / axis labels across the
    pipeline, so every visualisation names whatever source is actually being
    analysed instead of a hardcoded "vagus". Derived from the forward
    leadfield filename, which carries the ``--source-target`` slug
    (see :data:`SOURCE_TARGETS`). Known slugs use their curated label
    (``spine_vagus`` → "spine + vagus"); an unrecognised-but-present slug is
    prettified generically; an untagged/plain filename falls back to "nerve".
    """
    tag = source_target_tag(cfg)
    if not tag:
        return "nerve"
    if tag in SOURCE_TARGETS:
        return SOURCE_TARGETS[tag]["label"]
    return tag.replace("_", " + ")


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

def _check_keys(
    name: str, expected: set[str], got: set[str],
    *, optional: set[str] | None = None,
) -> None:
    """Validate a config level's keys.

    ``optional`` names sections that are recognised but may be omitted —
    used for blocks whose dataclass supplies a complete set of defaults, so
    that adding one does not invalidate every existing YAML.
    """
    optional = optional or set()
    missing = expected - got - optional
    extra = got - expected
    msgs = []
    if missing:
        msgs.append(f"missing keys: {sorted(missing)}")
    if extra:
        msgs.append(f"unknown keys: {sorted(extra)}")
    if msgs:
        raise ConfigError(f"[{name}] " + "; ".join(msgs))


def _coerce_scalar(name: str, key: str, ann: Any, value: Any) -> Any:
    """Validate/coerce a scalar config value against its field annotation.

    Only plain ``int``/``float``/``bool``/``str`` fields are checked — anything
    with a compound annotation (unions, tuples, dicts, ``Path``) is passed
    through untouched, preserving the loader's existing behaviour for those.

    The point is to catch a mistyped ``--set`` or YAML value (e.g.
    ``analytic.bootstrap_n=notanumber``) at load time with a message that
    names the field, rather than letting a string flow into numeric code and
    fail three stages later.
    """
    # With `from __future__ import annotations`, field types are strings.
    ann_s = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
    if ann_s not in {"int", "float", "bool", "str"}:
        return value

    def bad(expected: str) -> ConfigError:
        return ConfigError(
            f"[{name}] {key}: expected {expected}, got {value!r} "
            f"({type(value).__name__})"
        )

    if ann_s == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise bad("a boolean (true/false)")

    if ann_s == "int":
        if isinstance(value, bool):
            raise bad("an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                raise bad("an integer") from None
        raise bad("an integer")

    if ann_s == "float":
        if isinstance(value, bool):
            raise bad("a number")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                raise bad("a number") from None
        raise bad("a number")

    # str: accept a string; coerce other scalars, but reject collections.
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict, tuple)):
        raise bad("a string")
    return str(value)


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
    coerced = {
        k: _coerce_scalar(name, k, all_fields[k].type, v) for k, v in data.items()
    }
    return cls(**coerced)


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
    aniso = _build_dataclass(
        MuscleAnisotropyCfg, d.get("muscle_anisotropy", {}), "forward.muscle_anisotropy"
    )
    src_model = _build_dataclass(
        SourceModelCfg, d.get("source_model", {}), "forward.source_model"
    )
    body = {k: v for k, v in d.items()
            if k not in ("solver", "validate", "muscle_anisotropy", "source_model")}
    duneuro_path = body.get("duneuro_path")
    body["duneuro_path"] = Path(duneuro_path).expanduser() if duneuro_path else None
    if body.get("point_sources"):
        body["point_sources"] = tuple(
            tuple(float(c) for c in p) for p in body["point_sources"]
        )
    # allow_defaults=True so the optional point_sources field may be omitted;
    # the other forward fields have no defaults and so remain required.
    return _build_dataclass(
        ForwardCfg,
        {**body, "solver": solver, "validate": val, "muscle_anisotropy": aniso,
         "source_model": src_model},
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
        "sensitivity", "analytic", "reproducibility",
    }
    # Sections whose dataclass carries a complete set of defaults may be
    # omitted entirely, so that adding one does not invalidate existing YAMLs.
    _check_keys("(top level)", expected_top, set(raw.keys()),
                optional={"analytic"})

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
        analytic=_build_dataclass(
            AnalyticCfg, raw.get("analytic", {}), "analytic"
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
