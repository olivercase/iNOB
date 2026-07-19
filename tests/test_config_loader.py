"""Config loader tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from inob.config import (
    Config,
    ConfigError,
    apply_overrides,
    load_config,
    parse_override,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def test_load_default() -> None:
    cfg = load_config(DEFAULT_CFG)
    assert isinstance(cfg, Config)
    assert cfg.forward.conductivities_sm["vagus_left"] == 0.30
    assert cfg.forward.conductivities_sm["bone"] == 0.0042
    assert cfg.forward.source_tissue == "vagus_left"
    assert cfg.forward.solver.scheme == "sipg"
    assert cfg.fem.tissues == (
        "vagus_left", "vagus_right", "blood_vessel", "spinal_cord", "muscle", "bone", "skin",
    )


def test_load_tiny() -> None:
    cfg = load_config(TINY_CFG)
    assert cfg.cluster.n_chunks == 2
    assert cfg.fem.tissues == ("vagus_left", "skin")
    # paths resolve relative to the repo root (project_root: .)
    assert cfg.outputs.geometry_mat.is_absolute()
    assert cfg.data.bone_dir.is_absolute()


def test_paths_are_resolved_against_project_root() -> None:
    cfg = load_config(DEFAULT_CFG)
    assert cfg.outputs.geometry_mat == cfg.project_root / "outputs/geometry/vagus_geometry.mat"
    assert cfg.data.torso_skin == (
        cfg.project_root / "data/torso/FJ2810_BP22617_FMA7163_Skin.stl"
    )


def test_dotted_override_scalar() -> None:
    cfg = load_config(DEFAULT_CFG, overrides=["forward.source_spacing_mm=3.0"])
    assert cfg.forward.source_spacing_mm == 3.0


def test_dotted_override_int() -> None:
    cfg = load_config(DEFAULT_CFG, overrides=["cluster.n_chunks=8"])
    assert cfg.cluster.n_chunks == 8


def test_dotted_override_nested_solver() -> None:
    cfg = load_config(DEFAULT_CFG, overrides=["forward.solver.intorderadd=7"])
    assert cfg.forward.solver.intorderadd == 7


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(DEFAULT_CFG.read_text() + "\nbogus_key: 42\n")
    with pytest.raises(ConfigError, match="bogus_key"):
        load_config(p)


def test_unknown_nested_key_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(DEFAULT_CFG, overrides=["forward.bogus_field=1"])


def test_missing_config_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("/no/such/path.yaml")


def test_frozen_config() -> None:
    cfg = load_config(DEFAULT_CFG)
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.forward.solver.intorderadd = 99   # type: ignore[misc]


def test_parse_override_yaml_value() -> None:
    path, val = parse_override("forward.solver.subtract_mean=false")
    assert path == ["forward", "solver", "subtract_mean"]
    assert val is False


def test_parse_override_requires_equals() -> None:
    with pytest.raises(ValueError, match=r"key\.path=value"):
        parse_override("foo")


def test_apply_overrides_creates_intermediate_dicts() -> None:
    data: dict = {}
    apply_overrides(data, ["a.b.c=1"])
    assert data == {"a": {"b": {"c": 1}}}


def test_analytic_block_loaded_from_yaml() -> None:
    cfg = load_config(DEFAULT_CFG)
    assert cfg.analytic.source_axis_mm == 40.0
    assert cfg.analytic.sensor_axis_mm == 58.5
    assert cfg.analytic.band_tolerance_mm == 30.0


def test_analytic_block_is_optional(tmp_path: Path) -> None:
    """A config predating the `analytic:` block must still load on defaults.

    The dataclass supplies every value, so omitting the section is legal —
    otherwise adding a section would invalidate every existing YAML.
    """
    text = DEFAULT_CFG.read_text()
    head, sep, _ = text.partition("\nanalytic:")
    assert sep, "default.yaml no longer has an analytic block to strip"
    _, _, tail = text.partition("\nreproducibility:")
    p = tmp_path / "no_analytic.yaml"
    p.write_text(head + "\nreproducibility:" + tail)
    cfg = load_config(p, project_root=REPO_ROOT)
    assert cfg.analytic.source_axis_mm == 40.0
    assert cfg.analytic.bootstrap_n == 2000


def test_analytic_override_reaches_config() -> None:
    """Every analytic parameter must be reachable from `--set`."""
    cfg = load_config(DEFAULT_CFG, overrides=[
        "analytic.source_axis_mm=12.5",
        "analytic.sensor_axis_mm=20.0",
        "analytic.band_tolerance_mm=5.0",
        "analytic.silent_rel_threshold=0.05",
        "analytic.bootstrap_n=10",
        "analytic.bootstrap_seed=7",
    ])
    assert cfg.analytic.source_axis_mm == 12.5
    assert cfg.analytic.sensor_axis_mm == 20.0
    assert cfg.analytic.band_tolerance_mm == 5.0
    assert cfg.analytic.silent_rel_threshold == 0.05
    assert cfg.analytic.bootstrap_n == 10
    assert cfg.analytic.bootstrap_seed == 7


def test_analytic_unknown_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad_analytic.yaml"
    p.write_text(DEFAULT_CFG.read_text() + "\n  not_a_real_knob: 1\n")
    with pytest.raises(ConfigError):
        load_config(p, project_root=REPO_ROOT)


def test_override_type_validation_rejects_bad_scalar() -> None:
    """A mistyped --set must fail at load, naming the field (not 3 stages later)."""
    for ov, needle in [
        ("analytic.bootstrap_n=notanumber", "bootstrap_n"),
        ("analytic.source_axis_mm=hello", "source_axis_mm"),
        ("forward.source_spacing_mm=[1,2,3]", "source_spacing_mm"),
        ("forward.solver.penalty=3.5", "penalty"),
    ]:
        with pytest.raises(ConfigError, match=needle):
            load_config(DEFAULT_CFG, overrides=[ov])


def test_override_type_validation_coerces_valid_scalar() -> None:
    cfg = load_config(DEFAULT_CFG, overrides=[
        "analytic.bootstrap_n=2.0",       # integral float -> int
        "analytic.source_axis_mm=40",     # int -> float
        "forward.solver.subtract_mean=false",  # str -> bool
    ])
    assert cfg.analytic.bootstrap_n == 2 and isinstance(cfg.analytic.bootstrap_n, int)
    assert cfg.analytic.source_axis_mm == 40.0
    assert cfg.forward.solver.subtract_mean is False
