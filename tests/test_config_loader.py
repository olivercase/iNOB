"""Config loader tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from vagus_fm.config import (
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
    assert cfg.fem.tissues == ("vagus_left", "vagus_right", "bone", "skin")


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
