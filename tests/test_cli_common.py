"""CLI plumbing: shared argparse fragments + setup() config/logging bootstrap."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from inob.cli._common import DEFAULT_CONFIG, add_common_args, setup

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_common_args(p)
    return p


def test_add_common_args_defaults() -> None:
    args = _parser().parse_args([])
    assert args.config == Path(DEFAULT_CONFIG)
    assert args.overrides == []
    assert args.project_root is None
    assert args.log_file is None
    assert args.log_level in ("INFO", "DEBUG", "WARNING", "ERROR")


def test_add_common_args_parses_overrides_and_paths() -> None:
    args = _parser().parse_args([
        "--config", "configs/tiny_test.yaml",
        "--set", "forward.source_spacing_mm=3.0",
        "--set", "cluster.n_chunks=8",
        "--project-root", "/tmp/proj",
        "--log-level", "DEBUG",
        "--log-file", "/tmp/proj/run.log",
    ])
    assert args.config == Path("configs/tiny_test.yaml")
    assert args.overrides == ["forward.source_spacing_mm=3.0", "cluster.n_chunks=8"]
    assert args.project_root == Path("/tmp/proj")
    assert args.log_level == "DEBUG"
    assert args.log_file == Path("/tmp/proj/run.log")


def test_add_common_args_set_is_repeatable() -> None:
    args = _parser().parse_args(["--set", "a.b=1", "--set", "c.d=2"])
    assert args.overrides == ["a.b=1", "c.d=2"]


def test_setup_loads_config(tmp_path: Path) -> None:
    args = _parser().parse_args([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
    ])
    cfg = setup(args, log_prefix="unittest")
    assert cfg.project_root == tmp_path
    assert cfg.reproducibility.seed == 0


def test_setup_seeds_are_reproducible(tmp_path: Path) -> None:
    import random as random_mod

    import numpy as np

    args = _parser().parse_args([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
    ])
    setup(args, log_prefix="unittest")
    a, b = np.random.rand(), random_mod.random()

    args2 = _parser().parse_args([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
    ])
    setup(args2, log_prefix="unittest")
    assert np.random.rand() == a
    assert random_mod.random() == b


def test_setup_default_log_file_under_logs_dir(tmp_path: Path) -> None:
    args = _parser().parse_args([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
    ])
    cfg = setup(args, log_prefix="myprefix")
    logs = list(cfg.outputs.logs_dir.glob("myprefix-*.log"))
    assert len(logs) == 1


def test_setup_respects_explicit_log_file(tmp_path: Path) -> None:
    log_file = tmp_path / "custom" / "mine.log"
    args = _parser().parse_args([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--log-file", str(log_file),
    ])
    setup(args, log_prefix="x")
    assert log_file.exists()


def test_setup_applies_log_level(tmp_path: Path) -> None:
    args = _parser().parse_args([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--log-level", "WARNING",
    ])
    setup(args, log_prefix="lvl")
    assert logging.getLogger().level == logging.WARNING
    # restore a sane level for subsequent tests in the same process
    logging.getLogger().setLevel(logging.INFO)


def test_source_model_flag_is_sugar_for_the_config_field(tmp_path: Path) -> None:
    args = _parser().parse_args([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--source-model", "venant",
    ])
    cfg = setup(args, log_prefix="unittest")
    assert cfg.forward.source_model.type == "venant"
    # The Venant fit parameters keep their config defaults — the flag only
    # picks the model.
    assert cfg.forward.source_model.number_of_moments == 3


def test_muscle_anisotropy_flag_forces_the_mode(tmp_path: Path) -> None:
    args = _parser().parse_args([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--muscle-anisotropy", "on",
    ])
    cfg = setup(args, log_prefix="unittest")
    assert cfg.forward.muscle_anisotropy.mode == "on"
    # Forced on means on even for a non-muscle source, which is the whole point
    # of being able to force it.
    assert cfg.forward.muscle_anisotropy.active_for("vagus_left") is True


def test_forward_physics_flags_default_to_leaving_the_config_alone(
    tmp_path: Path,
) -> None:
    args = _parser().parse_args([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
    ])
    assert args.source_model is None
    assert args.muscle_anisotropy is None
    cfg = setup(args, log_prefix="unittest")
    assert cfg.forward.source_model.type == "partial_integration"
    assert cfg.forward.muscle_anisotropy.mode == "auto"


def test_later_set_override_wins_over_the_flag(tmp_path: Path) -> None:
    # --set is appended after the sugar, so an explicit --set is the last word.
    args = _parser().parse_args([
        "--config", str(TINY_CFG), "--project-root", str(tmp_path),
        "--source-model", "venant",
        "--set", "forward.source_model.number_of_moments=4",
    ])
    cfg = setup(args, log_prefix="unittest")
    assert cfg.forward.source_model.type == "venant"
    assert cfg.forward.source_model.number_of_moments == 4
