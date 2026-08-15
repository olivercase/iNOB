"""Tests for the configurable DUNEuro source model (no DUNEuro required).

The source model decides how a point dipole — a singularity with no exact
representation in a finite element space — becomes a FEM right-hand side.
These tests pin the YAML → DUNEuro-config translation and the guard rails, so
a bad Venant setup fails at config load with a readable message rather than
inside C++ hours into a cluster array job.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from inob.config import ConfigError, SourceModelCfg, load_config
from inob.forward.duneuro_driver import build_source_model_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"


def test_default_is_partial_integration_and_carries_no_venant_keys() -> None:
    # The historic default: every leadfield already in outputs/ was solved with
    # partial integration, so it must stay the default until Venant is
    # validated against the sphere.
    cfg = load_config(DEFAULT_CFG)
    assert cfg.forward.source_model.type == "partial_integration"
    assert build_source_model_config(cfg) == {"type": "partial_integration"}


@pytest.mark.parametrize("model", ["venant", "multipolar_venant"])
def test_venant_emits_every_key_duneuro_requires(model: str) -> None:
    # DUNEuro's MonopolarVenant/MultipolarVenant ctors call params.get<T>(...)
    # with no default for each of these, and the element patch needs
    # initialization + restrict, so a missing key is a hard throw.
    cfg = load_config(DEFAULT_CFG, overrides=[f"forward.source_model.type={model}"])
    d = build_source_model_config(cfg)
    assert d["type"] == model
    assert set(d) == {
        "type", "numberOfMoments", "referenceLength", "weightingExponent",
        "relaxationFactor", "mixedMoments", "restrict", "initialization",
        "extensions", "intorderadd",
    }
    # ParameterTree parses strings; bools must be lowercase true/false.
    assert all(isinstance(v, str) for v in d.values())
    assert d["mixedMoments"] in ("true", "false")
    assert d["restrict"] in ("true", "false")
    assert d["initialization"] == "closest_vertex"


def test_venant_parameters_are_overridable() -> None:
    cfg = load_config(DEFAULT_CFG, overrides=[
        "forward.source_model.type=venant",
        "forward.source_model.number_of_moments=4",
        "forward.source_model.reference_length_mm=5.0",
        "forward.source_model.restrict=false",
        "forward.source_model.extensions=",
    ])
    d = build_source_model_config(cfg)
    assert d["numberOfMoments"] == "4"
    assert d["referenceLength"] == "5.0"
    assert d["restrict"] == "false"
    assert d["extensions"] == ""


def test_unknown_source_model_is_rejected() -> None:
    with pytest.raises(ConfigError, match="source_model"):
        load_config(DEFAULT_CFG,
                    overrides=["forward.source_model.type=subtraction"])


def test_weighting_exponent_must_be_below_number_of_moments() -> None:
    # DUNEuro asserts this internally; catch it at load time instead.
    with pytest.raises(ConfigError, match="weighting_exponent"):
        SourceModelCfg(type="venant", number_of_moments=2, weighting_exponent=2)


def test_bad_patch_settings_are_rejected() -> None:
    with pytest.raises(ConfigError, match="initialization"):
        SourceModelCfg(type="venant", initialization="nearest")
    with pytest.raises(ConfigError, match="extensions"):
        SourceModelCfg(type="venant", extensions="faces")


def test_dg_reaches_duneuro_as_the_solver_type() -> None:
    import numpy as np

    from inob.forward.duneuro_driver import build_driver_config
    from inob.io.hdf5 import FemMesh

    cfg = load_config(DEFAULT_CFG, overrides=["forward.solver.type=dg"])
    nodes = np.eye(4, 3) * 50.0
    nodes[0] = 0
    fem = FemMesh(nodes, np.array([[0, 1, 2, 3]], dtype=np.int32),
                  np.array([1], dtype=np.int32), ("vagus_left",), "mm")
    d = build_driver_config(cfg, fem, np.array([3e-4]))
    assert d["solver_type"] == "dg"
    # The interior-penalty settings CG ignores are the ones DG actually reads.
    assert d["solver"]["scheme"] == "sipg"
    assert d["solver"]["penalty"] == "20"


def test_unknown_solver_type_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"forward\.solver"):
        load_config(DEFAULT_CFG, overrides=["forward.solver.type=udg"])


def test_dg_with_a_venant_source_model_is_refused_at_load() -> None:
    # DUNEuro's DG source-model factory has no vertex-based Venant; without
    # this guard the combination throws inside C++ after the transfer matrix
    # has already started.
    with pytest.raises(ConfigError, match="dg"):
        load_config(DEFAULT_CFG, overrides=[
            "forward.solver.type=dg",
            "forward.source_model.type=venant",
        ])


def test_dg_with_partial_integration_is_allowed() -> None:
    cfg = load_config(DEFAULT_CFG, overrides=[
        "forward.solver.type=dg",
        "forward.source_model.type=partial_integration",
    ])
    assert cfg.forward.solver.type == "dg"
