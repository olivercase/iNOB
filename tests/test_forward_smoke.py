"""DUNEuro forward-solve smoke test.

Skips automatically when ``duneuropy`` is not importable, so CI without the
DUNE toolchain still passes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inob.config import load_config
from inob.forward.duneuro_driver import (
    attach_coils,
    build_driver,
    build_orthogonal_dipoles,
    import_duneuro,
)
from inob.io.hdf5 import FemMesh

pytest.importorskip("duneuropy", reason="duneuropy not built locally")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"


def _build_synthetic_fem() -> FemMesh:
    """A tiny single-tet 'vagus_left' FEM (nodes in mm)."""
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [0.0, 50.0, 0.0],
            [0.0, 0.0, 50.0],
        ],
        dtype=np.float64,
    )
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    tissue = np.array([1], dtype=np.int32)
    return FemMesh(nodes, tets, tissue, tissue_labels=("vagus_left",), unit="mm")


@pytest.mark.duneuro
def test_smoke_forward_on_single_tet() -> None:
    cfg = load_config(DEFAULT_CFG)
    fem = _build_synthetic_fem()
    src_pos = np.array([[10.0, 10.0, 10.0]])
    coilpos = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
        ]
    )
    coilori = coilpos / np.linalg.norm(coilpos, axis=1, keepdims=True)

    dp = import_duneuro(cfg)
    driver, driver_cfg, _cond = build_driver(cfg, fem)
    attach_coils(driver, dp, coilpos, coilori)
    T_raw, _ = driver.computeMEGTransferMatrix(driver_cfg)
    T = np.array(T_raw)
    assert T.shape[0] == len(coilpos)

    dipoles = build_orthogonal_dipoles(dp, src_pos)
    driver_cfg["source_model"] = {"type": "partial_integration"}
    fields_raw, _ = driver.applyMEGTransfer(T, dipoles, driver_cfg)
    L = np.column_stack([np.asarray(f) for f in fields_raw])
    assert L.shape == (3, 3)  # 3 channels, 1 source, 3 moments
    assert np.isfinite(L).all()
