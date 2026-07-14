"""Tests for stitching per-chunk leadfields into one NPZ (no DUNEuro required)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inob.config import load_config
from inob.forward.reduce import reduce_chunks
from inob.io.hdf5 import FemMesh, SensorArray, save_fem, save_sensors
from inob.io.npz import load_leadfield

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"


def _fem_single_source() -> FemMesh:
    # Two "vagus_left" tets a couple mm apart in Z: with the default
    # spacing_mm (5.0) their combined Z-extent falls in a single slab, so
    # vagus_sources() yields exactly one source position. Plus a "skin"
    # tet for realism.
    nodes = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 3.0],
        [10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0], [10.0, 10.0, 11.0],
    ], dtype=np.float64)
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=np.int32)
    tissue = np.array([1, 1, 2], dtype=np.int32)
    return FemMesh(nodes=nodes, tets=tets, tissue=tissue,
                    tissue_labels=("vagus_left", "skin"), unit="mm")


def _sensors(n: int) -> SensorArray:
    pos = np.column_stack([np.arange(n, dtype=np.float64), np.zeros(n), np.zeros(n)])
    ori = np.tile([1.0, 0.0, 0.0], (n, 1))
    labels = tuple(f"mag-{i:04d}-R" for i in range(n))
    return SensorArray(coilpos=pos, coilori=ori, labels=labels,
                        chantype=tuple(["megmag"] * n), chanunit=tuple(["T"] * n),
                        unit="mm")


def _setup(tmp_path: Path, n_chan: int = 6):
    cfg = load_config(DEFAULT_CFG, project_root=tmp_path)
    fem = _fem_single_source()
    save_fem(cfg.outputs.fem_mat, fem)
    sensors = _sensors(n_chan)
    save_sensors(cfg.outputs.sensors_mat, sensors)
    return cfg, fem, sensors


def test_reduce_chunks_raises_if_no_chunks(tmp_path: Path) -> None:
    cfg, _, _ = _setup(tmp_path)
    with pytest.raises(FileNotFoundError, match="no L_chunk"):
        reduce_chunks(cfg)


def test_reduce_chunks_stitches_and_saves(tmp_path: Path) -> None:
    cfg, fem, sensors = _setup(tmp_path, n_chan=6)
    n_src = 1  # one Z-slab given the single vagus_left tet
    n_cols = 3 * n_src

    chunks_dir = cfg.outputs.forward_chunks_dir
    chunks_dir.mkdir(parents=True, exist_ok=True)

    idx_a = np.array([0, 2, 4])
    idx_b = np.array([1, 3, 5])
    # keep values well under the fT/nAm amplitude bound after the x1e6 scale
    L_a = np.arange(len(idx_a) * n_cols, dtype=np.float64).reshape(len(idx_a), n_cols) * 1e-8
    L_b = -np.arange(len(idx_b) * n_cols, dtype=np.float64).reshape(len(idx_b), n_cols) * 1e-8
    np.save(chunks_dir / "L_chunk_000.npy", L_a)
    np.save(chunks_dir / "coil_idx_000.npy", idx_a)
    np.save(chunks_dir / "L_chunk_001.npy", L_b)
    np.save(chunks_dir / "coil_idx_001.npy", idx_b)

    out = reduce_chunks(cfg)
    assert out == cfg.outputs.forward_npz
    assert out.exists()

    lf = load_leadfield(out)
    assert lf.L.shape == (6, n_cols)
    np.testing.assert_allclose(lf.L[idx_a], L_a)
    np.testing.assert_allclose(lf.L[idx_b], L_b)
    np.testing.assert_allclose(lf.L_fT_per_nAm, lf.L * 1e6)
    assert lf.source_pos.shape == (n_src, 3)
    assert lf.channel_names == sensors.labels
    assert lf.tissue_labels == fem.tissue_labels


def test_reduce_chunks_missing_index_file_raises(tmp_path: Path) -> None:
    cfg, _, _ = _setup(tmp_path, n_chan=3)
    chunks_dir = cfg.outputs.forward_chunks_dir
    chunks_dir.mkdir(parents=True, exist_ok=True)
    np.save(chunks_dir / "L_chunk_000.npy", np.zeros((3, 3)))
    with pytest.raises(FileNotFoundError, match="missing index"):
        reduce_chunks(cfg)


def test_reduce_chunks_shape_mismatch_raises(tmp_path: Path) -> None:
    cfg, _, _ = _setup(tmp_path, n_chan=3)
    chunks_dir = cfg.outputs.forward_chunks_dir
    chunks_dir.mkdir(parents=True, exist_ok=True)
    np.save(chunks_dir / "L_chunk_000.npy", np.zeros((2, 3)))   # 2 rows...
    np.save(chunks_dir / "coil_idx_000.npy", np.array([0, 1, 2]))  # ...3 indices
    with pytest.raises(ValueError, match="shape"):
        reduce_chunks(cfg)


def test_reduce_chunks_missing_rows_raises(tmp_path: Path) -> None:
    cfg, _, _ = _setup(tmp_path, n_chan=4)
    chunks_dir = cfg.outputs.forward_chunks_dir
    chunks_dir.mkdir(parents=True, exist_ok=True)
    # only cover 3 of 4 channels
    idx = np.array([0, 1, 2])
    np.save(chunks_dir / "L_chunk_000.npy", np.zeros((3, 3)))
    np.save(chunks_dir / "coil_idx_000.npy", idx)
    with pytest.raises(RuntimeError, match="missing rows"):
        reduce_chunks(cfg)


def test_reduce_chunks_source_count_mismatch_raises(tmp_path: Path) -> None:
    cfg, _, _ = _setup(tmp_path, n_chan=2)
    chunks_dir = cfg.outputs.forward_chunks_dir
    chunks_dir.mkdir(parents=True, exist_ok=True)
    idx = np.array([0, 1])
    # 6 columns implies 2 sources, but the FEM/cfg only produces 1
    np.save(chunks_dir / "L_chunk_000.npy", np.zeros((2, 6)))
    np.save(chunks_dir / "coil_idx_000.npy", idx)
    with pytest.raises(RuntimeError, match="source count mismatch"):
        reduce_chunks(cfg)
