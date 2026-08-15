"""Tests for the cluster array-job chunk worker (no DUNEuro required)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import inob.forward.chunk as chunk_mod
from inob.config import load_config
from inob.forward.chunk import run_chunk
from inob.io.hdf5 import FemMesh, SensorArray, save_fem, save_sensors

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"


class _FakeDp:
    @staticmethod
    def FieldVector3D(p):
        return tuple(p)

    @staticmethod
    def Dipole3d(pos, moment):
        return (tuple(pos), tuple(moment))


class _FakeDriver:
    def __init__(self, n_coils: int) -> None:
        self.n_coils = n_coils
        self.attached_coils = None

    def setCoilsAndProjections(self, coils, projs):
        self.attached_coils = (coils, projs)

    def computeMEGTransferMatrix(self, driver_cfg):
        return np.zeros((self.n_coils, 1)), None

    def applyMEGTransfer(self, T, dipoles, driver_cfg):
        fields = [np.full(self.n_coils, float(i)) for i in range(len(dipoles))]
        return fields, None

    def computeMEGPrimaryField(self, dipoles, driver_cfg):
        # zero primary keeps the chunk fixture's L values deterministic
        return [np.zeros(self.n_coils) for _ in dipoles]


def _fem_two_source_slabs() -> FemMesh:
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


@pytest.mark.parametrize("chunk_id,n_chunks", [(-1, 4), (4, 4), (5, 4)])
def test_run_chunk_out_of_range_raises_before_touching_disk(
    tmp_path: Path, chunk_id: int, n_chunks: int,
) -> None:
    cfg = load_config(DEFAULT_CFG, project_root=tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        run_chunk(cfg, chunk_id=chunk_id, n_chunks=n_chunks)
    # the guard fires before FEM/sensor loading, so no chunk dir is created
    assert not cfg.outputs.forward_chunks_dir.exists()


def test_run_chunk_valid_id_boundaries_pass_the_guard(
    tmp_path: Path, monkeypatch,
) -> None:
    """chunk_id == 0 and chunk_id == n_chunks - 1 must pass the range check
    and proceed to FEM loading (which then fails since no FEM file exists —
    proving the guard itself did not reject valid ids)."""
    cfg = load_config(DEFAULT_CFG, project_root=tmp_path)
    for chunk_id in (0, 3):
        with pytest.raises(FileNotFoundError):
            run_chunk(cfg, chunk_id=chunk_id, n_chunks=4)


def test_run_chunk_saves_slice_of_full_coil_array(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config(DEFAULT_CFG, project_root=tmp_path)
    save_fem(cfg.outputs.fem_mat, _fem_two_source_slabs())
    n_chan = 6
    sensors = _sensors(n_chan)
    save_sensors(cfg.outputs.sensors_mat, sensors)

    n_chunks = 2
    fake_driver = _FakeDriver(n_coils=n_chan // n_chunks)
    monkeypatch.setattr(chunk_mod, "import_duneuro", lambda cfg: _FakeDp)
    driver_calls = {}

    def fake_build_driver(cfg, fem, *, limit_threads=False):
        driver_calls["limit_threads"] = limit_threads
        return fake_driver, {"volume_conductor": {}}, np.array([3e-4, 4.3e-4])

    monkeypatch.setattr(chunk_mod, "build_driver", fake_build_driver)

    out_L, out_idx = run_chunk(cfg, chunk_id=0, n_chunks=n_chunks)
    # A chunk worker is one process of many, so it must cap its own TBB pool —
    # otherwise every worker claims every core and they fight for the machine.
    assert driver_calls["limit_threads"] is True
    assert out_L.is_file() and out_idx.is_file()
    L = np.load(out_L)
    idx = np.load(out_idx)
    np.testing.assert_array_equal(idx, np.arange(3))  # first half of 6 channels
    assert L.shape[0] == len(idx)
    assert L.shape[1] % 3 == 0
    assert fake_driver.attached_coils is not None
