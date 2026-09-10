"""Tests for the single-machine forward solve, with a fake DUNEuro driver
(no real duneuropy install required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import inob.forward.solve as solve_mod
from inob.config import load_config
from inob.io.hdf5 import FemMesh, SensorArray, save_fem, save_sensors
from inob.io.npz import load_leadfield

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
    def __init__(self, n_coils: int, rng: np.random.Generator):
        self.n_coils = n_coils
        self.rng = rng
        self.attached_coils = None

    def setCoilsAndProjections(self, coils, projs):
        self.attached_coils = (coils, projs)

    def computeMEGTransferMatrix(self, driver_cfg):
        return np.zeros((self.n_coils, 1)), None

    def applyMEGTransfer(self, T, dipoles, driver_cfg):
        # small deterministic amplitude, well under the fT/nAm bound
        fields = [self.rng.normal(scale=1e-8, size=self.n_coils) for _ in dipoles]
        return fields, None

    def computeMEGPrimaryField(self, dipoles, driver_cfg):
        # primary Biot–Savart term (real solve adds this to the transfer field)
        return [self.rng.normal(scale=1e-8, size=self.n_coils) for _ in dipoles]


def _fem_two_source_slabs() -> FemMesh:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 3.0],
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
            [10.0, 10.0, 11.0],
        ],
        dtype=np.float64,
    )
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=np.int32)
    tissue = np.array([1, 1, 2], dtype=np.int32)
    return FemMesh(
        nodes=nodes, tets=tets, tissue=tissue, tissue_labels=("vagus_left", "skin"), unit="mm"
    )


def _sensors(n: int) -> SensorArray:
    pos = np.column_stack([np.arange(n, dtype=np.float64), np.zeros(n), np.zeros(n)])
    ori = np.tile([1.0, 0.0, 0.0], (n, 1))
    labels = tuple(f"mag-{i:04d}-R" for i in range(n))
    return SensorArray(
        coilpos=pos,
        coilori=ori,
        labels=labels,
        chantype=tuple(["megmag"] * n),
        chanunit=tuple(["T"] * n),
        unit="mm",
    )


def test_run_forward_end_to_end_with_fake_driver(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config(DEFAULT_CFG, project_root=tmp_path)
    fem = _fem_two_source_slabs()
    save_fem(cfg.outputs.fem_mat, fem)
    sensors = _sensors(4)
    save_sensors(cfg.outputs.sensors_mat, sensors)

    rng = np.random.default_rng(0)
    fake_driver = _FakeDriver(n_coils=4, rng=rng)

    monkeypatch.setattr(solve_mod, "import_duneuro", lambda cfg: _FakeDp)
    monkeypatch.setattr(
        solve_mod,
        "build_driver",
        lambda cfg, fem: (fake_driver, {"volume_conductor": {}}, np.array([3e-4, 4.3e-4])),
    )

    out = solve_mod.run_forward(cfg)
    assert out == cfg.outputs.forward_npz
    assert out.exists()

    lf = load_leadfield(out)
    assert lf.L.shape[0] == 4
    assert lf.L.shape[1] % 3 == 0
    n_src = lf.L.shape[1] // 3
    assert n_src == 1  # both vagus_left tets fall in the same 5mm Z-slab
    np.testing.assert_allclose(lf.source_pos.shape, (n_src, 3))
    np.testing.assert_allclose(lf.L_fT_per_nAm, lf.L * 1e6)
    assert fake_driver.attached_coils is not None


def test_run_forward_missing_fem_raises(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CFG, project_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        solve_mod.run_forward(cfg)
