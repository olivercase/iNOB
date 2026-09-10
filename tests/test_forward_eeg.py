"""Tests for the EEG forward solve, with a fake DUNEuro driver
(no real duneuropy install required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import inob.forward.eeg as eeg_mod
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


class _FakeEegDriver:
    def __init__(self, n_elec: int, L_per_dipole: np.ndarray):
        self.n_elec = n_elec
        self.L_per_dipole = L_per_dipole  # (n_dipoles,) amplitude per dipole
        self.electrodes_attached = None

    def setElectrodes(self, elec_du, opts):
        self.electrodes_attached = (elec_du, opts)

    def computeEEGTransferMatrix(self, driver_cfg):
        return np.zeros((self.n_elec, 1)), None

    def applyEEGTransfer(self, T, dipoles, driver_cfg):
        fields = [np.full(self.n_elec, self.L_per_dipole[i]) for i in range(len(dipoles))]
        return fields, None


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


def _electrodes(n: int) -> SensorArray:
    pos = np.column_stack([np.arange(n, dtype=np.float64), np.zeros(n), np.zeros(n)])
    ori = np.tile([1.0, 0.0, 0.0], (n, 1))
    labels = tuple(f"elec-{i:04d}" for i in range(n))
    return SensorArray(
        coilpos=pos,
        coilori=ori,
        labels=labels,
        chantype=tuple(["eeg"] * n),
        chanunit=tuple(["V"] * n),
        unit="mm",
    )


def test_attach_electrodes_calls_set_electrodes() -> None:
    calls = []

    class FakeDriver:
        def setElectrodes(self, elec_du, opts):
            calls.append((elec_du, opts))

    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    eeg_mod.attach_electrodes(FakeDriver(), _FakeDp, pos)
    assert len(calls) == 1
    elec_du, opts = calls[0]
    assert len(elec_du) == 2
    assert opts["type"] == "closest_subentity_center"


def test_compute_eeg_leadfield_is_common_average_referenced() -> None:
    n_elec = 4
    src_pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    n_dipoles = 3 * len(src_pos)
    per_dipole = np.arange(1, n_dipoles + 1, dtype=np.float64)
    driver = _FakeEegDriver(n_elec=n_elec, L_per_dipole=per_dipole)

    L = eeg_mod.compute_eeg_leadfield(driver, _FakeDp, {"foo": "bar"}, src_pos)
    assert L.shape == (n_elec, n_dipoles)
    # common-average reference => each column sums to ~0 across electrodes
    np.testing.assert_allclose(L.mean(axis=0), 0.0, atol=1e-10)


def test_run_eeg_forward_end_to_end_with_fake_driver(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config(DEFAULT_CFG, project_root=tmp_path)
    fem = _fem_two_source_slabs()
    save_fem(cfg.outputs.fem_mat, fem)
    electrodes = _electrodes(5)
    save_sensors(cfg.outputs.electrodes_mat, electrodes)

    n_dipoles = 3  # 1 source slab x 3 moments
    fake_driver = _FakeEegDriver(n_elec=5, L_per_dipole=np.full(n_dipoles, 1e-8))

    monkeypatch.setattr(eeg_mod, "import_duneuro", lambda cfg: _FakeDp)
    monkeypatch.setattr(
        eeg_mod,
        "build_driver",
        lambda cfg, fem: (fake_driver, {"volume_conductor": {}}, np.array([3e-4, 4.3e-4])),
    )

    out = eeg_mod.run_eeg_forward(cfg)
    assert out == cfg.outputs.forward_eeg_npz
    assert out.exists()

    lf = load_leadfield(out)
    assert lf.L.shape == (5, n_dipoles)
    assert lf.channel_names == electrodes.labels
    # L_fT_per_nAm slot holds µV/(nA·m) = L * EEG_CALIBRATION_FACTOR
    np.testing.assert_allclose(
        lf.L_fT_per_nAm,
        lf.L * eeg_mod.EEG_CALIBRATION_FACTOR,
        atol=1e-12,
    )


def test_run_eeg_forward_missing_electrodes_raises(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CFG, project_root=tmp_path)
    fem = _fem_two_source_slabs()
    save_fem(cfg.outputs.fem_mat, fem)
    with pytest.raises(FileNotFoundError):
        eeg_mod.run_eeg_forward(cfg)
