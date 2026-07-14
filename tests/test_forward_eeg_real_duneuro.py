"""Real-DUNEuro end-to-end test for the EEG forward solve.

Companion to tests/test_forward_eeg.py's fake-driver tests: this one runs
the actual compiled duneuropy extension (skips automatically when it isn't
importable, same convention as tests/test_forward_smoke.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import inob.forward.eeg as eeg_mod
from inob.config import load_config
from inob.io.hdf5 import FemMesh, SensorArray, save_fem, save_sensors
from inob.io.npz import load_leadfield

pytest.importorskip("duneuropy", reason="duneuropy not built locally")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"


def _fem_bipyramid() -> FemMesh:
    """A tiny 2-tet convex 'vagus_left' FEM (nodes in mm): a bipyramid over a
    base triangle. A single tet would make vagus_sources' z-binning degenerate
    (z_lo == z_hi collapses np.arange to one edge, i.e. zero bins, i.e. zero
    sources) — two tets with distinct centroid Z give it a real bin to fill,
    and the bipyramid stays convex so the averaged centroid lands inside the
    mesh, same guarantee test_forward_smoke.py relies on for its single tet."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([50.0, 0.0, 0.0])
    c = np.array([0.0, 50.0, 0.0])
    top = np.array([50 / 3, 50 / 3, 10.0])
    bot = np.array([50 / 3, 50 / 3, -10.0])
    nodes = np.array([a, b, c, top, bot], dtype=np.float64)
    tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int32)
    tissue = np.array([1, 1], dtype=np.int32)
    return FemMesh(nodes=nodes, tets=tets, tissue=tissue,
                    tissue_labels=("vagus_left",), unit="mm")


def _electrodes(n: int) -> SensorArray:
    # Near-boundary points on the bipyramid's 6 triangular faces; setElectrodes
    # snaps each to the closest face-centre via "closest_subentity_center".
    pos = np.array([
        [20.0, 5.0, 3.0],     # near face a-b-top
        [5.0, 20.0, 3.0],     # near face a-c-top
        [20.0, 5.0, -3.0],    # near face a-b-bot
        [5.0, 20.0, -3.0],    # near face a-c-bot
        [16.0, 16.0, 0.0],    # near the shared base triangle a-b-c
    ])[:n]
    ori = np.tile([0.0, 0.0, 1.0], (n, 1))
    labels = tuple(f"elec-{i:04d}" for i in range(n))
    return SensorArray(coilpos=pos, coilori=ori, labels=labels,
                        chantype=tuple(["eeg"] * n), chanunit=tuple(["V"] * n),
                        unit="mm")


@pytest.mark.duneuro
def test_run_eeg_forward_end_to_end_with_real_duneuro(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CFG, project_root=tmp_path)
    save_fem(cfg.outputs.fem_mat, _fem_bipyramid())
    n_elec = 5
    electrodes = _electrodes(n_elec)
    save_sensors(cfg.outputs.electrodes_mat, electrodes)

    out = eeg_mod.run_eeg_forward(cfg)
    assert out == cfg.outputs.forward_eeg_npz
    assert out.exists()

    lf = load_leadfield(out)
    n_dipoles = 3  # 1 source slab x 3 moments
    assert lf.L.shape == (n_elec, n_dipoles)
    assert lf.channel_names == electrodes.labels
    assert np.isfinite(lf.L).all()
    # common-average reference => each column sums to ~0 across electrodes
    np.testing.assert_allclose(lf.L.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(
        lf.L_fT_per_nAm, lf.L * eeg_mod.EEG_CALIBRATION_FACTOR, atol=1e-12,
    )
