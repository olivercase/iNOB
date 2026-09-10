"""Tests for the rung-selective ladder runner (run_ladder).

The analytic rungs (biot, sarvas) must run from geometry alone; the fem rung
must require the leadfield and raise cleanly when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

import inob.analysis.sarvas_compare as sc
from inob.io.hdf5 import FemMesh, SensorArray
from inob.io.npz import Leadfield


def _fem() -> FemMesh:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [40.0, 0.0, 10.0],
            [41.0, 0.0, 10.0],
            [40.0, 1.0, 10.0],
            [40.0, 0.0, 11.0],
        ],
        dtype=np.float64,
    )
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    tissue = np.array([1, 2], dtype=np.int32)  # bone, vagus_left
    return FemMesh(
        nodes=nodes, tets=tets, tissue=tissue, tissue_labels=("bone", "vagus_left"), unit="mm"
    )


def _sensors(n: int = 4) -> SensorArray:
    # n radial + n T1 + n T2 so the triaxial radial mask selects the first n.
    # Spread the coils in x so the projected Biot–Savart field is non-zero
    # (a coil on the x=0 symmetry plane reads exactly zero for a z-moment).
    coilpos = np.array([[10.0 + z, 60.0, float(z)] for z in range(n)] * 3)
    coilori = np.tile(np.array([0.0, 1.0, 0.0]), (3 * n, 1))
    labels = tuple(
        [f"S{i}-R" for i in range(n)]
        + [f"S{i}-T1" for i in range(n)]
        + [f"S{i}-T2" for i in range(n)]
    )
    c = 3 * n
    return SensorArray(
        coilpos=coilpos,
        coilori=coilori,
        labels=labels,
        chantype=tuple("megmag" for _ in range(c)),
        chanunit=tuple("T" for _ in range(c)),
    )


_SRC = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 10.0],
    ],
    dtype=np.float64,
)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(sc, "load_fem", lambda p: _fem())
    monkeypatch.setattr(sc, "load_sensors", lambda p: _sensors())
    monkeypatch.setattr(sc, "vagus_sources", lambda *a, **k: _SRC)


class _Cfg:
    class outputs:
        fem_mat = "fem.mat"
        sensors_mat = "sensors.mat"
        forward_npz = "lf.npz"

    class forward:
        source_tissue = "vagus_left"
        source_spacing_mm = 5.0

    class analytic:
        source_axis_mm = 40.0
        sensor_axis_mm = 58.5
        band_tolerance_mm = 30.0
        silent_rel_threshold = 0.01
        bootstrap_n = 10
        bootstrap_seed = 0


def test_analytic_rungs_need_no_leadfield(patched):
    r = sc.run_ladder(_Cfg, rungs=["biot", "sarvas"])
    assert set(r["rungs"]) == {"biot", "sarvas"}
    assert r["rungs"]["biot"]["peak_fT_per_nAm"] > 0
    assert r["rungs"]["sarvas"]["peak_fT_per_nAm"] > 0
    assert "sarvas_to_biot" in r["ratios"]
    assert r["n_sources"] == 3


def test_source_idx_default_is_middle(patched):
    r = sc.run_ladder(_Cfg, rungs=["biot"])
    assert r["source_index"] == 1  # middle of 3


def test_fem_rung_requires_leadfield(patched, monkeypatch):
    def _raise(path):
        raise FileNotFoundError(f"leadfield not found: {path}")

    monkeypatch.setattr(sc, "load_leadfield", _raise)
    with pytest.raises(FileNotFoundError):
        sc.run_ladder(_Cfg, rungs=["fem"])


def test_full_ladder_with_leadfield(patched, monkeypatch):
    n_coils = 4  # matches the radial mask count
    S = 3
    L = np.ones((3 * n_coils, 3 * S)) * 1e-12
    lf = Leadfield(
        L=L,
        L_fT_per_nAm=L * 1e15,
        source_pos=_SRC,
        coil_pos=np.zeros((3 * n_coils, 3)),
        coil_orient=np.zeros((3 * n_coils, 3)),
        channel_names=tuple(f"c{i}" for i in range(3 * n_coils)),
        conductivities=np.array([0.3]),
        tissue_labels=("vagus_left",),
    )
    monkeypatch.setattr(sc, "load_leadfield", lambda p: lf)
    r = sc.run_ladder(_Cfg, rungs=["biot", "sarvas", "fem"])
    assert set(r["rungs"]) == {"biot", "sarvas", "fem"}
    assert "fem_to_sarvas" in r["ratios"]
    assert "fem_to_biot" in r["ratios"]


def test_invalid_rung_rejected(patched):
    with pytest.raises(ValueError):
        sc.run_ladder(_Cfg, rungs=["nonsense"])
