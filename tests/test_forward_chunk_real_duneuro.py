"""Real-DUNEuro end-to-end test for the cluster array-job chunk worker.

Companion to tests/test_forward_chunk.py's fake-driver tests: this one runs
the actual compiled duneuropy extension (skips automatically when it isn't
importable, same convention as tests/test_forward_smoke.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inob.config import load_config
from inob.forward.chunk import run_chunk
from inob.io.hdf5 import FemMesh, SensorArray, save_fem, save_sensors

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
    return FemMesh(nodes=nodes, tets=tets, tissue=tissue, tissue_labels=("vagus_left",), unit="mm")


def _sensors(n: int) -> SensorArray:
    pos = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
            [-100.0, 0.0, 0.0],
        ]
    )[:n]
    ori = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    labels = tuple(f"mag-{i:04d}-R" for i in range(n))
    return SensorArray(
        coilpos=pos,
        coilori=ori,
        labels=labels,
        chantype=tuple(["megmag"] * n),
        chanunit=tuple(["T"] * n),
        unit="mm",
    )


@pytest.mark.duneuro
def test_run_chunk_saves_slice_with_real_duneuro(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CFG, project_root=tmp_path)
    save_fem(cfg.outputs.fem_mat, _fem_bipyramid())
    n_chan = 4
    sensors = _sensors(n_chan)
    save_sensors(cfg.outputs.sensors_mat, sensors)

    n_chunks = 2
    out_L, out_idx = run_chunk(cfg, chunk_id=0, n_chunks=n_chunks)
    assert out_L.is_file() and out_idx.is_file()
    L = np.load(out_L)
    idx = np.load(out_idx)
    np.testing.assert_array_equal(idx, np.arange(2))  # first half of 4 channels
    assert L.shape[0] == len(idx)
    assert L.shape[1] % 3 == 0
    assert np.isfinite(L).all()
