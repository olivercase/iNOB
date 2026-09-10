"""Volume-field read-out: point sampling, DUNEuro call shapes, NPZ round-trip.

duneuropy is not importable on a dev Mac, so the driver is faked. What these
tests pin is the part that is easy to get silently wrong and expensive to
discover on the cluster: the layout DUNEuro returns
(``output(row, dim*i + j)`` — position-major) and the reciprocity arithmetic
that turns transfer-matrix rows into a stimulation montage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inob.config import load_config
from inob.forward import volume_field as vf
from inob.io.hdf5 import FemMesh

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"


def _fem() -> FemMesh:
    """Two tets, one per tissue, with easily-checked centroids."""
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [30.0, 30.0, 30.0],
            [33.0, 30.0, 30.0],
            [30.0, 33.0, 30.0],
            [30.0, 30.0, 33.0],
        ]
    )
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    tissue = np.array([1, 2], dtype=np.int32)
    return FemMesh(nodes, tets, tissue, ("vagus_left", "muscle"), "mm")


class _FakeDriver:
    """Records what it was asked, returns a matrix in DUNEuro's layout."""

    def __init__(self, n_dofs: int = 5, n_electrodes: int = 3) -> None:
        self.n_dofs = n_dofs
        self.transfer = np.arange(n_electrodes * n_dofs, dtype=float).reshape(n_electrodes, n_dofs)
        self.calls: list[dict] = []

    def computeEEGTransferMatrix(self, cfg):  # DUNEuro's spelling
        return self.transfer, {}

    def evaluateMultipleFunctionsAtPositions(self, rows, positions, cfg):
        self.calls.append({"rows": np.asarray(rows), "positions": positions, "cfg": cfg})
        n_rows = len(rows)
        n_pos = len(positions)
        stride = 1 if cfg["evaluation_return_type"] == "direct" else 3
        out = np.arange(n_rows * n_pos * stride, dtype=float).reshape(n_rows, n_pos * stride)
        return out, {}


# ── sampling ──────────────────────────────────────────────────────────────────


def test_sample_points_uses_centroids_so_every_point_is_inside() -> None:
    pos, tid = vf.sample_points(_fem())
    assert pos.shape == (2, 3)
    # Centroid of the first tet: mean of its four nodes.
    np.testing.assert_allclose(pos[0], [0.75, 0.75, 0.75])
    np.testing.assert_array_equal(tid, [1, 2])


def test_sample_points_can_restrict_to_a_tissue() -> None:
    pos, tid = vf.sample_points(_fem(), tissues=("muscle",))
    assert len(pos) == 1
    np.testing.assert_array_equal(tid, [2])


def test_sample_points_thins_by_spacing() -> None:
    fem = _fem()
    # A spacing that swallows the whole mesh keeps one point per occupied cell.
    pos, _ = vf.sample_points(fem, spacing_mm=1000.0)
    assert len(pos) == 1
    # A fine spacing keeps both, since the two tets are 30 mm apart.
    pos, _ = vf.sample_points(fem, spacing_mm=1.0)
    assert len(pos) == 2


# ── evaluation ────────────────────────────────────────────────────────────────


def test_evaluate_unpacks_duneuros_position_major_layout() -> None:
    driver = _FakeDriver()
    positions = np.zeros((4, 3))
    out = vf.evaluate_dof_rows(driver, np.ones((2, 5)), positions, evaluation_type="current")
    # (n_functions, n_positions, 3), with each position's three components
    # consecutive — output(row, dim*i + j).
    assert out.shape == (2, 4, 3)
    np.testing.assert_allclose(out[0, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(out[0, 1], [3.0, 4.0, 5.0])


def test_evaluate_direct_returns_one_value_per_point() -> None:
    driver = _FakeDriver()
    out = vf.evaluate_dof_rows(driver, np.ones((1, 5)), np.zeros((4, 3)), evaluation_type="direct")
    assert out.shape == (1, 4)


def test_evaluate_rejects_an_unknown_return_type() -> None:
    with pytest.raises(ValueError, match="evaluation_type"):
        vf.evaluate_dof_rows(
            _FakeDriver(), np.ones((1, 5)), np.zeros((2, 3)), evaluation_type="everything"
        )


# ── stimulation (reciprocity used forwards) ───────────────────────────────────


def _patch_driver(monkeypatch, driver: _FakeDriver) -> None:
    monkeypatch.setattr(vf, "import_duneuro", lambda cfg: object())
    monkeypatch.setattr(vf, "build_driver", lambda cfg, fem: (driver, {}, None))
    monkeypatch.setattr("inob.forward.eeg.attach_electrodes", lambda *a, **k: None)


def test_stimulation_field_is_the_difference_of_two_transfer_rows(
    monkeypatch,
) -> None:
    driver = _FakeDriver()
    _patch_driver(monkeypatch, driver)
    cfg = load_config(DEFAULT_CFG)

    field = vf.stimulation_field(
        cfg,
        _fem(),
        np.zeros((3, 3)),
        anode=2,
        cathode=0,
        current_mA=2.0,
        spacing_mm=0.0,
    )
    # I * (T[anode] - T[cathode]) — the DOF vector of the montage.
    expected = 2.0 * (driver.transfer[2] - driver.transfer[0])
    np.testing.assert_allclose(driver.calls[0]["rows"][0], expected)
    assert driver.calls[0]["cfg"] == {"evaluation_return_type": "current"}
    assert field.values.shape == (2, 3)
    assert field.tissue_ids is not None
    assert "anode 2" in field.description


def test_stimulation_field_rejects_a_degenerate_montage(monkeypatch) -> None:
    _patch_driver(monkeypatch, _FakeDriver())
    cfg = load_config(DEFAULT_CFG)
    with pytest.raises(ValueError, match="different electrodes"):
        vf.stimulation_field(cfg, _fem(), np.zeros((3, 3)), anode=1, cathode=1)


def test_stimulation_field_rejects_an_out_of_range_electrode(monkeypatch) -> None:
    _patch_driver(monkeypatch, _FakeDriver(n_electrodes=3))
    cfg = load_config(DEFAULT_CFG)
    with pytest.raises(ValueError, match="out of range"):
        vf.stimulation_field(cfg, _fem(), np.zeros((3, 3)), anode=0, cathode=7)


# ── artefact ──────────────────────────────────────────────────────────────────


def test_volume_field_npz_round_trip(tmp_path: Path) -> None:
    field = vf.VolumeField(
        positions_mm=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        values=np.array([[1.0, 0.0, 0.0], [0.0, 3.0, 4.0]]),
        evaluation_type="current",
        tissue_ids=np.array([1, 2]),
        description="test montage",
    )
    path = tmp_path / "field.npz"
    vf.save_volume_field(path, field)
    back = vf.load_volume_field(path)
    np.testing.assert_allclose(back.positions_mm, field.positions_mm)
    np.testing.assert_allclose(back.values, field.values)
    assert back.evaluation_type == "current"
    assert back.description == "test montage"
    np.testing.assert_array_equal(back.tissue_ids, [1, 2])
    # magnitude collapses a vector field to one number per point for colouring.
    np.testing.assert_allclose(back.magnitude, [1.0, 5.0])


def test_missing_volume_field_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="volume field not found"):
        vf.load_volume_field(tmp_path / "nope.npz")
