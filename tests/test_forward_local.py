"""Local multi-core forward orchestrator: worker resolution + serial fallback.

The parallel path itself spawns processes and needs duneuropy, so it is covered
by the real-DUNEuro tests / manual runs; here we test the dispatch logic that
decides serial-vs-parallel and how many workers to use.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import inob.forward.local as local_mod
from inob.forward.local import effective_workers


def _cfg(local_workers: int):
    return SimpleNamespace(forward=SimpleNamespace(local_workers=local_workers))


def test_effective_workers_zero_means_all_cores() -> None:
    assert effective_workers(_cfg(0)) == (os.cpu_count() or 1)


def test_effective_workers_explicit_count() -> None:
    assert effective_workers(_cfg(4)) == 4


def test_effective_workers_negative_falls_back_to_all() -> None:
    # A nonsensical negative is treated as "auto" rather than crashing.
    assert effective_workers(_cfg(-3)) == (os.cpu_count() or 1)


def test_run_forward_local_serial_when_one_worker(monkeypatch, tmp_path) -> None:
    """local_workers=1 must call the serial solve, not spawn a pool."""
    called = {}

    # 8 channels available, but 1 worker requested → serial.
    fake_sensors = SimpleNamespace(coilpos=[0] * 8)
    monkeypatch.setattr(local_mod, "load_sensors", lambda p: fake_sensors)

    import inob.forward.solve as solve_mod

    def _fake_serial(cfg):
        called["serial"] = True
        return "npz"

    monkeypatch.setattr(solve_mod, "run_forward", _fake_serial)

    cfg = SimpleNamespace(
        forward=SimpleNamespace(local_workers=1),
        outputs=SimpleNamespace(sensors_mat=tmp_path / "s.mat"),
    )
    out = local_mod.run_forward_local(cfg)
    assert out == "npz"
    assert called == {"serial": True}


def test_run_forward_local_serial_when_more_workers_than_channels(monkeypatch, tmp_path) -> None:
    """A single-channel array can't be split, so it falls back to serial."""
    called = {}
    monkeypatch.setattr(local_mod, "load_sensors", lambda p: SimpleNamespace(coilpos=[0]))
    import inob.forward.solve as solve_mod

    def _fake_serial(cfg):
        called["serial"] = True
        return "npz"

    monkeypatch.setattr(solve_mod, "run_forward", _fake_serial)
    cfg = SimpleNamespace(
        forward=SimpleNamespace(local_workers=0),  # all cores, but 1 channel
        outputs=SimpleNamespace(sensors_mat=tmp_path / "s.mat"),
    )
    assert local_mod.run_forward_local(cfg) == "npz"
    assert called == {"serial": True}
