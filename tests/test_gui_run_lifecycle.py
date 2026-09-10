"""Run-lifecycle guarantees for the GUI backend's /api/run WebSocket.

Two properties matter here and neither was covered before:

  1. The single-flight lock must be held for the *pipeline's* lifetime, not the
     socket's. A browser disconnect does not stop the run, so releasing on
     disconnect allowed a second run to start writing the same outputs/
     artefacts as the first.
  2. Cancellation must be cooperative and honest — it lands at a stage
     boundary and reports the stages that did not run as "cancelled".
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path

import pytest

from inob.cli.pipeline import ALL_STAGES, run_pipeline

# See test_cluster_submit: this import happens at collection time, so without
# the guard an image that has no GUI extra fails the run outright.
pytest.importorskip(
    "gui.backend",
    reason="the GUI backend is not installed in this environment",
)

app_mod = importlib.import_module("gui.backend.app")


# ── cooperative cancellation in the orchestrator ────────────────────────────


def test_run_pipeline_stops_at_the_next_stage_boundary(monkeypatch) -> None:
    ran: list[str] = []
    cancel = threading.Event()

    class _FakeStage:
        description = "fake"

        def __init__(self, name: str) -> None:
            self.name = name

        def run(self, cfg) -> None:
            ran.append(self.name)
            if self.name == "fem":
                cancel.set()  # ask to stop *during* the second stage

    monkeypatch.setattr("inob.cli.pipeline.STAGES", {n: _FakeStage(n) for n in ALL_STAGES})
    monkeypatch.setattr("inob.cli.pipeline._all_outputs_exist", lambda cfg, s: False)
    monkeypatch.setattr(
        "inob.cli.pipeline._failed_marker", lambda cfg, s: Path("/nonexistent/.FAILED")
    )

    statuses = run_pipeline(object(), stages=list(ALL_STAGES), should_cancel=cancel.is_set)

    # The in-flight stage completes (we cannot interrupt a DUNEuro solve), the
    # next one is marked cancelled, and nothing after it runs.
    assert ran == ["geom", "fem"]
    assert statuses["geom"] == "ran" and statuses["fem"] == "ran"
    assert statuses["sensors"] == "cancelled"
    assert "forward" not in statuses and "viz" not in statuses


def test_run_pipeline_without_cancel_hook_is_unchanged(monkeypatch) -> None:
    """The hook is optional — the CLI passes nothing and must be unaffected."""
    ran: list[str] = []

    class _FakeStage:
        description = "fake"

        def __init__(self, name: str) -> None:
            self.name = name

        def run(self, cfg) -> None:
            ran.append(self.name)

    monkeypatch.setattr("inob.cli.pipeline.STAGES", {n: _FakeStage(n) for n in ALL_STAGES})
    monkeypatch.setattr("inob.cli.pipeline._all_outputs_exist", lambda cfg, s: False)
    monkeypatch.setattr(
        "inob.cli.pipeline._failed_marker", lambda cfg, s: Path("/nonexistent/.FAILED")
    )

    statuses = run_pipeline(object(), stages=list(ALL_STAGES))
    assert ran == list(ALL_STAGES)
    assert set(statuses.values()) == {"ran"}


# ── single-flight lock ──────────────────────────────────────────────────────


def test_run_lock_is_not_held_at_import_time() -> None:
    assert app_mod._RUN_LOCK.acquire(blocking=False)
    app_mod._RUN_LOCK.release()


def test_lock_survives_client_disconnect_until_the_pipeline_finishes(
    monkeypatch,
) -> None:
    """Regression: the lock used to be released in the socket handler's
    ``finally``, which fires the moment the browser disconnects — while the
    pipeline thread is still writing outputs/. A second run could then start
    concurrently and race it on the same artefacts.

    Here the client hangs up mid-run; the lock must stay held until the
    (still-running) pipeline actually completes.
    """
    pytest.importorskip("httpx", reason="starlette TestClient needs httpx")
    from fastapi.testclient import TestClient

    entered = threading.Event()
    may_finish = threading.Event()

    # The pipeline runs as a child process now (VTK's renderer is main-thread
    # only, so it cannot share this one), which is what has to be stood in for:
    # a process whose output stream blocks until the test lets it finish.
    class _SlowChild:
        pid = -1

        def __init__(self) -> None:
            self.returncode: int | None = None

        @property
        def stdout(self):
            def lines():
                yield "[run]  geom: building"
                entered.set()
                may_finish.wait(timeout=10)
                yield "[ok]   geom"

            return lines()

        def wait(self, timeout=None) -> int:
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(app_mod.subprocess, "Popen", lambda *a, **k: _SlowChild())
    monkeypatch.setattr(
        app_mod,
        "_solver_choice",
        lambda: {"python": "python3", "duneuro_path": None, "found": True},
    )
    monkeypatch.setattr(app_mod, "load_config", lambda *a, **k: object())
    monkeypatch.setattr(
        app_mod,
        "compute_detectability",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no lf")),
    )

    client = TestClient(app_mod.app)
    with client.websocket_connect("/api/run") as ws:
        ws.send_json({"stages": "all"})
        assert entered.wait(timeout=10), "pipeline never started"
        # client goes away while the solve is still in flight

    # The pipeline is still running, so the lock must still be held.
    assert not app_mod._RUN_LOCK.acquire(blocking=False), (
        "the run lock was released while the pipeline was still running — a "
        "second run could now race it on the same output artefacts"
    )

    may_finish.set()
    for _ in range(100):  # ≤10 s for the worker to unwind
        if app_mod._RUN_LOCK.acquire(blocking=False):
            app_mod._RUN_LOCK.release()
            break
        threading.Event().wait(0.1)
    else:
        pytest.fail("lock was never released after the pipeline finished")
