"""Rendering must work off the main thread.

The browser GUI solves on a worker thread and renders figures there. macOS'
default matplotlib backend raises "Cannot create a GUI FigureManager outside
the main thread" in that situation, so every GUI run died at the visualisation
stage while the identical CLI run (main thread) passed. Caught only by actually
driving the GUI end to end, hence this test.
"""

from __future__ import annotations

import threading

import matplotlib as mpl
import pytest

from inob.viz.style import apply_nature_style, ensure_headless_backend


def _in_thread(fn):
    """Run ``fn`` on a worker thread, re-raising whatever it raised."""
    box: dict[str, BaseException | None] = {"exc": None}

    def _target() -> None:
        try:
            fn()
        except BaseException as e:
            box["exc"] = e

    t = threading.Thread(target=_target)
    t.start()
    t.join(timeout=60)
    assert not t.is_alive(), "worker thread hung"
    if box["exc"] is not None:
        raise box["exc"]


def test_ensure_headless_backend_leaves_the_main_thread_alone() -> None:
    before = mpl.get_backend()
    ensure_headless_backend()
    assert mpl.get_backend() == before


def test_figure_can_be_built_and_saved_off_the_main_thread(tmp_path) -> None:
    out = tmp_path / "offthread.png"

    def _render() -> None:
        apply_nature_style()  # this is what switches the backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        fig.savefig(out)
        plt.close(fig)

    _in_thread(_render)
    assert out.is_file() and out.stat().st_size > 0


def test_interactive_backend_is_swapped_for_agg_on_a_worker_thread(
    monkeypatch,
) -> None:
    """Simulate the macOS default without needing a Mac to run the test."""
    monkeypatch.setattr(mpl, "get_backend", lambda: "MacOSX")
    used: list[str] = []
    monkeypatch.setattr(mpl, "use", lambda name, force=False: used.append(name))

    _in_thread(ensure_headless_backend)
    assert used == ["Agg"], f"expected a switch to Agg, got {used}"


@pytest.mark.parametrize("backend", ["agg", "Agg", "pdf", "svg"])
def test_non_interactive_backends_are_left_untouched(monkeypatch, backend) -> None:
    monkeypatch.setattr(mpl, "get_backend", lambda: backend)
    used: list[str] = []
    monkeypatch.setattr(mpl, "use", lambda name, force=False: used.append(name))

    _in_thread(ensure_headless_backend)
    assert used == []
