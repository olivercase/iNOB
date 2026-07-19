"""CLI: terminal presentation helpers (colour gating, sizes, ages, paths)."""
from __future__ import annotations

from pathlib import Path

import inob.cli._ui as ui


class _Stream:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_paint_plain_when_no_color_set(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert ui.paint("hello", "red", stream=_Stream(True)) == "hello"


def test_paint_plain_when_not_a_tty(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    assert ui.paint("hello", "red", stream=_Stream(False)) == "hello"


def test_paint_colours_a_tty(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    painted = ui.paint("hello", "red", stream=_Stream(True))
    assert painted.startswith("\033[31m")
    assert painted.endswith("\033[0m")


def test_human_bytes_boundaries() -> None:
    assert ui.human_bytes(0) == "0 B"
    assert ui.human_bytes(512) == "512 B"
    assert ui.human_bytes(1024) == "1.0 KB"
    assert ui.human_bytes(1536) == "1.5 KB"
    assert ui.human_bytes(1024 * 1024) == "1.0 MB"


def test_human_age_scales() -> None:
    now = 1_000_000.0
    assert ui.human_age(now, now=now) == "just now"
    assert ui.human_age(now - 60, now=now) == "1 minute ago"
    assert ui.human_age(now - 180, now=now) == "3 minutes ago"
    assert ui.human_age(now - 7200, now=now) == "2 hours ago"
    assert ui.human_age(now - 3 * 86400, now=now) == "3 days ago"
    assert ui.human_age(now - 604800, now=now) == "1 week ago"
    assert ui.human_age(now - 3 * 604800, now=now) == "3 weeks ago"


def test_rel_inside_and_outside_root(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    assert ui.rel(root / "outputs" / "fem.mat", root) == str(
        Path("outputs") / "fem.mat")
    outside = tmp_path / "elsewhere" / "fem.mat"
    assert ui.rel(outside, root) == str(outside)
