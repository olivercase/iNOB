"""Logging-setup smoke tests."""
from __future__ import annotations

import logging
from pathlib import Path

from vagus_fm.logging_setup import configure_logging, run_log_path


def test_configure_logging_idempotent(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    configure_logging("DEBUG", log_path)
    n1 = sum(1 for h in logging.getLogger().handlers if getattr(h, "_vagus_fm", False))
    configure_logging("INFO", log_path)
    n2 = sum(1 for h in logging.getLogger().handlers if getattr(h, "_vagus_fm", False))
    assert n1 == n2 == 2  # stream + file


def test_log_writes_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    configure_logging("INFO", log_path)
    logging.getLogger("vagus_fm.test").info("hello world")
    for h in logging.getLogger().handlers:
        h.flush()
    contents = log_path.read_text()
    assert "hello world" in contents
    assert "vagus_fm.test" in contents


def test_run_log_path_format(tmp_path: Path) -> None:
    p = run_log_path(tmp_path)
    assert p.parent == tmp_path
    assert p.name.startswith("run-") and p.name.endswith(".log")
