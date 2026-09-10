"""Logging-setup smoke tests."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from inob.logging_setup import configure_logging, run_log_path


@pytest.fixture(autouse=True)
def _clean_root_logger():
    """Each test starts from a bare root logger.

    configure_logging() installs handlers on the process-wide root, so without
    this one test's handlers decide what the next one observes.
    """
    import logging

    root = logging.getLogger()
    saved, saved_level = list(root.handlers), root.level
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved:
        root.addHandler(h)
    root.setLevel(saved_level)


def test_configure_logging_idempotent(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    configure_logging("DEBUG", log_path)
    n1 = sum(1 for h in logging.getLogger().handlers if getattr(h, "_inob", False))
    configure_logging("INFO", log_path)
    n2 = sum(1 for h in logging.getLogger().handlers if getattr(h, "_inob", False))
    assert n1 == n2 == 2  # stream + file


def test_log_writes_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    configure_logging("INFO", log_path)
    logging.getLogger("inob.test").info("hello world")
    for h in logging.getLogger().handlers:
        h.flush()
    contents = log_path.read_text()
    assert "hello world" in contents
    assert "inob.test" in contents


def test_run_log_path_format(tmp_path: Path) -> None:
    p = run_log_path(tmp_path)
    assert p.parent == tmp_path
    assert p.name.startswith("run-") and p.name.endswith(".log")


def test_early_warnings_reach_both_the_console_and_the_log(tmp_path: Path, capsys) -> None:
    """A warning logged before the log file is known still ends up in it.

    Loading the config is what tells us where the log goes, and loading the
    config is itself a source of warnings. Without the buffer those warnings
    were only ever seen on stderr, so the log was not a full record of the run.
    """
    import logging

    from inob.logging_setup import begin_capture

    begin_capture("INFO")
    logging.getLogger("inob.config").warning("band wider than the sensor")
    assert "band wider than the sensor" in capsys.readouterr().err

    log_path = tmp_path / "run.log"
    configure_logging("INFO", log_path)
    logging.getLogger("inob.config").warning("and one after")

    body = log_path.read_text(encoding="utf-8")
    assert "band wider than the sensor" in body
    assert "and one after" in body
    # Order is the run's order, not "buffered ones last".
    assert body.index("band wider") < body.index("and one after")


def test_begin_capture_leaves_a_configured_logger_alone(tmp_path: Path) -> None:
    import logging

    from inob.logging_setup import begin_capture

    log_path = tmp_path / "run.log"
    configure_logging("INFO", log_path)
    before = list(logging.getLogger().handlers)
    begin_capture("DEBUG")
    assert list(logging.getLogger().handlers) == before
