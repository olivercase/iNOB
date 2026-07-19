"""CLI: the unified `inob` dispatcher (registry, help, routing, error report)."""
from __future__ import annotations

import importlib

import pytest

import inob.cli.main as cli_mod
from inob import __version__


def test_no_args_prints_greeting(capsys) -> None:
    rc = cli_mod.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "inob doctor" in out
    assert "inob run" in out
    assert "inob detect" in out


def test_help_lists_every_group_and_command(capsys) -> None:
    rc = cli_mod.main(["--help"])
    assert rc == 0
    out = capsys.readouterr().out
    for group in cli_mod.GROUPS:
        assert group in out
    for cmd in cli_mod.COMMANDS:
        assert cmd.name in out


def test_version_prints_package_version(capsys) -> None:
    rc = cli_mod.main(["--version"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == f"inob {__version__}"


def test_unknown_command_returns_2_and_suggests(capsys) -> None:
    rc = cli_mod.main(["detct"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Unknown command" in err
    assert "inob detect" in err


def test_every_registered_module_exposes_callable_main() -> None:
    for cmd in cli_mod.COMMANDS:
        module = importlib.import_module(cmd.module)
        assert callable(module.main), f"{cmd.module} has no callable main()"


def test_command_names_and_aliases_are_unique() -> None:
    spellings = [c.name for c in cli_mod.COMMANDS]
    spellings += [a for c in cli_mod.COMMANDS for a in c.aliases]
    assert len(spellings) == len(set(spellings))
    assert len(cli_mod.BY_NAME) == len(spellings)


def test_dispatch_routes_remaining_argv_to_module(monkeypatch) -> None:
    import inob.cli.status as status_mod

    calls = {}
    monkeypatch.setattr(
        status_mod, "main", lambda argv: (calls.setdefault("argv", argv), 0)[1])
    rc = cli_mod.main(["status", "--json", "--config", "x.yaml"])
    assert rc == 0
    assert calls["argv"] == ["--json", "--config", "x.yaml"]


def test_debug_flag_is_stripped_before_dispatch(monkeypatch) -> None:
    import inob.cli.status as status_mod

    calls = {}
    monkeypatch.setattr(
        status_mod, "main", lambda argv: (calls.setdefault("argv", argv), 0)[1])
    rc = cli_mod.main(["status", "--debug", "--json"])
    assert rc == 0
    assert calls["argv"] == ["--json"]


def test_config_error_is_reported_not_raised(monkeypatch, capsys) -> None:
    import inob.cli.status as status_mod
    from inob.config import ConfigError

    def _boom(argv):
        raise ConfigError("forward.solver.type is not a string")

    monkeypatch.setattr(status_mod, "main", _boom)
    rc = cli_mod.main(["status"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Config error" in err
    assert "forward.solver.type" in err


def test_config_error_propagates_with_debug(monkeypatch) -> None:
    import inob.cli.status as status_mod
    from inob.config import ConfigError

    def _boom(argv):
        raise ConfigError("bad field")

    monkeypatch.setattr(status_mod, "main", _boom)
    with pytest.raises(ConfigError):
        cli_mod.main(["status", "--debug"])
