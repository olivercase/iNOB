"""Every command's ``--help`` is documentation, so it is tested like code.

Help text rots silently: a flag gets renamed and the example underneath it goes
on quoting the old spelling, which is worse than no example because the reader
trusts it. These tests hold the three properties that make `inob <cmd> --help`
usable — it exists, it says how to invoke the command, and every flag it
demonstrates is a flag the command actually has.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import pathlib
import re

import pytest

from inob.cli.main import BY_NAME, COMMANDS

# Flags named in an example that belong to a *different* command. `inob status`
# points at `inob run --with-viz` because that is the command that draws the
# figures it reports on, and that cross-reference is the useful part.
CROSS_REFERENCED: dict[str, set[str]] = {
    "status": {"--with-viz"},
    "visualise": {"--with-viz"},
}


# argparse wraps to the terminal width, so a help-text width assertion is only
# meaningful against a fixed one. 80 is the narrowest terminal worth designing
# for, and the one every line here is written to fit.
WIDTH = 80


def _help_text(module: str) -> str:
    mod = importlib.import_module(module)
    buf = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
        mod.main(["--help"])
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _fixed_terminal_width(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", str(WIDTH))


@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: c.name)
def test_help_names_the_inob_spelling(cmd) -> None:
    """The usage line reads `inob <cmd>`, not the module or the legacy binary.

    Both entry points reach the same parser, and a user who typed `inob eeg`
    should not be told about `inob-eeg` in the reply.
    """
    assert _help_text(cmd.module).startswith(f"usage: inob {cmd.name}")


@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: c.name)
def test_help_carries_worked_examples(cmd) -> None:
    text = _help_text(cmd.module)
    assert "examples:" in text, f"{cmd.name} has no examples in its --help"
    body = text.split("examples:", 1)[1]
    assert f"inob {cmd.name}" in body, (
        f"{cmd.name}'s examples never invoke it"
    )


@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: c.name)
def test_every_demonstrated_flag_exists(cmd) -> None:
    """An example that quotes a flag the parser does not have is a lie."""
    text = _help_text(cmd.module)
    options, examples = text.split("examples:", 1)
    flags = re.compile(r"--[A-Za-z0-9][A-Za-z0-9-]*")
    real = set(flags.findall(options)) | CROSS_REFERENCED.get(cmd.name, set())
    used = set(flags.findall(examples))
    assert not (used - real), (
        f"inob {cmd.name} demonstrates flags it does not accept: "
        f"{sorted(used - real)}"
    )


# Everything below holds `--help` to the standard of published prose: it is the
# first thing a new user reads, and the only documentation many of them read.

#: reStructuredText that Sphinx renders and a terminal does not. A double
#: backtick or a `:mod:` role reaching stdout means a docstring written for the
#: maintainer has leaked into the text written for the user.
DEV_MARKUP = ("``", ":mod:", ":class:", ":func:", ":meth:", ":attr:")


@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: c.name)
def test_help_is_free_of_developer_markup(cmd) -> None:
    text = _help_text(cmd.module)
    found = [m for m in DEV_MARKUP if m in text]
    assert not found, (
        f"inob {cmd.name} --help prints unrendered Sphinx markup {found}. "
        f"Give the parser its own description= rather than a docstring."
    )


@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: c.name)
def test_help_does_not_lead_with_a_module_label(cmd) -> None:
    """`CLI: render the topoplots` is a note to the author, not to the reader."""
    text = _help_text(cmd.module)
    assert not re.search(r"^CLI:", text, re.M), (
        f"inob {cmd.name} --help opens with a 'CLI:' module label"
    )


@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: c.name)
def test_help_never_advertises_the_legacy_binary(cmd) -> None:
    """The `inob-*` scripts still work, but nothing should teach them."""
    text = _help_text(cmd.module)
    legacy = re.findall(r"\binob-[a-z][a-z-]*", text)
    assert not legacy, (
        f"inob {cmd.name} --help points at the legacy binaries {sorted(set(legacy))}"
    )


@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: c.name)
def test_help_fits_an_eighty_column_terminal(cmd) -> None:
    over = [ln for ln in _help_text(cmd.module).splitlines() if len(ln) > WIDTH - 1]
    assert not over, (
        f"inob {cmd.name} --help wraps badly at {WIDTH} columns:\n"
        + "\n".join(f"  {len(ln)}: {ln}" for ln in over)
    )


@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: c.name)
def test_example_descriptions_share_one_column(cmd) -> None:
    """A ragged example block reads as three lists, not one.

    Each example is `  inob <cmd> ...` then whitespace then a description. The
    descriptions have to start in the same column, or the eye cannot scan down
    them — which is the entire reason the block is laid out as a table.
    """
    body = _help_text(cmd.module).split("examples:", 1)[1]
    columns = {
        m.end(1) for ln in body.splitlines()
        if (m := re.match(r"^(  inob \S.*?\s{2,})\S", ln))
    }
    assert len(columns) <= 1, (
        f"inob {cmd.name} --help starts its example descriptions in columns "
        f"{sorted(columns)}; they should all share one."
    )


def test_readme_command_table_lists_every_command() -> None:
    """The README's table is the same list as `inob --help`, so it must not drift.

    It went stale once already — six commands shipped without ever reaching the
    table, so the only place most readers look said the toolkit was smaller
    than it is.
    """
    readme = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text()
    table = readme.split("| Group | Commands |", 1)[1].split("\n\n", 1)[0]
    listed = {name for name in re.findall(r"`([a-z][a-z-]*)`", table)}
    missing = {c.name for c in COMMANDS} - listed
    assert not missing, (
        f"README command table omits {sorted(missing)}; `inob --help` has them"
    )
    unknown = listed - set(BY_NAME)
    assert not unknown, f"README command table invents {sorted(unknown)}"


@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: c.name)
def test_readme_files_each_command_under_its_own_group(cmd) -> None:
    """A command listed under the wrong heading is worse than one left out."""
    readme = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text()
    table = readme.split("| Group | Commands |", 1)[1].split("\n\n", 1)[0]
    row = next(
        (r for r in table.splitlines() if f"`{cmd.name}`" in r), None
    )
    assert row is not None, f"{cmd.name} is not in the README table"
    assert row.split("|")[1].strip() == cmd.group, (
        f"README files {cmd.name} under {row.split('|')[1].strip()!r}, "
        f"but `inob --help` groups it under {cmd.group!r}"
    )
