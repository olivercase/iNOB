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
import re

import pytest

from inob.cli.main import COMMANDS

# Flags named in an example that belong to a *different* command. `inob status`
# points at `inob run --with-viz` because that is the command that draws the
# figures it reports on, and that cross-reference is the useful part.
CROSS_REFERENCED: dict[str, set[str]] = {
    "status": {"--with-viz"},
    "visualise": {"--with-viz"},
}


def _help_text(module: str) -> str:
    mod = importlib.import_module(module)
    buf = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
        mod.main(["--help"])
    return buf.getvalue()


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
