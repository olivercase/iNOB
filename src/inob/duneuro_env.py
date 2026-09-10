"""Finding the Python that can import ``duneuropy``.

DUNEuro has no PyPI wheel. It is compiled locally against one specific Python,
and the resulting ``duneuropy.so`` carries no ABI tag, so it can only be
imported by the interpreter it was built for. In practice that means the
interpreter you type ``inob`` with and the interpreter that can run a forward
solve are usually *different* — a 3.11 venv next to the build, versus whatever
``python3`` resolves to on the PATH.

That mismatch used to surface as "DUNEuro not importable", which reads as "you
have not built it" when the truth is "you built it, and it is ten metres away".
This module closes that gap: it finds interpreters that can actually import the
extension, verifies them by asking them, and remembers the answer.

Two things use it:

* :mod:`inob.cli.doctor` — to report *where* DUNEuro is rather than that it is
  absent here.
* :func:`reexec_with_duneuro` — so ``inob forward`` / ``inob eeg`` re-run
  themselves under a capable interpreter instead of failing.

Nothing here imports ``duneuropy`` into this process: every check runs in a
subprocess, because a mismatched extension does not raise ImportError, it
segfaults.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: Set on a re-executed child so it can never re-exec again.
REEXEC_GUARD = "INOB_DUNEURO_REEXEC"

#: Explicit override: the interpreter to use for DUNEuro stages.
PYTHON_ENV_VAR = "INOB_DUNEURO_PYTHON"

#: Extra root to search, for a build somewhere unusual.
ROOT_ENV_VAR = "INOB_DUNEURO_ROOT"

#: How long to give a candidate interpreter to import the extension. A cold
#: import off a network or external volume is slow the first time.
PROBE_TIMEOUT_S = 60.0


def duneuropy_importable() -> bool:
    """Can *this* interpreter import ``duneuropy``?"""
    try:
        return importlib.util.find_spec("duneuropy") is not None
    except (ImportError, ValueError):
        return False


def _cache_file() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "inob" / "duneuro-python"


def _read_cache() -> Path | None:
    try:
        text = _cache_file().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None


def _write_cache(python: Path) -> None:
    try:
        cache = _cache_file()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(str(python), encoding="utf-8")
    except OSError:
        pass  # a cache we cannot write is not worth failing over


def _search_roots() -> list[Path]:
    """Directories that plausibly contain a local DUNEuro build.

    Deliberately the same places :mod:`gui.backend.duneuro_setup` looks, so the
    CLI and the GUI agree about what exists on this machine.
    """
    roots: list[Path] = []
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        roots.append(Path(env))

    here = Path(__file__).resolve().parents[2]  # the project checkout
    home = Path.home()
    named = ("duneuro_build", "duneuro", "duneuro-py", "duneuro-src")

    for base in (
        here,
        here.parent,
        home,
        home / "Scratch" / "inob",
        Path("/opt"),
        Path("/usr/local"),
    ):
        roots.append(base)
        for name in named:
            roots.append(base / name)

    # The documented local build (environment.yml + scripts/build_duneuro_local.sh)
    # installs duneuropy into a conda env and builds the sources under
    # ~/.local/share/inob-duneuro. Neither sits under any root above once `inob`
    # is run from some *other* interpreter, which is exactly when this search
    # has to do the work.
    roots.append(home / ".local" / "share" / "inob-duneuro")
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        roots.append(Path(conda))
    for install in ("miniforge3", "miniconda3", "anaconda3", "mambaforge"):
        envs = home / install / "envs"
        try:
            roots.extend(p for p in envs.iterdir() if p.is_dir())
        except OSError:
            pass

    # An external volume is where a multi-gigabyte build usually ends up.
    volumes = Path("/Volumes")
    if volumes.is_dir():
        try:
            for volume in volumes.iterdir():
                if volume.is_dir():
                    roots.append(volume)
                    for name in named:
                        roots.append(volume / name)
        except OSError:
            pass

    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            continue
        if key not in seen and root.is_dir():
            seen.add(key)
            out.append(root)
    return out


#: Where a venv's interpreter sits relative to a search root. Narrow patterns
#: on purpose: an unbounded rglob over /Volumes or $HOME takes minutes.
_VENV_PATTERNS = (
    "bin/python3*",
    "venv/bin/python3*",
    ".venv/bin/python3*",
    "*/venv/bin/python3*",
    "*/.venv/bin/python3*",
    "*/*/venv/bin/python3*",
)


def candidate_interpreters() -> list[Path]:
    """Interpreters worth probing, best-first, without duplicates.

    Ordering matters: an explicit override is tried before a remembered answer,
    and both before anything discovered by searching.
    """
    out: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | str | None) -> None:
        if not path:
            return
        p = Path(path)
        try:
            key = str(p.resolve())
        except OSError:
            return
        # A symlinked python3 → python3.11 is one interpreter, not two.
        if key in seen or not p.exists() or p.is_dir():
            return
        seen.add(key)
        out.append(p)

    add(os.environ.get(PYTHON_ENV_VAR))
    add(_read_cache())
    for root in _search_roots():
        for pattern in _VENV_PATTERNS:
            try:
                for hit in sorted(root.glob(pattern)):
                    add(hit)
            except OSError:
                continue
    return out


def _can_import(python: Path, *, timeout: float = PROBE_TIMEOUT_S) -> bool:
    """Ask ``python`` whether it can import duneuropy.

    In a subprocess, because an extension built for a different Python does not
    fail cleanly — it can abort the process outright, which is not something a
    caller can catch.
    """
    if Path(python).resolve() == Path(sys.executable).resolve():
        return duneuropy_importable()
    try:
        proc = subprocess.run(
            [str(python), "-c", "import duneuropy"],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def find_duneuro_python(*, use_cache: bool = True) -> Path | None:
    """An interpreter that can import ``duneuropy``, or ``None``.

    Prefers the running interpreter when it already works, so nothing is
    searched or re-executed in the common case of a correctly-set-up venv.
    """
    if duneuropy_importable():
        return Path(sys.executable)

    cached = _read_cache() if use_cache else None
    if cached is not None and _can_import(cached):
        return cached

    for candidate in candidate_interpreters():
        if _can_import(candidate):
            _write_cache(candidate)
            return candidate
    return None


def reexec_with_duneuro(argv: list[str] | None = None) -> None:
    """Re-run this command under an interpreter that has DUNEuro, if needed.

    Returns immediately (and the caller carries on) when the current
    interpreter can already import ``duneuropy``, when re-execution is turned
    off, or when no capable interpreter exists — the last of those leaves the
    caller to fail with its own message, which is more specific than anything
    this function could say.

    When it does re-execute, it does not return: the child replaces this
    process, so its exit status and streams are the ones the user sees.
    """
    if duneuropy_importable():
        return
    if os.environ.get(REEXEC_GUARD):
        # We are already the child of a re-exec and duneuropy is still missing.
        # Trying again would loop forever.
        return
    if os.environ.get("INOB_NO_REEXEC"):
        return

    python = find_duneuro_python()
    if python is None:
        return

    argv = list(sys.argv if argv is None else argv)
    # Re-enter through the module rather than the resolved script path: the
    # target venv may not have console scripts installed even when it has the
    # package, and `-m` works either way.
    cmd = [str(python), "-m", "inob.cli.main", *_as_subcommand(argv)]

    env = dict(os.environ)
    env[REEXEC_GUARD] = "1"
    # Make the checkout importable even if the venv has no `inob` installed,
    # without shadowing an install that is already there.
    src = str(Path(__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{existing}{os.pathsep}{src}" if existing else src

    logger.info("DUNEuro is not importable here; re-running under %s", python)
    print(f"→ using the DUNEuro interpreter at {python}", file=sys.stderr)
    os.execve(cmd[0], cmd, env)  # deliberate process replacement


def _as_subcommand(argv: list[str]) -> list[str]:
    """Turn this process's argv into arguments for ``inob.cli.main``.

    ``inob forward …``, ``inob-forward …`` and ``python -m inob.cli.forward …``
    all have to come out as ``forward …`` so the child runs the same stage.
    """
    prog = Path(argv[0]).name if argv else ""
    rest = argv[1:]
    # `inob <cmd> …` — argv[0] is the umbrella entry point, so it is already
    # in subcommand form.
    if prog in ("inob", "main.py", "__main__.py"):
        return rest
    # `inob-forward …` / `inob-eeg …` — recover the subcommand from the name.
    if prog.startswith("inob-"):
        return [prog[len("inob-") :], *rest]
    # `python -m inob.cli.run_forward …` — argv[0] is that module's file.
    stem = Path(argv[0]).stem if argv else ""
    mapped = {"run_forward": "forward", "run_eeg": "eeg"}.get(stem, stem)
    return [mapped, *rest] if mapped else rest


def describe() -> dict[str, object]:
    """Summary for ``inob doctor`` — where DUNEuro is, and how to use it."""
    if duneuropy_importable():
        return {
            "importable_here": True,
            "python": sys.executable,
            "same_interpreter": True,
        }
    python = find_duneuro_python()
    return {
        "importable_here": False,
        "python": str(python) if python else None,
        "same_interpreter": False,
    }
