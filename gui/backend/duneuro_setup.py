"""Discover local DUNEuro (``duneuropy``) builds for the GUI's setup picker.

The forward solve needs the ``duneuropy`` extension importable in the running
interpreter. A build is a compiled extension (``duneuropy.so`` / ``.dylib``),
so it is tied to one Python version — a 3.11 build cannot load into 3.13. This
module locates candidate builds, and, crucially, reports whether each one is
*actually importable by this backend* rather than just present on disk, so the
UI can tell the user the truth instead of offering a dead end.
"""
from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from dataclasses import asdict, dataclass
from pathlib import Path

# Extension file names duneuro-py produces. The directory *containing* one of
# these is what gets prepended to sys.path (that's the shim `duneuro_path`).
_MODULE_NAMES = ("duneuropy.so", "duneuropy.dylib", "duneuropy.pyd")

# Where local builds tend to live. Ordered best-first. The env var wins so a
# non-standard build can be surfaced without code changes.
def _search_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("INOB_DUNEURO_ROOT")
    if env:
        roots.append(Path(env))
    roots += [
        Path("/Volumes/UCL/duneuro_build"),
        Path.home() / "duneuro_build",
        Path.home() / "Scratch" / "inob" / "duneuro",
    ]
    return roots


@dataclass
class Candidate:
    path: str            # directory to put on sys.path
    module: str          # the extension file found there
    label: str           # human label for the dropdown
    python_tag: str      # e.g. "python3.11", or "unknown"
    importable: bool     # actually loads in THIS interpreter?
    note: str = ""


def _python_tag_for(module_file: Path) -> str:
    """Best guess at the Python version a build targets.

    duneuro-py names its module plainly (no ABI tag), so we read it from the
    venv path (``.../python3.11/...``) when possible.
    """
    for part in module_file.parts:
        if part.startswith("python3."):
            return part
    return "unknown"


def _find_modules() -> list[Path]:
    """Locate ``duneuropy`` extension files under the known roots."""
    found: list[Path] = []
    seen: set[Path] = set()
    for root in _search_roots():
        if not root.is_dir():
            continue
        for name in _MODULE_NAMES:
            for hit in root.rglob(name):
                resolved = hit.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(hit)
    return found


def _running_tag() -> str:
    """This interpreter as a ``python3.X`` tag, to compare against a build."""
    return f"python3.{sys.version_info.minor}"


def _tags_conflict(build_tag: str, running_tag: str) -> bool:
    """True when a build is known to target a different Python than ours.

    ``unknown`` never conflicts: we cannot tell, so the build still gets
    probed. A known mismatch is worth trusting, because probing one is not a
    harmless experiment — see ``_importable_from``.
    """
    return build_tag.startswith("python3.") and build_tag != running_tag


def _importable_from(directory: Path, *, timeout: float = 20.0,
                     build_tag: str = "unknown") -> bool:
    """Can THIS interpreter import duneuropy with ``directory`` on sys.path?

    Runs in a subprocess of the current executable so a heavy or crashy
    extension can't take the backend down, and so sys.path/sys.modules stay
    clean.

    A build compiled for another Python is NOT probed at all. duneuro-py names
    its module plainly, with no ABI tag, so an incompatible interpreter will
    happily dlopen it and then segfault inside ``PyInit_duneuropy`` — pybind11
    calls into a CPython ABI that isn't there. That crash is contained by the
    subprocess, but it still writes a macOS crash report every time the setup
    panel is opened, so the version check comes first.
    """
    if _tags_conflict(build_tag, _running_tag()):
        return False

    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import duneuropy"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, str(directory)],
            capture_output=True, timeout=timeout,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def running_interpreter() -> dict[str, object]:
    """Describe the interpreter the backend runs under."""
    try:
        import duneuropy  # noqa: F401
        importable = True
        where = getattr(sys.modules["duneuropy"], "__file__", None)
    except Exception:
        importable = False
        where = None
    return {
        "executable": sys.executable,
        "python": sysconfig.get_python_version()
        + f" ({sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro})",
        "duneuropy_importable": importable,
        "duneuropy_location": where,
    }


def discover(active_path: str | None) -> dict[str, object]:
    """Full picture for the setup panel: interpreter + candidate builds.

    ``active_path`` is the current ``forward.duneuro_path`` from the config, so
    a user-set custom path is always represented even if it's outside the known
    roots.
    """
    interp = running_interpreter()
    candidates: list[Candidate] = []
    seen_dirs: set[str] = set()

    for module_file in _find_modules():
        directory = module_file.parent
        key = str(directory.resolve())
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        tag = _python_tag_for(module_file)
        importable = _importable_from(directory, build_tag=tag)
        note = ""
        if not importable:
            note = (
                f"Built for {tag}, but this backend runs Python "
                f"{sys.version_info.major}.{sys.version_info.minor}. "
                "Restart the backend with the matching interpreter."
                if tag != "unknown"
                else "This backend can't import this build."
            )
        candidates.append(Candidate(
            path=str(directory),
            module=module_file.name,
            label=f"{directory.name}  ·  {tag}",
            python_tag=tag,
            importable=importable,
            note=note,
        ))

    # Surface an already-set custom path that wasn't auto-discovered.
    if active_path and active_path not in seen_dirs and \
            Path(active_path).resolve().as_posix() not in seen_dirs:
        directory = Path(active_path)
        candidates.append(Candidate(
            path=active_path,
            module="(from config)",
            label=f"{directory.name}  ·  custom",
            python_tag=_python_tag_for(directory / "x"),
            importable=(
                _importable_from(directory,
                                 build_tag=_python_tag_for(directory / "x"))
                if directory.is_dir() else False
            ),
            note="" if directory.is_dir() else "Path does not exist.",
        ))

    # Importable builds first, then by label.
    candidates.sort(key=lambda c: (not c.importable, c.label))

    hint = _overall_hint(interp, candidates)
    return {
        "running": interp,
        "candidates": [asdict(c) for c in candidates],
        "active_path": active_path,
        "hint": hint,
    }


def _overall_hint(interp: dict[str, object],
                  candidates: list[Candidate]) -> str:
    if interp["duneuropy_importable"]:
        return "DUNEuro is ready — the forward solve will run on this machine."
    importable = [c for c in candidates if c.importable]
    if importable:
        return (
            "Select a build below to point the solver at it, then save. "
            "No backend restart needed."
        )
    if candidates:
        # A build exists but the interpreter can't load it (the common case).
        best = candidates[0]
        return (
            f"A DUNEuro build was found at {best.path} but it targets "
            f"{best.python_tag}, which this backend can't load. Restart the "
            "backend with that interpreter to run the forward solve."
        )
    return (
        "No local DUNEuro build was found. Build it with "
        "duneuro-build, or run the forward solve on the cluster."
    )
