#!/usr/bin/env python3
"""Every count the paper claims about the package, measured from the package.

The manuscript states a test count, three line counts, a subcommand count and
an array size. All of them drift as the code changes, and by submission four
of the six were wrong — the test count by 48%. Rather than re-counting by
hand at each revision, run this and paste what it prints.

    python3 scripts/paper_stats.py

Counts are deliberately crude and stated as such: "lines of Python" means
lines in ``.py`` files under ``src/``, blank and comment lines included,
because that is the only definition a reader can check without this script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _line_count(root: Path, *patterns: str) -> int:
    return sum(
        sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
        for pattern in patterns
        for path in root.rglob(pattern)
        if "node_modules" not in path.parts and ".venv" not in path.parts
    )


def _collected_tests() -> int | None:
    """How many tests pytest collects, or None if it cannot be run."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--collect-only"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  (could not collect tests: {exc})", file=sys.stderr)
        return None
    for line in reversed(proc.stdout.splitlines()):
        if "test" in line and "collected" in line:
            return int(line.split()[0])
    return None


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    from inob.cli.main import COMMANDS

    scripts = 0
    try:
        import tomllib

        with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
            # The `inob` umbrella entry point is not one of the legacy
            # per-stage binaries the paper counts.
            scripts = len(tomllib.load(fh)["project"].get("scripts", {})) - 1
    except (OSError, KeyError, ValueError) as exc:
        # A missing count is worth reporting, not worth failing the run.
        print(f"  (could not read console scripts: {exc})", file=sys.stderr)

    src = _line_count(REPO_ROOT / "src", "*.py")
    tests = _line_count(REPO_ROOT / "tests", "*.py")
    web = _line_count(REPO_ROOT / "gui" / "web", "*.ts", "*.tsx")
    collected = _collected_tests()

    print("Numbers the manuscript states about the package")
    print("=" * 47)
    print(f"  subcommands of `inob`        {len(COMMANDS)}")
    print(f"  standalone console scripts   {scripts}")
    print(f"  lines of Python (src/)       {src:,}")
    print(f"  lines of Python (+ tests/)   {src + tests:,}")
    print(f"  lines of TypeScript (gui)    {web:,}")
    if collected is not None:
        print(f"  tests collected              {collected}")
    print()
    print("Counts include blank and comment lines. Sensor-array size depends")
    print("on the config: `inob sensors` logs positions and channels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
