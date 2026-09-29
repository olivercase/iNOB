"""The script that measures the numbers the manuscript states.

It exists because four of the six counts in the submitted draft were wrong.
A broken counter is worse than no counter, so the arithmetic is pinned here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from paper_stats import _line_count  # noqa: E402


def test_line_count_counts_every_line_including_blanks(tmp_path) -> None:
    (tmp_path / "a.py").write_text("one\n\n# three\n")
    (tmp_path / "b.py").write_text("one\n")
    assert _line_count(tmp_path, "*.py") == 4


def test_line_count_recurses_and_skips_vendored_trees(tmp_path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x\n")
    vendored = tmp_path / "node_modules" / "dep"
    vendored.mkdir(parents=True)
    (vendored / "b.py").write_text("y\ny\ny\n")
    assert _line_count(tmp_path, "*.py") == 1


def test_line_count_takes_several_patterns(tmp_path) -> None:
    (tmp_path / "a.ts").write_text("x\n")
    (tmp_path / "b.tsx").write_text("y\ny\n")
    (tmp_path / "c.py").write_text("ignored\n")
    assert _line_count(tmp_path, "*.ts", "*.tsx") == 3


def test_line_count_is_zero_for_a_missing_tree(tmp_path) -> None:
    assert _line_count(tmp_path / "nope", "*.py") == 0


def test_script_runs_and_reports_each_number() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "paper_stats.py")],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    for label in (
        "subcommands of `inob`",
        "standalone console scripts",
        "lines of Python (src/)",
        "lines of TypeScript (gui)",
    ):
        assert label in proc.stdout
