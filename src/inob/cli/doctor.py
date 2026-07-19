"""``inob doctor`` — check the environment before a long run fails halfway.

Targets the failure modes that actually bite: anatomical meshes left as Git LFS
pointer stubs, a missing ``duneuropy`` extension, an unparseable config, and an
unwritable outputs directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from inob.cli import _ui
from inob.cli._common import DEFAULT_CONFIG

# Runtime imports that every stage needs, as (module, pypi name).
CORE_MODULES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"), ("scipy", "scipy"), ("h5py", "h5py"),
    ("trimesh", "trimesh"), ("skimage", "scikit-image"),
    ("matplotlib", "matplotlib"), ("pyvista", "pyvista"),
    ("iso2mesh", "iso2mesh"), ("pymeshfix", "pymeshfix"),
    ("yaml", "pyyaml"), ("rtree", "rtree"),
)

LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    hints: list[str] = field(default_factory=list)


def check_python() -> Check:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < (3, 11):
        return Check("Python", FAIL, f"{version} — iNOB needs 3.11 or newer",
                     ["Install Python 3.11+ and recreate your environment."])
    return Check("Python", OK, f"{version} ({sys.executable})")


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_core_deps() -> Check:
    missing = [pypi for mod, pypi in CORE_MODULES if not _importable(mod)]
    if missing:
        return Check(
            "Core dependencies", FAIL,
            f"{len(missing)} missing: {', '.join(missing)}",
            ["python3 -m pip install -e .[dev]"],
        )
    return Check("Core dependencies", OK,
                 f"all {len(CORE_MODULES)} present")


def check_duneuro() -> Check:
    """Optional: only ``inob forward`` and ``inob eeg`` need it."""
    if _importable("duneuropy"):
        return Check("DUNEuro (duneuropy)", OK, "importable")
    return Check(
        "DUNEuro (duneuropy)", WARN,
        "not importable — every stage works except `inob forward` / `inob eeg`",
        ["Build it: https://github.com/olivercase/duneuro-build",
         "Then run inob from that venv's interpreter."],
    )


def _stl_files(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.rglob("*.stl"))


def _is_lfs_pointer(path: Path) -> bool:
    """LFS stubs are ~132-byte text files with a known first line."""
    try:
        if path.stat().st_size > 1024:
            return False
        with path.open("rb") as fh:
            return fh.read(len(LFS_MAGIC)) == LFS_MAGIC
    except OSError:
        return False


def check_meshes(project_root: Path) -> Check:
    data_dir = project_root / "data"
    stls = _stl_files(data_dir)
    if not stls:
        return Check(
            "Anatomical meshes", FAIL, f"no .stl files under {data_dir}",
            ["git lfs install && git lfs pull"],
        )
    pointers = [p for p in stls if _is_lfs_pointer(p)]
    if pointers:
        sample = ", ".join(_ui.rel(p, project_root) for p in pointers[:3])
        more = f" (+{len(pointers) - 3} more)" if len(pointers) > 3 else ""
        return Check(
            "Anatomical meshes", FAIL,
            f"{len(pointers)} of {len(stls)} are Git LFS pointers, not meshes: "
            f"{sample}{more}",
            ["git lfs install", "git lfs pull"],
        )
    return Check("Anatomical meshes", OK, f"{len(stls)} STL files present")


def check_config(config_path: Path,
                 project_root: Path | None) -> tuple[Check, Path | None]:
    """Returns the check plus the resolved project root when the config loads."""
    from inob.config import load_config
    if not config_path.exists():
        return Check("Config", FAIL, f"{config_path} does not exist",
                     [f"Pass --config, or restore {DEFAULT_CONFIG}."]), None
    try:
        cfg = load_config(config_path, project_root=project_root)
    except Exception as exc:  # ConfigError and anything YAML throws
        return Check("Config", FAIL, f"{config_path} failed to load: {exc}",
                     ["Fix the reported field, or start from "
                      "configs/default.yaml."]), None
    return Check("Config", OK,
                 f"{config_path} loads, project root {cfg.project_root}"), \
        cfg.project_root


def check_outputs_writable(project_root: Path) -> Check:
    out = project_root / "outputs"
    probe = out / ".inob-doctor-write-test"
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check("Outputs directory", FAIL, f"{out} is not writable: {exc}",
                     ["Check permissions or free disk space."])
    return Check("Outputs directory", OK, f"{out} is writable")


def run_checks(config_path: Path, project_root: Path | None) -> list[Check]:
    from inob.paths import find_project_root
    config_check, resolved_root = check_config(config_path, project_root)
    # Fall back to marker-file discovery so the mesh/outputs checks still run
    # even when the config itself is the thing that's broken.
    root = resolved_root or project_root or find_project_root(config_path)
    return [
        check_python(),
        check_core_deps(),
        config_check,
        check_meshes(root),
        check_outputs_writable(root),
        check_duneuro(),
    ]


def _render(checks: list[Check], out) -> None:
    glyph_for = {OK: "ok", WARN: "warn", FAIL: "fail"}
    print(_ui.heading("Environment checks"), file=out)
    for check in checks:
        print(f"  {_ui.mark(glyph_for[check.status])} "
              f"{check.name:<22}{check.detail}", file=out)
        for hint in check.hints:
            print(f"      {_ui.arrow()} {_ui.paint(hint, 'cyan')}", file=out)

    failures = [c for c in checks if c.status == FAIL]
    warnings = [c for c in checks if c.status == WARN]
    print("", file=out)
    if failures:
        print(_ui.paint(
            f"{len(failures)} problem(s) will stop the pipeline. "
            f"Fix the arrows above.", "red"), file=out)
    elif warnings:
        print(_ui.paint(
            "Ready to run, with limits noted above.", "yellow"), file=out)
    else:
        print(_ui.paint("Everything checks out. Try: inob run", "green"),
              file=out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob doctor",
        description="Check that this machine can run the pipeline.",
    )
    p.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG),
                   help=f"Path to YAML config (default {DEFAULT_CONFIG}).")
    p.add_argument("--project-root", type=Path, default=None,
                   help="Override the project root used to resolve paths.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    args = p.parse_args(argv)

    checks = run_checks(args.config, args.project_root)

    if args.json:
        json.dump(
            {"checks": [
                {"name": c.name, "status": c.status,
                 "detail": c.detail, "hints": c.hints} for c in checks],
             "ok": all(c.status != FAIL for c in checks)},
            sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _render(checks, sys.stdout)

    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
