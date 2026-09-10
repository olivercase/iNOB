"""CLI: build the multi-compartment geometry HDF5."""

from __future__ import annotations

import argparse
import sys

from inob.cli._common import add_common_args, setup
from inob.geometry.builder import build_geometry, check_existing

# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Turn the raw anatomical STLs into watertight, non-overlapping
compartments that the mesher can use.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob build-geom",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob build-geom                        watertighten every configured tissue
  inob build-geom --only skin,bone       just two compartments
  inob build-geom --check-only           report what exists; build nothing
  inob build-geom --shrinkwrap-only      skip repair/union, use the voxel path

Writes outputs/geometry/geometry.mat. `inob run` calls this as its first stage.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument(
        "--shrinkwrap-only",
        action="store_true",
        help="Skip cheap repair / boolean union; force the voxel shrinkwrap pipeline.",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate existing compartments only (do not rebuild).",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated compartment names to build (default: all).",
    )
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="build_geom")

    if args.check_only:
        ok = check_existing(cfg)
        return 0 if ok else 1

    only = tuple(s.strip() for s in args.only.split(",")) if args.only else None
    build_geometry(cfg, force_shrinkwrap=args.shrinkwrap_only, only_compartments=only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
