"""CLI: render geometry + FEM visualisation PNGs."""

from __future__ import annotations

import argparse
import sys

from inob.cli._common import add_common_args, setup
from inob.viz.fem import render_fem
from inob.viz.geometry import render_geometry

# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Render overview PNGs of the anatomical geometry and the FEM mesh.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob visualise",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob visualise                         geometry and FEM overview PNGs
  inob visualise --target geom           just the anatomy
  inob visualise --no-sensors            leave the array off the render

The same renders `inob run --with-viz` produces as its viz stage.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument(
        "--target",
        choices=("geom", "fem", "all"),
        default="all",
        help="Which visualisation to render (default: all).",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--show",
        action="store_true",
        help="Show interactively (matplotlib for geom; PyVista for fem).",
    )
    p.add_argument(
        "--no-sensors", action="store_true", help="Skip sensor overlay on the geometry view."
    )
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="visualise")

    if args.target in ("geom", "all"):
        render_geometry(cfg, dpi=args.dpi, show=args.show, with_sensors=not args.no_sensors)
    if args.target in ("fem", "all"):
        render_fem(cfg, interactive=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
