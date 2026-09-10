"""CLI: build a 4-tissue tetrahedral FEM mesh via iso2mesh + CGAL."""
from __future__ import annotations

import argparse
import sys

from inob.cli._common import add_common_args, setup
from inob.fem.cgal_builder import build_fem

# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Mesh the watertight compartments into one multi-tissue tetrahedral
FEM, with a conductivity region per tissue.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob build-fem",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob build-fem                         mesh at the configured pitch
  inob build-fem --set fem.pitch_mm=2.0  finer voxels, bigger mesh, slower
  inob build-fem --set fem.maxvol=10     smaller tetrahedra

Needs geometry.mat (`inob build-geom`). Writes outputs/fem/fem.mat.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="build_fem")
    build_fem(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
