"""CLI: build a 4-tissue tetrahedral FEM mesh via iso2mesh + CGAL."""
from __future__ import annotations

import argparse
import sys

from inob.cli._common import add_common_args, setup
from inob.fem.cgal_builder import build_fem


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=build_fem.__doc__)
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="build_fem")
    build_fem(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
