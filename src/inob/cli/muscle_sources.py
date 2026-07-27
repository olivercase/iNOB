"""CLI: render pre-solve muscle source-dipole figures (pairing + orientation).

Muscle-specific and opt-in: this is the only command that imports
:mod:`inob.sources.muscle` for figure rendering, so a vagus or spine run
(``inob visualise``, ``inob cap-compare``, ...) never pulls in muscle
sourcing code. Requires a built FEM (``inob build-fem``) with the muscle
compartment; does not require a forward solve or ``--source-target muscle``,
since the FEM mesh carries every tissue regardless of the current target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inob.cli._common import add_common_args, setup
from inob.viz.muscle_sources import (
    render_muscle_source_orientations,
    render_muscle_source_pairs,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--which", choices=("pairs", "orientations", "all"), default="all",
                   help="Which figure(s) to render (default: all).")
    p.add_argument("--spacing-mm", type=float, default=15.0,
                   help="Volume-fill dipole spacing (mm), orientation figure only. "
                        "Default 15 mm matches the cluster's muscle SOURCE_SPACING "
                        "default (~380 sources instead of thousands at the "
                        "pipeline's 5 mm default).")
    p.add_argument("--out-pairs", type=Path, default=None,
                   help="Output PNG for the pairing figure "
                        "(default: outputs/muscle_sources_pairs.png).")
    p.add_argument("--out-orientations", type=Path, default=None,
                   help="Output PNG for the orientation figure "
                        "(default: outputs/muscle_sources_orientations.png).")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="muscle_sources")

    if args.which in ("pairs", "all"):
        render_muscle_source_pairs(
            cfg, out_path=args.out_pairs, dpi=args.dpi,
        )
    if args.which in ("orientations", "all"):
        render_muscle_source_orientations(
            cfg, spacing_mm=args.spacing_mm, out_path=args.out_orientations, dpi=args.dpi,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
