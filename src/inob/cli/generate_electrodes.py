"""CLI: generate the HD surface-electrode array."""
from __future__ import annotations

import argparse
import sys

from inob.anatomy import VERTEBRA_LEVELS
from inob.cli._common import add_common_args, setup
from inob.sensors.electrodes import generate_electrode_array


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=generate_electrode_array.__doc__)
    add_common_args(p)
    p.add_argument("--rows", type=int, default=None,
                   help="Override electrodes.rows.")
    p.add_argument("--cols", type=int, default=None,
                   help="Override electrodes.cols.")
    p.add_argument("--pitch", type=float, default=None, dest="contact_pitch_mm",
                   help="Override electrodes.contact_pitch_mm.")
    p.add_argument("--level", default=None, choices=VERTEBRA_LEVELS, metavar="LEVEL",
                   help="Centre the patch on this vertebra's Z band (e.g. 'c7') "
                        "instead of the fractional body-height slab, so it sits "
                        "over the source of interest. Overrides the per-source-"
                        "target default (spine → c7). One of: "
                        f"{', '.join(VERTEBRA_LEVELS)}.")
    p.add_argument("--full-spine", action="store_true",
                   help="Full-region survey: centre the patch on the mid-level "
                        "slab (the pre-level default) instead of the per-source-"
                        "target vertebral level. Use for a whole-cord render "
                        "where no single source is of interest. Overrides the "
                        "spine → c7 default; mutually exclusive with --level.")
    args = p.parse_args(argv)
    if args.full_spine and args.level:
        p.error("--full-spine and --level are mutually exclusive")
    cfg = setup(args, log_prefix="electrodes")
    generate_electrode_array(
        cfg, rows=args.rows, cols=args.cols,
        contact_pitch_mm=args.contact_pitch_mm,
        # "" is the explicit slab opt-out; None means "use the source-target default".
        target_level="" if args.full_spine else args.level,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
