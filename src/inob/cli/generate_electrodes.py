"""CLI: generate the HD surface-electrode array."""
from __future__ import annotations

import argparse
import sys

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
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="electrodes")
    generate_electrode_array(
        cfg, rows=args.rows, cols=args.cols,
        contact_pitch_mm=args.contact_pitch_mm,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
