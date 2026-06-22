"""CLI: render the propagating-CAP vs stationary-dipole comparison figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vagus_fm.cli._common import add_common_args, setup
from vagus_fm.viz.cap_compare import render_cap_compare


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--ap-width-ms", type=float, default=0.5,
                   help="Width parameter (σ) of the biphasic action-potential shape.")
    p.add_argument("--fs-hz", type=float, default=30_000.0,
                   help="Sampling frequency (Hz). Default 30 kHz captures sub-AP detail.")
    p.add_argument("--duration-ms", type=float, default=30.0,
                   help="Trace duration (ms). Default 30 ms = ample for one CAP event.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG (default: cfg.outputs.base/cap_compare.png).")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="cap_compare")
    render_cap_compare(
        cfg,
        ap_width_ms=args.ap_width_ms,
        fs_hz=args.fs_hz,
        duration_ms=args.duration_ms,
        out_path=args.out,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
