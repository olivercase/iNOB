"""CLI: surface topoplots + SNR-vs-trials detectability charts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inob.cli._common import add_common_args, setup
from inob.viz.detectability import detectability_summary, render_detectability
from inob.viz.surface_topoplot import render_surface_topoplots


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument(
        "--target", choices=("surface", "detect", "all"), default="all",
        help="Which figure to render (default: both).",
    )
    p.add_argument("--source-idx", type=int, default=-1,
                   help="Source index along the vagus polyline (default: middle).")
    p.add_argument(
        "--snr-threshold", type=float, default=3.0,
        help="Detection threshold (post-averaging SNR; default 3 = Rose criterion).",
    )
    p.add_argument(
        "--max-trials", type=int, default=1_000_000,
        help="Upper bound for the trials axis.",
    )
    p.add_argument("--out-surface", type=Path, default=None)
    p.add_argument("--out-detect", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--print-summary", action="store_true",
        help="Print the JSON detectability summary to stdout.",
    )
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="detectability")

    if args.target in ("surface", "all"):
        render_surface_topoplots(
            cfg, source_idx=args.source_idx, out_path=args.out_surface,
            dpi=args.dpi,
        )
    if args.target in ("detect", "all"):
        render_detectability(
            cfg, source_idx=args.source_idx, out_path=args.out_detect,
            dpi=args.dpi,
            snr_threshold=args.snr_threshold, max_trials=args.max_trials,
        )

    if args.print_summary or args.target in ("detect", "all"):
        summary = detectability_summary(cfg, source_idx=args.source_idx)
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
