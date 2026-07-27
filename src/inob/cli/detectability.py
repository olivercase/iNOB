"""CLI: surface topoplots + SNR-vs-trials detectability charts."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from inob.anatomy import VERTEBRA_LEVELS, vertebra_z_band
from inob.cli._common import add_common_args, setup
from inob.config import Config
from inob.io.npz import load_leadfield
from inob.viz.detectability import detectability_summary, render_detectability
from inob.viz.surface_topoplot import render_surface_topoplots

logger = logging.getLogger(__name__)


def _resolve_source_idx(cfg: Config, level: str | None, source_idx: int) -> int:
    """Map ``--level`` (e.g. c7) to the cord source nearest that vertebra's Z-band
    centre, so placement and analysis name the same landmark. Without a level the
    explicit ``--source-idx`` is returned unchanged."""
    if not level:
        return source_idx
    lf = load_leadfield(cfg.outputs.forward_npz)
    z = lf.source_pos[:, 2]
    z_lo, z_hi = vertebra_z_band(cfg.data.bone_dir, level)
    centre = 0.5 * (z_lo + z_hi)
    in_band = np.flatnonzero((z >= z_lo) & (z <= z_hi))
    pool = in_band if in_band.size else np.arange(len(z))
    if not in_band.size:
        logger.warning("no sources in %s Z band [%.1f, %.1f] mm; using nearest overall",
                       level.upper(), z_lo, z_hi)
    idx = int(pool[np.argmin(np.abs(z[pool] - centre))])
    logger.info("--level %s → source #%d (z=%.1f mm)", level, idx, z[idx])
    return idx


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument(
        "--target", choices=("surface", "detect", "all"), default="all",
        help="Which figure to render (default: both).",
    )
    p.add_argument("--source-idx", type=int, default=-1,
                   help="Source index along the source polyline (default: middle). "
                        "Ignored when --level is given.")
    p.add_argument("--level", default=None, choices=VERTEBRA_LEVELS, metavar="LEVEL",
                   help="Analyse the cord source at this vertebral level (e.g. "
                        "'c7'), resolved from its segmented STL. Overrides "
                        f"--source-idx. One of: {', '.join(VERTEBRA_LEVELS)}.")
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
    source_idx = _resolve_source_idx(cfg, args.level, args.source_idx)

    if args.target in ("surface", "all"):
        render_surface_topoplots(
            cfg, source_idx=source_idx, out_path=args.out_surface,
            dpi=args.dpi,
        )
    if args.target in ("detect", "all"):
        render_detectability(
            cfg, source_idx=source_idx, out_path=args.out_detect,
            dpi=args.dpi,
            snr_threshold=args.snr_threshold, max_trials=args.max_trials,
        )

    if args.print_summary or args.target in ("detect", "all"):
        summary = detectability_summary(cfg, source_idx=source_idx)
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
