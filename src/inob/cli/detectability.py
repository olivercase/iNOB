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
from inob.config import Config, source_target_tag
from inob.io.npz import load_leadfield
from inob.viz.detectability import (
    _compatible_eeg_leadfield,
    _optional_leadfield,
    default_source_idx,
    detectability_summary,
    fixed_q_scenarios,
    render_detectability,
)
from inob.viz.surface_topoplot import render_surface_topoplots

logger = logging.getLogger(__name__)


def _resolve_source_idx(cfg: Config, level: str | None, source_idx: int) -> int:
    """Map ``--level`` (e.g. c7) to a cord source index, so placement and analysis
    name the same landmark. Without a level the explicit ``--source-idx`` is
    returned unchanged.

    When ``level`` is the level the HD electrode patch was itself centred on
    (``cfg.electrodes.target_level``, set per-target in ``SOURCE_TARGETS``), we
    resolve via :func:`default_source_idx` — the source nearest the patch's
    actual centroid — so this agrees with the same-named default used by
    ``render_detectability``/``render_surface_topoplots`` and by
    ``source_models.py``. The patch centroid can sit tens of mm from the raw
    vertebra Z-bounding-box centre once skin curvature is accounted for, so
    these two methods silently disagreed until this unification.

    For any other level (no patch was sited there) we fall back to the
    vertebra-STL Z-band lookup, since there is no patch centroid to anchor to.
    """
    if not level:
        return source_idx
    lf = load_leadfield(cfg.outputs.forward_npz)
    if level == cfg.electrodes.target_level:
        eeg_lf = _compatible_eeg_leadfield(lf, _optional_leadfield(cfg.outputs.forward_eeg_npz))
        if eeg_lf is not None:
            idx = default_source_idx(lf, eeg_lf)
            logger.info(
                "--level %s → source #%d (z=%.1f mm) via patch centroid",
                level,
                idx,
                lf.source_pos[idx, 2],
            )
            return idx
    z = lf.source_pos[:, 2]
    z_lo, z_hi = vertebra_z_band(cfg.data.bone_dir, level)
    centre = 0.5 * (z_lo + z_hi)
    in_band = np.flatnonzero((z >= z_lo) & (z <= z_hi))
    pool = in_band if in_band.size else np.arange(len(z))
    if not in_band.size:
        logger.warning(
            "no sources in %s Z band [%.1f, %.1f] mm; using nearest overall",
            level.upper(),
            z_lo,
            z_hi,
        )
    idx = int(pool[np.argmin(np.abs(z[pool] - centre))])
    logger.info("--level %s → source #%d (z=%.1f mm) via vertebra Z-band", level, idx, z[idx])
    return idx


# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Work out how many averaged trials each source needs before it
clears the noise floor, and draw the charts that show it.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob detect",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob detect                            trials to detect, for the whole array
  inob detect --snr-threshold 5          a stricter detection criterion
  inob detect --q-nAm 20                 assume a 20 nA·m source
  inob detect --level c7                 only sources at one vertebral level
  inob detect --print-summary            JSON to stdout as well as the figure

The planning question this whole package exists to answer.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument(
        "--target",
        choices=("surface", "detect", "all"),
        default="all",
        help="Which figure to render (default: both).",
    )
    p.add_argument(
        "--source-idx",
        type=int,
        default=-1,
        help="Source index along the source polyline (default: middle). "
        "Ignored when --level is given.",
    )
    p.add_argument(
        "--level",
        default=None,
        choices=VERTEBRA_LEVELS,
        metavar="LEVEL",
        help="Analyse the cord source at this vertebral level (e.g. "
        "'c7'), resolved from its segmented STL. Overrides "
        f"--source-idx. One of: {', '.join(VERTEBRA_LEVELS)}.",
    )
    p.add_argument(
        "--snr-threshold",
        type=float,
        default=3.0,
        help="Detection threshold (post-averaging SNR; default 3 = Rose criterion).",
    )
    p.add_argument(
        "--max-trials",
        type=int,
        default=1_000_000,
        help="Upper bound for the trials axis.",
    )
    p.add_argument(
        "--q-nAm",
        type=float,
        default=None,
        dest="q_nAm",
        metavar="Q",
        nargs="+",
        help="Run at explicit source strengths instead of the target's own Q "
        "ladder — e.g. --source-target vagus --q-nAm 5.11 puts the vagus "
        "at the spine's magnetospinography anchor, and --q-nAm 1 5.11 10 "
        "20 plots it over the cord's whole reported range.",
    )
    p.add_argument("--out-surface", type=Path, default=None)
    p.add_argument("--out-detect", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the JSON detectability summary to stdout.",
    )
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="detectability")
    source_idx = _resolve_source_idx(cfg, args.level, args.source_idx)
    scenarios = fixed_q_scenarios(*args.q_nAm) if args.q_nAm else None
    if scenarios is not None:
        logger.info(
            "source strengths fixed at %s nA·m (--q-nAm); the %s physiology ladder is not used",
            ", ".join(f"{q:g}" for q in sorted(args.q_nAm)),
            source_target_tag(cfg) or "default",
        )

    if args.target in ("surface", "all"):
        render_surface_topoplots(
            cfg,
            source_idx=source_idx,
            out_path=args.out_surface,
            dpi=args.dpi,
        )
    if args.target in ("detect", "all"):
        render_detectability(
            cfg,
            source_idx=source_idx,
            out_path=args.out_detect,
            dpi=args.dpi,
            scenarios=scenarios,
            snr_threshold=args.snr_threshold,
            max_trials=args.max_trials,
        )

    if args.print_summary or args.target in ("detect", "all"):
        summary = detectability_summary(cfg, source_idx=source_idx, scenarios=scenarios)
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
