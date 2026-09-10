"""CLI: characterise the sensor field pattern for a source region.

Renders the sensor-field figure (dipolar pattern, topography, along-axis
strength, falloff) and prints quantitative characteristics. Use
``--source-target spine`` to analyse a specific region's leadfield.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inob.anatomy import VERTEBRA_LEVELS
from inob.cli._common import add_common_args, setup
from inob.io.npz import load_leadfield
from inob.viz.sensor_field import (
    peak_amplitude_along_axis,
    region_summary,
    render_aggregate_field,
    render_sensor_field,
    sensor_field_characteristics,
    sources_in_z_band,
    vertebra_z_band,
)

_MOMENTS = {"dominant": None, "x": 0, "y": 1, "z": 2}


# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Characterise what the array actually sees from a source region.

Draws the dipolar pattern, the topography, the along-axis strength and the
falloff with distance, and prints the same numbers. Use --source-target to
characterise a different region's leadfield.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob sensor-field",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob sensor-field                                 strongest source's pattern
  inob sensor-field --aggregate rms                 the array's sensitivity map
  inob sensor-field --aggregate coherent            every source summed in step
  inob sensor-field --level c7                      one vertebral level only
  inob sensor-field --no-figure --print-summary     the numbers only

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument("--source-idx", type=int, default=-1,
                   help="Source index; default (-1) picks the strongest source.")
    p.add_argument("--moment", choices=tuple(_MOMENTS), default="dominant",
                   help="Source-moment orientation to visualise (default: dominant).")
    p.add_argument("--aggregate", choices=("coherent", "rms"), default=None,
                   help="Show the global field from ALL sources instead of one: "
                        "'coherent' (in-phase sum) or 'rms' (sensitivity map). "
                        "Defaults to the longitudinal (z) moment.")
    p.add_argument("--level", choices=VERTEBRA_LEVELS, default=None,
                   metavar="LEVEL",
                   help="Restrict to cord sources at one vertebral level (e.g. "
                        "'c7'). The Z band is taken from that vertebra's segmented "
                        "STL in the bone_dir. Mutually exclusive with --z-range. "
                        f"One of: {', '.join(VERTEBRA_LEVELS)}.")
    p.add_argument("--z-range", type=float, nargs=2, metavar=("ZLO", "ZHI"),
                   default=None,
                   help="Restrict to cord sources with ZLO <= z <= ZHI (mm). "
                        "Escape hatch when a level isn't segmented.")
    p.add_argument("--out", type=Path, default=None, help="Figure output path.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-figure", action="store_true",
                   help="Skip the figure; print characteristics only.")
    p.add_argument("--print-summary", action="store_true",
                   help="Print JSON characteristics to stdout.")
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="sensor_field")

    if args.level and args.z_range:
        p.error("--level and --z-range are mutually exclusive")

    moment = _MOMENTS[args.moment]

    # Resolve an optional Z band from a vertebral level or an explicit range.
    z_range: tuple[float, float] | None = None
    region_label: str | None = None
    if args.level:
        z_range = vertebra_z_band(cfg, args.level)
        region_label = args.level
    elif args.z_range:
        z_range = (min(args.z_range), max(args.z_range))
        region_label = f"z{z_range[0]:.0f}-{z_range[1]:.0f}"

    if args.aggregate:
        # Aggregate needs a concrete moment; default to longitudinal (z).
        agg_moment = moment if moment is not None else 2
        render_aggregate_field(cfg, moment=agg_moment, mode=args.aggregate,
                               z_range=z_range, region_label=region_label,
                               out_path=args.out, dpi=args.dpi)
    elif not args.no_figure:
        render_sensor_field(cfg, source_idx=args.source_idx, moment=moment,
                            z_range=z_range, region_label=region_label,
                            out_path=args.out, dpi=args.dpi)

    if args.print_summary or args.no_figure:
        lf = load_leadfield(cfg.outputs.forward_npz)
        region = region_summary(lf)
        if args.source_idx >= 0:
            idx = args.source_idx
        elif z_range is not None:
            # Strongest source within the band (matches what the figure shows).
            band = sources_in_z_band(lf, *z_range)
            _, peaks = peak_amplitude_along_axis(lf)
            idx = int(band[peaks[band].argmax()])
        else:
            idx = region["strongest_source_idx"]
        summary = {
            "leadfield_path": str(cfg.outputs.forward_npz),
            "region": region,
            "level": region_label,
            "source": sensor_field_characteristics(lf, idx, moment=moment),
        }
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
