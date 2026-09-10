"""CLI: generate the triaxial OPM sensor array."""

from __future__ import annotations

import argparse
import sys

from inob.cli._common import add_common_args, setup
from inob.sensors.triaxial import generate_sensor_array

# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Place the triaxial OPM array on a stand-off shell wrapped around
the body.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob sensors",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob sensors                                 the whole configured torso wrap
  inob sensors --zmin 1200 --zmax 1450         cervical band only
  inob sensors --set sensors.resolution_mm=20  a denser array

Writes outputs/sensors/sensor_array.mat: one position, three coils (R, T1, T2).

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument(
        "--zmin",
        type=float,
        default=None,
        help="Lower head-foot crop (mm). Default: from sensors.z_crop_low_factor.",
    )
    p.add_argument(
        "--zmax", type=float, default=None, help="Upper head-foot crop (mm). Default: top of mesh."
    )
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="sensors")
    generate_sensor_array(cfg, z_min=args.zmin, z_max=args.zmax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
