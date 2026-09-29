"""CLI: generate the OPM sensor array, or import one from a file."""

from __future__ import annotations

import argparse
import sys

from inob.cli._common import add_common_args, setup
from inob.sensors.triaxial import generate_sensor_array

# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Place the OPM array on a stand-off shell wrapped around the body, or
import a real array from a file.
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
  inob sensors --set sensors.axes=radial       one coil per position
  inob sensors --from my_array.csv             your own array instead

Writes outputs/sensors/sensor_array.mat. With sensors.axes=triaxial (the
default) each position has three coils: R, T1, T2. With sensors.axes=radial
it has R alone. R is the outward skin normal under the sensor, so a
positive reading is field pointing out of the body.

--from FILE (or sensors.from_file) takes a magnetometer array as:
  .csv/.txt  one coil per row: x y z nx ny nz, in mm
  .npz       coil_pos and coil_orient arrays (mm, or set a 'unit' entry)
  .mat       a FieldTrip grad: coilpos/coilori, or chanpos/chanori, with
             its unit (m, cm or mm). Magnetometers only.
The positions must already be in the model's body frame, in which +X is
left, +Y posterior (anterior is -Y) and +Z superior; see
docs/COORDINATES.md. Registering the array is your responsibility: the
command warns when coils sit inside the skin or far from it, but cannot
tell a mis-registered array from an unusual one. Stand-off and axes
settings do not apply to an imported array.

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
    p.add_argument(
        "--from",
        dest="from_file",
        default=None,
        metavar="FILE",
        help="Import this array instead of generating one (see below).",
    )
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="sensors")
    generate_sensor_array(cfg, z_min=args.zmin, z_max=args.zmax, from_file=args.from_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
