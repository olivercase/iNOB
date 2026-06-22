"""CLI: generate the triaxial OPM sensor array."""
from __future__ import annotations

import argparse
import sys

from inob.cli._common import add_common_args, setup
from inob.sensors.triaxial import generate_sensor_array


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=generate_sensor_array.__doc__)
    add_common_args(p)
    p.add_argument("--zmin", type=float, default=None,
                   help="Lower head-foot crop (mm). Default: from sensors.z_crop_low_factor.")
    p.add_argument("--zmax", type=float, default=None,
                   help="Upper head-foot crop (mm). Default: top of mesh.")
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="sensors")
    generate_sensor_array(cfg, z_min=args.zmin, z_max=args.zmax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
