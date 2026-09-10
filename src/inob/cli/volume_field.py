"""CLI: read the FEM solution *inside* the volume, not just at the sensors.

Two modes, matching the two things DUNEuro's bindings actually expose:

  ``--source``      one dipole's potential over the whole mesh, written as VTK
                    (potential per vertex, gradient per cell) for ParaView.
  ``--stimulation`` the field a bipolar electrode montage drives through the
                    volume, sampled at points inside the mesh and written as an
                    NPZ the GUI can render as a 3-D cloud. This is reciprocity
                    used forwards — DUNEuro's own tDCS forward problem is the
                    EEG transfer matrix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from inob.cli._common import add_common_args, setup
from inob.config import source_tissue_labels
from inob.forward.volume_field import (
    EVALUATION_TYPES,
    export_source_field_vtk,
    save_volume_field,
    stimulation_field,
)
from inob.io.hdf5 import load_fem, load_sensors, validate_fem

# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Read the solved field inside the body, not just at the sensors.

Two modes:

  --source        one dipole's potential over the whole mesh, written as
                  VTK (potential per vertex, gradient per cell) for
                  ParaView.
  --stimulation   the field a bipolar electrode montage drives through the
                  volume, sampled on a grid inside the mesh and written as
                  an NPZ the GUI renders as a 3-D cloud.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob volume-field",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob volume-field --source                     one dipole's own field, to VTK
  inob volume-field --evaluate --spacing-mm 5    the field as numbers on a grid
  inob volume-field --stimulation \\
      --anode e01 --cathode e17 --current-mA 1.0   a 1 mA bipolar montage

Reads the FEM solution inside the body rather than only at the sensors.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--source", action="store_true",
        help="Export one dipole's potential field over the whole mesh as VTK "
             "(the default mode).",
    )
    mode.add_argument(
        "--stimulation", action="store_true",
        help="Compute the field of a bipolar electrode montage instead. Needs "
             "the HD electrode array (inob electrodes).",
    )
    p.add_argument("--source-idx", type=int, default=0,
                   help="Which sampled source to use for --source (default 0).")
    p.add_argument("--moment", default="z", choices=("x", "y", "z"),
                   help="Dipole moment direction for --source (default z, the "
                        "longitudinal direction for a nerve).")
    p.add_argument("--anode", type=int, default=0,
                   help="--stimulation: electrode index current is driven into.")
    p.add_argument("--cathode", type=int, default=1,
                   help="--stimulation: electrode index current returns from.")
    p.add_argument("--current-mA", type=float, default=1.0,
                   help="--stimulation: injected current in mA (default 1).")
    p.add_argument(
        "--evaluate", default="current", choices=list(EVALUATION_TYPES),
        help="--stimulation: what to return per point. 'direct' = potential, "
             "'gradient' = grad u (the E field up to sign), 'current' = "
             "-sigma*grad u (current density, the default).",
    )
    p.add_argument("--spacing-mm", type=float, default=5.0,
                   help="--stimulation: thin the sample points to roughly one "
                        "per cube of this size (0 = every tetrahedron).")
    p.add_argument("--tissue", default=None,
                   help="--stimulation: restrict sample points to these tissue "
                        "labels (comma-separated). Default: the whole volume.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path. Default: outputs/volume_field/<mode>.")
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="volume_field")

    fem = load_fem(cfg.outputs.fem_mat)
    validate_fem(fem)
    out_dir = cfg.outputs.base / "volume_field"

    if args.stimulation:
        electrodes = load_sensors(cfg.outputs.electrodes_mat)
        tissues = (tuple(source_tissue_labels(args.tissue)) if args.tissue else ())
        field = stimulation_field(
            cfg, fem, electrodes.coilpos,
            anode=args.anode, cathode=args.cathode, current_mA=args.current_mA,
            evaluation_type=args.evaluate, spacing_mm=args.spacing_mm,
            tissues=tissues,
        )
        out = args.out or (out_dir / "stimulation_field.npz")
        save_volume_field(out, field)
        mag = field.magnitude
        sys.stdout.write(
            f"\n[volume field] {field.description}\n"
            f"  points : {len(field.positions_mm)}\n"
            f"  |value|: max {mag.max():.4e}  median {np.median(mag):.4e}\n"
            f"  written: {out}\n"
        )
        return 0

    from inob.sources.vagus import resolve_source_positions

    src_pos = resolve_source_positions(cfg, fem)
    if not 0 <= args.source_idx < len(src_pos):
        raise SystemExit(
            f"--source-idx {args.source_idx} out of range (have {len(src_pos)})")
    moment = np.eye(3)["xyz".index(args.moment)]
    out = args.out or (out_dir / f"source_{args.source_idx}_{args.moment}")
    written = export_source_field_vtk(
        cfg, fem, source_pos_mm=src_pos[args.source_idx], moment=moment,
        out_path=out,
    )
    sys.stdout.write(
        f"\n[volume field] dipole {args.source_idx} at "
        f"{np.round(src_pos[args.source_idx], 1).tolist()} mm, moment {args.moment}\n"
        f"  written: {written}\n"
        "  open in ParaView: 'potential' on the vertices, 'gradient' per cell.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
