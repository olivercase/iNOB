"""CLI: the four-panel field map painted on the body surface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inob.cli._common import add_common_args, setup
from inob.viz.torso_topoplot import render_torso_topoplot


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob torso",
        description=__doc__,
        epilog="""\
examples:
  inob torso                             four views, field painted on the body
  inob torso --moment z                  the longitudinal moment only
  inob torso --source-idx 40             one source by index

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument("--source-idx", type=int, default=-1,
                   help="Which source to map. Default: the one the electrode "
                        "patch was sited over.")
    p.add_argument("--moment", default="z", choices=("x", "y", "z"),
                   help="Dipole moment direction (default z, longitudinal for "
                        "a nerve or cord).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG (default: outputs/torso_topoplot_<target>.png).")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="torso_topoplot")
    out = render_torso_topoplot(cfg, source_idx=args.source_idx,
                                moment=args.moment, out_path=args.out,
                                dpi=args.dpi)
    print(f"figure: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
