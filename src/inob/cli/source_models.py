"""CLI: OPM vs electrodes on a stationary and an ascending cord source."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inob.analysis.source_models import source_model_summary
from inob.cli._common import add_common_args, setup
from inob.viz.source_models_plot import render_source_models


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument(
        "--Q-nAm", type=float, default=None, dest="Q_nAm",
        help="Moment per active source (default: the target profile's). Only "
             "scales the absolute amplitudes — every ratio reported is "
             "independent of it.",
    )
    p.add_argument(
        "--source-idx", type=int, default=None,
        help="Which source the stationary/ascending models use (default: the "
             "one the electrode array is sited over).",
    )
    p.add_argument("--json-out", type=Path, default=None,
                   help="Also write the summary to this path.")
    p.add_argument("--out", type=Path, default=None,
                   help="Figure path (default: outputs/source_models_<target>.png).")
    p.add_argument("--no-figure", action="store_true",
                   help="Print the numbers without rendering the figure.")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="source_models")
    if not args.no_figure:
        render_source_models(cfg, Q_nAm=args.Q_nAm, source_idx=args.source_idx,
                             out_path=args.out, dpi=args.dpi)
    summary = source_model_summary(cfg, Q_nAm=args.Q_nAm,
                                   source_idx=args.source_idx)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
