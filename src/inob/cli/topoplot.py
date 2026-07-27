"""CLI: render dual-modality MEG/EEG topoplots."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inob.cli._common import add_common_args, setup
from inob.config import source_region_label
from inob.viz.topoplot import (
    render_dual_topoplot,
    render_eeg_topoplot,
    render_meg_montage,
    render_meg_topoplot,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--target", choices=("dual", "meg", "eeg", "montage"), default="dual",
                   help="Which topoplot to render.")
    p.add_argument("--source-idx", type=int, default=-1,
                   help="Source index along the vagus polyline (default: middle).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: cfg.outputs.base/<auto>.png).")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--n-sources", type=int, default=5,
                   help="Montage: number of sources to render.")
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="topoplot")

    import matplotlib.pyplot as plt

    # Tag default filenames with the source-target so a run for one region never
    # overwrites another's figure (e.g. muscle vs spine at the same source-idx).
    tag = source_region_label(cfg).replace(" + ", "_").replace(" ", "_")

    if args.target == "dual":
        render_dual_topoplot(cfg, source_idx=args.source_idx, out_path=args.out,
                              dpi=args.dpi)
    elif args.target == "meg":
        fig, _ = render_meg_topoplot(cfg, source_idx=args.source_idx, show_skin=True)
        out = args.out or (cfg.outputs.base / f"meg_topoplot_{tag}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
    elif args.target == "eeg":
        fig = render_eeg_topoplot(cfg, source_idx=args.source_idx)
        out = args.out or (cfg.outputs.base / f"eeg_topoplot_{tag}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
    elif args.target == "montage":
        render_meg_montage(cfg, n_sources=args.n_sources, out_path=args.out,
                            dpi=args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
