"""CLI: predict per-source SNR for a leadfield (MEG or EEG)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from inob.analysis.snr import (
    array_summary,
    compute_noise_floors,
    snr_per_source,
)
from inob.cli._common import add_common_args, setup
from inob.io.npz import load_leadfield

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--leadfield", type=Path, default=None,
                   help="Override leadfield NPZ path (default: cfg.outputs.forward_npz).")
    p.add_argument("--modality", choices=("meg", "eeg"), default="meg")
    p.add_argument("--moment", choices=("rms", "max"), default="rms")
    p.add_argument("--n-averages", type=int, default=1,
                   help="Number of trial averages (SNR ∝ √n).")
    p.add_argument("--out", type=Path, default=None,
                   help="JSON output path (default: stdout only).")
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix=f"snr_{args.modality}")

    lf_path = args.leadfield or (
        cfg.outputs.forward_npz if args.modality == "meg" else cfg.outputs.forward_eeg_npz
    )
    lf = load_leadfield(lf_path)
    floors = compute_noise_floors(cfg)
    sigma = floors.meg_per_channel_fT if args.modality == "meg" \
            else floors.eeg_per_channel_uV
    snr = snr_per_source(
        lf.L_fT_per_nAm, sigma, moment=args.moment,
        n_averages=args.n_averages,
    )
    summary = {
        "modality": args.modality,
        "moment": args.moment,
        "n_averages": args.n_averages,
        "sigma_per_channel": float(sigma),
        "leadfield_path": str(lf_path),
        "snr_summary": array_summary(snr),
    }
    msg = json.dumps(summary, indent=2)
    sys.stdout.write(msg + "\n")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(msg)
        logger.info("[saved] %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
