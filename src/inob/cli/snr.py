"""CLI: predict per-source SNR for a leadfield (MEG or EEG)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from inob.analysis.reference import parse_reference, rereference
from inob.analysis.snr import (
    array_summary,
    compute_noise_floors,
    parse_orientation,
    snr_per_source,
)
from inob.cli._common import (
    add_common_args,
    add_eeg_reference_arg,
    add_orientation_arg,
    setup,
)
from inob.io.npz import load_leadfield

logger = logging.getLogger(__name__)


# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Predict the signal-to-noise ratio of every source at the configured
noise floor, for either modality.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob snr",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob snr                                      per-source SNR, as configured
  inob snr --modality eeg                       the electric leadfield instead
  inob snr --n-averages 100                     SNR after averaging 100 trials
  inob snr --set noise.opm_sensor=fieldline_v3  a different OPM
  inob snr --orientation posterior              dipole out of the back
  inob snr --modality eeg --eeg-reference farthest
                                                electrodes against the
                                                contact farthest away

Signal is the channel-RMS amplitude of each source. --moment rms averages
the three axis directions; --moment max takes, per channel, the direction
that channel is most sensitive to. --orientation fixes one direction for
every channel instead and replaces --moment. Electrode amplitudes are single
contacts against --eeg-reference (default: the average of all contacts).

Noise floors come from cfg.noise — a sensor is a density AND a bandwidth.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument(
        "--leadfield",
        type=Path,
        default=None,
        help="Override leadfield NPZ path (default: cfg.outputs.forward_npz).",
    )
    p.add_argument("--modality", choices=("meg", "eeg"), default="meg")
    p.add_argument(
        "--moment",
        choices=("rms", "max"),
        default="rms",
        help="rms: average over x, y, z moments; max: per channel, the most "
        "sensitive direction (default rms).",
    )
    add_orientation_arg(p, default_help="Default: --moment decides. Replaces --moment when given.")
    add_eeg_reference_arg(p)
    p.add_argument("--n-averages", type=int, default=1, help="Number of trial averages (SNR ∝ √n).")
    p.add_argument(
        "--out", type=Path, default=None, help="JSON output path (default: stdout only)."
    )
    args = p.parse_args(argv)
    if args.eeg_reference is not None and args.modality != "eeg":
        p.error("--eeg-reference applies to --modality eeg only")
    cfg = setup(args, log_prefix=f"snr_{args.modality}")

    lf_path = args.leadfield or (
        cfg.outputs.forward_npz if args.modality == "meg" else cfg.outputs.forward_eeg_npz
    )
    lf = load_leadfield(lf_path)
    floors = compute_noise_floors(cfg)
    sigma = floors.meg_per_channel_fT if args.modality == "meg" else floors.eeg_per_channel_uV
    orientation = parse_orientation(args.orientation)
    L = lf.L_fT_per_nAm
    if args.eeg_reference is not None:
        L = rereference(
            L,
            args.eeg_reference,
            electrode_pos=lf.coil_pos,
            source_pos=lf.source_pos,
        )
    snr = snr_per_source(
        L,
        sigma,
        moment=args.moment,
        n_averages=args.n_averages,
        orientation=orientation,
    )
    summary = {
        "modality": args.modality,
        "moment": args.moment,
        "n_averages": args.n_averages,
        "sigma_per_channel": float(sigma),
        "leadfield_path": str(lf_path),
        "snr_summary": array_summary(snr),
    }
    # Only named when chosen, so the default output is unchanged.
    if orientation is not None:
        summary["orientation"] = [float(v) for v in orientation]
    if args.eeg_reference is not None:
        summary["eeg_reference"] = parse_reference(args.eeg_reference).label
    msg = json.dumps(summary, indent=2)
    sys.stdout.write(msg + "\n")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(msg)
        logger.info("[saved] %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
