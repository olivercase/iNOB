"""CLI: whole-body EEG sensitivity comparison vs the cervical paddle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inob.cli._common import add_common_args, setup
from inob.config import source_target_tag, tag_path
from inob.viz.location_optimisation import render_location_optimisation


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob location",
        description=__doc__,
        epilog="""\
examples:
  inob location                          whole-body array vs cervical paddle
  inob location --source-idx 40          judge the comparison at one source

Answers where to put the electrodes, given two arrays already solved.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    # The paddle side and the wholebody leadfield both depend on the source
    # region, so their defaults are left as None and resolved from the
    # source-target-aware config below — hardcoding vagus paths here would make
    # e.g. `--source-target spine` compare spine sources against a vagus paddle.
    p.add_argument(
        "--paddle-mat", type=Path, default=None,
        help="Cervical paddle electrode positions "
             "(default: this target's outputs.electrodes_mat).",
    )
    p.add_argument(
        "--paddle-npz", type=Path, default=None,
        help="Paddle EEG leadfield NPZ "
             "(default: this target's outputs.forward_eeg_npz).",
    )
    p.add_argument(
        "--wholebody-mat", type=Path,
        default=Path("outputs/sensors/electrode_array_wholebody.mat"),
        help="Whole-body electrode positions (target-independent net).",
    )
    p.add_argument(
        "--wholebody-npz", type=Path, default=None,
        help="Whole-body EEG leadfield NPZ (default: "
             "duneuro_eeg_leadfield_wholebody.npz tagged with the source target).",
    )
    p.add_argument("--source-idx", type=int, default=-1)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="location")
    # The paddle leadfield/electrodes follow --source-target (setup() has already
    # repointed forward_eeg_npz / electrodes_mat). The wholebody leadfield shares
    # the same sources, so it is tagged with the same slug; only its electrode
    # net (positions) is source-independent and stays untagged.
    tag = source_target_tag(cfg)
    paddle_npz = args.paddle_npz or cfg.outputs.forward_eeg_npz
    paddle_mat = args.paddle_mat or cfg.outputs.electrodes_mat
    wholebody_npz = args.wholebody_npz or tag_path(
        Path("outputs/forward/duneuro_eeg_leadfield_wholebody.npz"), tag
    )
    render_location_optimisation(
        cfg,
        paddle_mat=paddle_mat, paddle_npz=paddle_npz,
        wholebody_mat=args.wholebody_mat, wholebody_npz=wholebody_npz,
        source_idx=args.source_idx, out_path=args.out, dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
