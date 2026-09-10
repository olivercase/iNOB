"""CLI: moving-dipole physiology simulation.

Which scenarios run is decided by the target's physiology profile
(:mod:`inob.physiology.profiles`), selected with ``--source-target``:

    inob-physiology --source-target vagus   # baroreceptor + deep breathing
    inob-physiology --source-target spine   # median- + tibial-nerve SSEP

The per-scenario knobs below apply to whichever scenarios the target defines;
irrelevant ones are ignored.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from inob.cli._common import add_common_args, setup
from inob.config import source_target_tag
from inob.physiology.profiles import profile_for_tag
from inob.viz.physiology_plot import render_physiology

logger = logging.getLogger(__name__)


# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Simulate physiological activity as a dipole moving along the nerve.

Which scenarios run is decided by the target's physiology profile, chosen
with --source-target:

  inob physiology --source-target vagus   baroreceptor + deep breathing
  inob physiology --source-target spine   median- and tibial-nerve SSEP

The per-scenario options below apply to whichever scenarios the target
defines; the rest are ignored.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob physiology",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob physiology                                the target's own scenarios
  inob physiology --hr-bpm 50                    slower heart, fewer bursts
  inob physiology --source-target spine          SSEP volleys instead
  inob physiology --median-rate-hz 3             a faster stimulus train

Time-domain traces at every sensor, from the target's physiology profile.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    # vagus scenarios
    p.add_argument("--hr-bpm", type=float, default=None,
                   help="[vagus] Heart rate for the baroreceptor scenario.")
    p.add_argument("--breath-bpm", type=float, default=None,
                   help="[vagus] Respiratory rate for the deep-breathing scenario.")
    p.add_argument("--baro-duration-s", type=float, default=None)
    p.add_argument("--resp-duration-s", type=float, default=None)
    p.add_argument("--n-fibres-baro", type=int, default=None,
                   help="[vagus] Fibres per cardiac burst (baroreceptor).")
    p.add_argument("--n-fibres-rar", type=int, default=None,
                   help="[vagus] RAR phasic burst fibres at inspiration onset.")
    p.add_argument("--n-fibres-sar", type=int, default=None,
                   help="[vagus] SAR tonic-train fibres.")
    # spine scenarios
    p.add_argument("--median-rate-hz", type=float, default=None,
                   help="[spine] Median-nerve stimulation rate (default 4.7 Hz).")
    p.add_argument("--tibial-rate-hz", type=float, default=None,
                   help="[spine] Tibial-nerve stimulation rate (default 3.1 Hz).")
    p.add_argument("--median-duration-s", type=float, default=None)
    p.add_argument("--tibial-duration-s", type=float, default=None)
    p.add_argument("--n-fibres-median", type=int, default=None,
                   help="[spine] Afferents in the median-nerve volley.")
    p.add_argument("--n-fibres-tibial", type=int, default=None,
                   help="[spine] Afferents in the tibial-nerve volley.")
    p.add_argument("--fs-hz", type=float, default=30_000.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="physiology")

    tag = source_target_tag(cfg)
    profile = profile_for_tag(tag)
    logger.info("[physiology] profile:\n%s", profile.describe())
    if not profile.validated:
        logger.warning(
            "[physiology] PHYSIOLOGY-TODO: '%s' has no validated dynamics (%s); "
            "output is PROVISIONAL.", profile.label, profile.paradigm,
        )

    # Only forward the knobs the user actually set, so each scenario keeps its
    # own documented defaults.
    kw = {
        "baro_hr_bpm": args.hr_bpm,
        "baro_duration_s": args.baro_duration_s,
        "baro_n_fibres_per_burst": args.n_fibres_baro,
        "resp_breath_bpm": args.breath_bpm,
        "resp_duration_s": args.resp_duration_s,
        "resp_n_phasic_fibres_RAR": args.n_fibres_rar,
        "resp_n_tonic_fibres_SAR": args.n_fibres_sar,
        "median_rate_hz": args.median_rate_hz,
        "median_duration_s": args.median_duration_s,
        "median_n_fibres": args.n_fibres_median,
        "tibial_rate_hz": args.tibial_rate_hz,
        "tibial_duration_s": args.tibial_duration_s,
        "tibial_n_fibres": args.n_fibres_tibial,
    }
    kw = {k: v for k, v in kw.items() if v is not None}

    try:
        scenarios = profile.scenarios(**kw)
    except NotImplementedError as e:
        logger.error("%s", e)
        return 2
    except TypeError as e:
        logger.error("scenario parameter not accepted by the %s profile: %s",
                     profile.label, e)
        return 2

    render_physiology(
        cfg, scenarios=scenarios, fs_hz=args.fs_hz,
        out_path=args.out, dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
