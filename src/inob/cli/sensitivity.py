"""CLI: conductivity sensitivity sweep (MEG vs EEG) + comparison figure.

For each (tissue, factor) in ``cfg.sensitivity`` the chosen modality's
leadfield is recomputed with that tissue's conductivity scaled, and the
per-channel relative change vs the unperturbed baseline is summarised to
``cfg.outputs.sensitivity_dir/sensitivity_<modality>.json``.

The headline paper claim — that the magnetic forward is far less sensitive
than the electric one to bone/skin conductivity uncertainty — is rendered as
a grouped bar figure by ``--plot`` (or ``--plot-only`` to skip the solves and
draw from existing JSON).

Recomputing a leadfield is the expensive step; MEG transfer-matrix solves on
the full array are slow, so the EEG sweep is the cheap one. Use ``--modality``
to run them independently (e.g. EEG locally, MEG on the cluster).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from inob.analysis.sensitivity import sweep
from inob.cli._common import add_common_args, setup
from inob.config import Config
from inob.io.npz import load_leadfield

logger = logging.getLogger(__name__)


def _meg_forward_fn(cfg: Config, out_npz: Path) -> np.ndarray:
    from inob.forward.solve import run_forward

    cfg2 = replace(cfg, outputs=replace(cfg.outputs, forward_npz=out_npz))
    run_forward(cfg2)
    return load_leadfield(out_npz).L


def _eeg_forward_fn(cfg: Config, out_npz: Path) -> np.ndarray:
    from inob.forward.eeg import run_eeg_forward

    cfg2 = replace(cfg, outputs=replace(cfg.outputs, forward_eeg_npz=out_npz))
    run_eeg_forward(cfg2)
    return load_leadfield(out_npz).L


_MODALITIES = {
    "meg": (_meg_forward_fn, "forward_npz"),
    "eeg": (_eeg_forward_fn, "forward_eeg_npz"),
}


def _run_modality(cfg: Config, modality: str, *, skip_unit: bool) -> Path:
    """Run the sweep for one modality; return its JSON path."""
    forward_fn, baseline_attr = _MODALITIES[modality]
    baseline_path = getattr(cfg.outputs, baseline_attr)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"{modality} baseline leadfield missing: {baseline_path}. "
            f"Run the {modality} forward solve first."
        )
    # Factor 1.0 reproduces the baseline (relative change ≡ 0) — skip the
    # expensive no-op solve unless explicitly asked to keep it.
    cfg_run = cfg
    if skip_unit:
        kept = tuple(f for f in cfg.sensitivity.perturbations if f != 1.0)
        cfg_run = replace(cfg, sensitivity=replace(cfg.sensitivity, perturbations=kept))
    out_path = cfg.outputs.sensitivity_dir / f"sensitivity_{modality}.json"
    sweep(
        cfg_run,
        baseline_path=baseline_path,
        forward_fn=forward_fn,
        out_path=out_path,
        modality=modality,
    )
    return out_path


# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Measure how much the answer moves when a tissue conductivity is
wrong.

Each tissue in turn has its conductivity scaled by the factors in the
config, the leadfield is re-solved, and the per-channel change against the
unperturbed baseline is summarised to JSON. --plot draws the comparison;
--plot-only redraws from JSON already on disk.

Re-solving is the expensive part, and MEG on the full array is much slower
than EEG. Use --modality to run the two independently — EEG locally, MEG on
the cluster.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob sensitivity",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob sensitivity                       sweep both modalities and plot
  inob sensitivity --modality eeg        electric only (where sigma matters)
  inob sensitivity --plot-only           redraw from results already on disk

Expensive: each perturbation is a fresh forward solve. Consider the cluster.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument(
        "--modality",
        choices=["meg", "eeg", "both"],
        default="both",
        help="Which modality(ies) to sweep (default: both).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Render the MEG-vs-EEG comparison figure after the sweep.",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip the solves; render the figure from existing JSON.",
    )
    p.add_argument(
        "--keep-unit",
        action="store_true",
        help="Also run the factor=1.0 perturbation (a no-op; off by default).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Figure output PNG (default: sensitivity_dir/sensitivity_comparison.png).",
    )
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="sensitivity")

    modalities = ["meg", "eeg"] if args.modality == "both" else [args.modality]

    if not args.plot_only:
        for m in modalities:
            out = _run_modality(cfg, m, skip_unit=not args.keep_unit)
            logger.info("[sensitivity:%s] wrote %s", m, out)

    if args.plot or args.plot_only:
        from inob.viz.sensitivity_plot import render_sensitivity

        out_png = args.out or (cfg.outputs.sensitivity_dir / "sensitivity_comparison.png")
        render_sensitivity(cfg, out_path=out_png, dpi=args.dpi)
        logger.info("[sensitivity] figure → %s", out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
