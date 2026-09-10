"""CLI: DUNEuro sphere calibration and validation.

Two jobs on the same homogeneous-sphere FEM:

  * default — the empirical mm-mode → µV/(nA·m) EEG factor (Berg–Scherg).
  * ``--meg`` — validate the MEG forward against the Sarvas analytic field,
    reporting RDM (topography) and MAG (magnitude).

``--compare-source-models`` runs the MEG validation once per source model on
the *same* mesh and analytic reference. That is the honest way to decide
whether to move a production solve off partial integration: same geometry,
same reference, only the right-hand side differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inob.analysis.sphere_calibration import calibrate_eeg_factor, validate_meg_sphere
from inob.cli._common import add_common_args, setup
from inob.config import SOURCE_MODEL_TYPES, replace_source_model
from inob.forward.duneuro_driver import build_source_model_config

# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Calibrate and validate the forward solve on a homogeneous sphere.

Two jobs on the same sphere FEM:

  default   the empirical mm-mode to µV/(nA·m) EEG scale factor
            (Berg–Scherg).
  --meg     validate the MEG forward against the Sarvas analytic field,
            reporting RDM (topography) and MAG (magnitude).

--compare-source-models runs the MEG validation once per source model on
the same mesh and the same analytic reference — the honest way to decide
whether to move a production solve off partial integration, since only the
right-hand side differs.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob calibrate",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob calibrate                          EEG mm-mode to SI, from a sphere FEM
  inob calibrate --meg                    the MEG sphere validation instead
  inob calibrate --radius-mm 80           a smaller sphere
  inob calibrate --compare-source-models  partial integration vs St. Venant

Determines the unit factor empirically rather than by dimensional analysis.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument(
        "--radius-mm",
        type=float,
        default=100.0,
        help="Sphere radius (default 100 mm — neck-scale).",
    )
    p.add_argument(
        "--pitch-mm", type=float, default=3.0, help="Voxelisation pitch for the sphere FEM."
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=0.43,
        help="Homogeneous conductivity in S/m (default 0.43 = skin).",
    )
    p.add_argument(
        "--source-radius-mm",
        type=float,
        default=50.0,
        help="Source eccentricity along Z (mm; must be < radius).",
    )
    p.add_argument("--n-electrodes", type=int, default=200)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/calibration"))
    p.add_argument(
        "--meg",
        action="store_true",
        help="Validate the MEG forward against the Sarvas analytic sphere "
        "instead of calibrating the EEG factor. Reports RDM (topography) "
        "and MAG (magnitude); both should be within a few percent.",
    )
    p.add_argument(
        "--compare-source-models",
        nargs="*",
        metavar="MODEL",
        default=None,
        choices=list(SOURCE_MODEL_TYPES),
        help="Run the MEG sphere validation once per source model and print a "
        "table. With no values, compares every model this pipeline "
        "supports. Implies --meg.",
    )
    p.add_argument(
        "--coil-radius-mm",
        type=float,
        default=120.0,
        help="MEG only: radius the magnetometers sit at.",
    )
    p.add_argument(
        "--n-coils", type=int, default=60, help="MEG only: number of radial magnetometers."
    )
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="calibrate")

    if args.compare_source_models is not None:
        models = args.compare_source_models or list(SOURCE_MODEL_TYPES)
        return _compare_source_models(cfg, args, models)
    if args.meg:
        return _validate_meg(cfg, args)
    summary = calibrate_eeg_factor(
        radius_mm=args.radius_mm,
        pitch_mm=args.pitch_mm,
        sigma_S_per_m=args.sigma,
        source_radius_mm=args.source_radius_mm,
        n_electrodes=args.n_electrodes,
        duneuro_path=cfg.forward.duneuro_path,
        out_dir=args.out_dir,
    )
    sys.stdout.write(
        f"\n[calibration] empirical EEG factor (raw → µV/nA·m):\n"
        f"  peak    : {summary.factor_peak:.4e}\n"
        f"  median  : {summary.factor_median:.4e}\n"
        f"  geomean : {summary.factor_geomean:.4e}\n"
        f"  (n_elec={summary.n_electrodes}, R={summary.radius_mm:.0f} mm,\n"
        f"   src_r={summary.source_radius_mm:.0f} mm, σ={summary.sigma_S_per_m} S/m,\n"
        f"   n_tets={summary.n_tets})\n"
        f"\nApply to forward/eeg.py: L_uV_per_nAm = L * {summary.factor_median:.3e}\n"
    )
    return 0


def _meg_kwargs(cfg, args) -> dict:
    """Sphere geometry shared by every MEG run here, so an A/B is like-for-like."""
    return dict(
        radius_mm=args.radius_mm,
        pitch_mm=args.pitch_mm,
        coil_radius_mm=args.coil_radius_mm,
        source_radius_mm=args.source_radius_mm,
        sigma_S_per_m=args.sigma,
        n_coils=args.n_coils,
        duneuro_path=cfg.forward.duneuro_path,
        out_dir=args.out_dir,
    )


def _validate_meg(cfg, args) -> int:
    v = validate_meg_sphere(
        **_meg_kwargs(cfg, args),
        source_model=build_source_model_config(cfg),
    )
    sys.stdout.write(
        f"\n[MEG sphere] source model: {cfg.forward.source_model.type}\n"
        f"  RDM (topography, 0 = perfect) : {v.rdm:.4f}\n"
        f"  MAG (magnitude, 1 = perfect)  : {v.mag:.4f}\n"
        f"  FEM peak                      : {v.fem_peak_fT_per_nAm:.4f} fT/nAm\n"
        f"  Sarvas peak                   : {v.sarvas_peak_fT_per_nAm:.4f} fT/nAm\n"
        f"  (n_coils={v.n_coils}, n_tets={v.n_tets}, R={v.radius_mm:.0f} mm)\n"
    )
    return 0


def _compare_source_models(cfg, args, models: list[str]) -> int:
    """Same sphere, same analytic reference, one row per source model."""
    rows: list[tuple[str, float, float]] = []
    for model in models:
        model_cfg = replace_source_model(cfg, type=model)
        v = validate_meg_sphere(
            **_meg_kwargs(cfg, args),
            source_model=build_source_model_config(model_cfg),
        )
        rows.append((model, v.rdm, v.mag))
        sys.stdout.write(f"  {model}: RDM {v.rdm:.4f}  MAG {v.mag:.4f}\n")
        sys.stdout.flush()

    width = max(len(m) for m, _, _ in rows)
    sys.stdout.write(
        f"\n[MEG sphere] source models against Sarvas "
        f"(R={args.radius_mm:.0f} mm, src_r={args.source_radius_mm:.0f} mm)\n"
        f"  {'model'.ljust(width)}  {'RDM':>8}  {'MAG':>8}\n"
    )
    for model, rdm, mag in rows:
        sys.stdout.write(f"  {model.ljust(width)}  {rdm:8.4f}  {mag:8.4f}\n")
    best = min(rows, key=lambda r: r[1])[0]
    sys.stdout.write(
        f"\nLowest RDM: {best}\n"
        "A homogeneous sphere has no thin compartment, so this validates the\n"
        "machinery, not the hard case — check a real mesh before switching a\n"
        "production solve.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
