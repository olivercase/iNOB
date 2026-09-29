#!/usr/bin/env python3
"""Is the EEG unit factor 1, whatever the sphere?

`inob calibrate` checks DUNEuro's mm-mode electrode output against the
Berg-Scherg series on one homogeneous sphere. A units factor has to be the
same for every sphere, so this runs that check across radius, source depth,
conductivity and mesh pitch and reports the spread. What is left after the
unit conversion is the finite-element discretisation error of a coarse
sphere mesh, and it should shrink as the pitch does.

    python3 scripts/eeg_calibration_sweep.py
    python3 scripts/eeg_calibration_sweep.py --radius-mm 100 --depth-mm 50 --pitch-mm 3 2

Needs duneuropy. Each case writes its meshes and arrays under --out-dir.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--radius-mm", type=float, nargs="+", default=[80.0, 100.0])
    ap.add_argument("--depth-mm", type=float, nargs="+", default=[30.0, 50.0, 70.0])
    ap.add_argument("--sigma", type=float, nargs="+", default=[0.2, 0.43])
    ap.add_argument("--pitch-mm", type=float, nargs="+", default=[3.0])
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/calibration/sweep")
    args = ap.parse_args(argv)

    from inob.analysis.sphere_calibration import calibrate_eeg_factor

    rows = []
    for R, depth, sigma, pitch in itertools.product(
        args.radius_mm, args.depth_mm, args.sigma, args.pitch_mm
    ):
        if depth >= R:
            continue
        s = calibrate_eeg_factor(
            radius_mm=R,
            pitch_mm=pitch,
            sigma_S_per_m=sigma,
            source_radius_mm=R - depth,
            out_dir=args.out_dir / f"R{R:g}_d{depth:g}_s{sigma:g}_p{pitch:g}",
        )
        rows.append(asdict(s))
        print(
            f"R {R:>5g} mm  depth {depth:>4g} mm  sigma {sigma:>4g} S/m  pitch {pitch:g} mm  "
            f"tets {s.n_tets:>6d}  median {s.factor_median:.3f}  lstsq {s.factor_lsq:.3f}",
            flush=True,
        )
    med = np.array([r["factor_median"] for r in rows])
    lsq = np.array([r["factor_lsq"] for r in rows])
    print(
        f"\n{len(rows)} cases: median of medians {np.median(med):.3f} "
        f"(CV {med.std() / med.mean():.1%}), median of lstsq {np.median(lsq):.3f} "
        f"(CV {lsq.std() / lsq.mean():.1%}). A pure unit factor would be 1 in every case."
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sweep.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
