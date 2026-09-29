#!/usr/bin/env python3
"""Every number the worked examples quote, from the leadfields on disk.

`paper_stats.py` measures what the manuscript says about the package; this
measures what it says about the physics. Each figure in the vagus, spinal
cord and muscle examples is computed here by the same package function the
corresponding command uses, from the solved leadfields in `outputs/forward/`,
so a revision can paste the output instead of carrying numbers forward from
an older solve. Run it after any re-solve:

    python3 scripts/paper_numbers.py            # all three targets
    python3 scripts/paper_numbers.py --target vagus

Needs, per target, `duneuro_leadfield_<t>.npz`, `duneuro_eeg_leadfield_<t>.npz`
and (for the coverage comparison) `duneuro_eeg_leadfield_wholebody_<t>.npz`,
all from one FEM and one source list. Sarvas comparisons read the FEM and the
sensor array as well. No DUNEuro needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Source strengths the examples quote, nA·m.
SPINE_Q = 5.11
VAGUS_SWEEP_Q = (0.74, 1.0, 5.11, 10.0, 20.0, 70.0)
MUSCLE_Q = 10.0


def _n(signal: float, sigma: float) -> int:
    """Averages to reach SNR 3, as a whole number of trials (at least one)."""
    from inob.viz.detectability import required_trials

    return math.ceil(required_trials(signal, sigma) - 1e-9)


def _load(cfg, which: str):
    from inob.io.npz import load_leadfield

    fwd = cfg.outputs.forward_npz.parent
    tag = cfg.outputs.forward_npz.stem.removeprefix("duneuro_leadfield_")
    path = {
        "meg": cfg.outputs.forward_npz,
        "eeg": cfg.outputs.forward_eeg_npz,
        "wholebody": fwd / f"duneuro_eeg_leadfield_wholebody_{tag}.npz",
    }[which]
    return load_leadfield(path) if path.exists() else None


def _sarvas(cfg) -> dict:
    from inob.analysis.sarvas_compare import compare_sarvas_vs_fem, save_comparison_summary

    with tempfile.TemporaryDirectory() as tmp:
        p = save_comparison_summary(compare_sarvas_vs_fem(cfg), Path(tmp) / "s.json")
        d = json.loads(Path(p).read_text())
    lo, hi = d["ratio_fem_to_sarvas_peak_band_ci95"]
    return {
        "fem_over_sarvas": d["ratio_fem_to_sarvas_peak_band"],
        "ci95": (lo, hi),
        "sarvas_pT_at_70": d["sarvas_peak_pT_at_Q70_band"],
        "bootstrap_n": d["ratio_fem_to_sarvas_peak_band_bootstrap_n"],
    }


def _arc_step(pos: np.ndarray) -> tuple[float, float]:
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return float(d.mean()), float(d.sum())


def section_noise(base) -> None:
    from inob.analysis.snr import compute_noise_floors
    from inob.cli._common import apply_source_target
    from inob.physiology.profiles import profile_for

    print("== Noise floors (per target: the OPM floor depends on the CAP width)")
    for t in ("vagus", "spine", "muscle"):
        cfg = apply_source_target(base, t)
        f = compute_noise_floors(cfg)
        n = cfg.noise
        print(
            f"  {t:<7} AP {profile_for(cfg).ap_width_ms:.1f} ms -> f_CAP {f.signal_hz:.0f} Hz, "
            f"gain {f.meg_sensor_gain:.3f}; OPM {n.opm_intrinsic_fT_sqrtHz:g} fT/rtHz x "
            f"sqrt({f.meg_noise_bandwidth_hz:.0f} Hz) = "
            f"{n.opm_intrinsic_fT_sqrtHz * math.sqrt(f.meg_noise_bandwidth_hz):.1f} fT "
            f"-> effective {f.meg_per_channel_fT:.1f} fT;  electrode {f.eeg_per_channel_uV:.2f} uV"
        )


def section_spine(base) -> None:
    from inob.analysis.snr import compute_noise_floors, per_source_peak
    from inob.analysis.source_models import compare_source_models
    from inob.anatomy import level_suffix
    from inob.cli._common import apply_source_target
    from inob.physiology.profiles import profile_for
    from inob.viz.detectability import default_source_idx

    cfg = apply_source_target(base, "spine")
    meg, eeg, wb = _load(cfg, "meg"), _load(cfg, "eeg"), _load(cfg, "wholebody")
    f = compute_noise_floors(cfg)
    prof = profile_for(cfg)
    idx = default_source_idx(meg, eeg)
    z = meg.source_pos[idx, 2]
    step, span = _arc_step(meg.source_pos)
    print("\n== Spinal cord")
    print(
        f"  {len(meg.source_pos)} sources, mean step {step:.1f} mm along {span:.0f} mm; "
        f"default source #{idx}, z = {z:.0f} mm{level_suffix(cfg.data.bone_dir, z)}"
    )
    s = _sarvas(cfg)
    print(
        f"  FEM/Sarvas peak {s['fem_over_sarvas']:.2f} "
        f"(95% CI {s['ci95'][0]:.2f}-{s['ci95'][1]:.2f}, B = {s['bootstrap_n']})"
    )
    for name, e in (("32-contact patch", eeg), ("whole-body array", wb)):
        if e is None:
            print(f"  [{name}: no leadfield]")
            continue
        r = compare_source_models(
            meg,
            e,
            prof,
            Q_nAm=SPINE_Q,
            source_idx=idx,
            meg_sigma_fT=f.meg_per_channel_fT,
            eeg_sigma_uV=f.eeg_per_channel_uV,
        )
        print(
            f"  {name} ({len(e.coil_pos)} contacts), Q = {SPINE_Q} nA.m, span {r.span_mm:.0f} mm, "
            f"transit {r.transit_ms:.1f} ms:"
        )
        for row in r.rows:
            print(
                f"    {row.name:<12} OPM {row.meg_fT:8.1f} fT / {_n(row.meg_fT, f.meg_per_channel_fT):>5} "
                f"trials   electrodes {row.eeg_uV:6.3f} uV / "
                f"{_n(row.eeg_uV, f.eeg_per_channel_uV):>6} trials"
            )
        print(
            f"    synchronous / ascending OPM amplitude: {r.rows[0].meg_fT / r.rows[2].meg_fT:.1f}x"
        )
        asc = r.rows[2]
        print(
            f"    ascending: electrode trials / OPM trials = "
            f"{_n(asc.eeg_uV, f.eeg_per_channel_uV) / _n(asc.meg_fT, f.meg_per_channel_fT):.1f}x"
        )
    if wb is not None:
        ratio = per_source_peak(wb.L_fT_per_nAm) / per_source_peak(eeg.L_fT_per_nAm)
        print(
            f"  whole-body / patch single-contact peak over the cord: median "
            f"{np.median(ratio):.1f}x, range {ratio.min():.1f}-{ratio.max():.0f}x"
        )


def section_vagus(base) -> None:
    from inob.analysis.propagation import compute_propagation_signals
    from inob.analysis.snr import compute_noise_floors, per_source_best_bipolar, per_source_peak
    from inob.anatomy import level_suffix
    from inob.cli._common import apply_source_target
    from inob.physiology.profiles import profile_for
    from inob.sensors.layout import radial_channels
    from inob.viz.detectability import default_source_idx, matching_eeg_source_idx

    cfg = apply_source_target(base, "vagus")
    meg, eeg, wb = _load(cfg, "meg"), _load(cfg, "eeg"), _load(cfg, "wholebody")
    f = compute_noise_floors(cfg)
    sm, se = f.meg_per_channel_fT, f.eeg_per_channel_uV
    pos = meg.source_pos
    step, span = _arc_step(pos)
    print("\n== Cervical vagus")
    print(
        f"  {len(pos)} sources, z = {pos[:, 2].min():.0f}-{pos[:, 2].max():.0f} mm, "
        f"mean step {step:.1f} mm along {span:.0f} mm"
    )
    s = _sarvas(cfg)
    print(
        f"  FEM/Sarvas peak {s['fem_over_sarvas']:.2f} (95% CI {s['ci95'][0]:.2f}-"
        f"{s['ci95'][1]:.2f}); Sarvas peak at 70 nA.m {s['sarvas_pT_at_70']:.2f} pT"
    )
    pk = per_source_peak(meg.L_fT_per_nAm)
    print(
        f"  OPM coupling per source: best {pk.max():.1f}, median {np.median(pk):.1f}, "
        f"worst {pk.min():.1f} fT/nA.m ({pk.max() / pk.min():.0f}-fold)"
    )
    n5 = np.array([_n(p * 5.0, sm) for p in pk])
    hi = pos[:, 2] > 1300
    print(
        f"  at 5 nA.m: {n5[hi].min()}-{n5[hi].max()} trials above z = 1300 mm, "
        f"{n5.min()}-{n5.max()} over the whole nerve"
    )
    idx = default_source_idx(meg, eeg)
    j = matching_eeg_source_idx(meg, eeg, idx)
    z = pos[idx, 2]
    pm = pk[idx]
    bb = per_source_best_bipolar(eeg.L_fT_per_nAm)[j]
    ep = per_source_peak(eeg.L_fT_per_nAm)[j]
    print(
        f"  patch-centred source #{idx}, z = {z:.0f} mm"
        f"{level_suffix(cfg.data.bone_dir, z)}: OPM {pm:.1f} fT/nA.m; "
        f"patch peak {ep:.3f}, best pair {bb:.3f} uV/nA.m"
    )
    for q in VAGUS_SWEEP_Q:
        nm, ne = _n(pm * q, sm), _n(bb * q, se)
        print(
            f"    Q {q:>5g}: OPM {pm * q:8.1f} fT SNR {pm * q / sm:6.2f} -> {nm:>5}   "
            f"electrodes {bb * q:6.3f} uV SNR {bb * q / se:6.3f} -> {ne:>6}"
        )
    print(f"    trials gap (electrodes / OPM), exact in Q: {(pm / sm) ** 2 / (bb / se) ** 2:.1f}x")
    if wb is not None:
        k = matching_eeg_source_idx(meg, wb, idx)
        bw = per_source_best_bipolar(wb.L_fT_per_nAm)
        print(
            f"  whole-body array best pair: {bw[k] / bb:.2f}x the patch at this source, "
            f"{bw.max() / per_source_best_bipolar(eeg.L_fT_per_nAm).max():.2f}x at the best source"
        )
    sig = compute_propagation_signals(meg, profile_for(cfg))
    radial = np.zeros(len(meg.channel_names), dtype=bool)
    radial[radial_channels(meg.channel_names)] = True
    c = int(np.argmax(np.sqrt(np.mean(sig.stationary**2, axis=1)) * radial))
    st, sg, t = sig.stationary[c] * 1e15, sig.segment[c] * 1e15, sig.t_ms

    def fwhm(x):
        a = np.where(np.abs(x) >= 0.5 * np.abs(x).max())[0]
        return float(t[a[-1]] - t[a[0]])

    print(
        f"  propagation: {sig.n_fibres}-fibre event, segment {sig.segment_mm:.0f} mm of "
        f"{sig.total_span_mm:.0f} mm, CV {sig.cv_mean_m_per_s:.1f} m/s, transit "
        f"{sig.transit_segment_ms:.2f} ms vs AP {sig.ap_width_ms:.2f} ms; "
        f"Q {sig.Q_total_nAm:.2f} nA.m"
    )
    print(
        f"    travelling {np.abs(sg).max():.2f} fT vs stationary {np.abs(st).max():.2f} fT "
        f"(ratio {np.abs(sg).max() / np.abs(st).max():.3f}); FWHM {fwhm(sg):.2f} vs {fwhm(st):.2f} ms"
    )


def section_muscle(base) -> None:
    from inob.analysis.snr import compute_noise_floors, per_source_best_bipolar, per_source_peak
    from inob.anatomy import level_suffix
    from inob.cli._common import apply_source_target
    from inob.viz.detectability import default_source_idx, matching_eeg_source_idx

    cfg = apply_source_target(base, "muscle")
    meg, eeg = _load(cfg, "meg"), _load(cfg, "eeg")
    f = compute_noise_floors(cfg)
    idx = default_source_idx(meg, eeg)
    j = matching_eeg_source_idx(meg, eeg, idx)
    z = meg.source_pos[idx, 2]
    pm = per_source_peak(meg.L_fT_per_nAm)[idx] * MUSCLE_Q
    bb = per_source_best_bipolar(eeg.L_fT_per_nAm)[j] * MUSCLE_Q
    print("\n== Neck muscle")
    print(
        f"  {len(meg.source_pos)} MG sources ({len(eeg.source_pos)} electrode); default #{idx}, "
        f"z = {z:.0f} mm{level_suffix(cfg.data.bone_dir, z)}"
    )
    print(
        f"  Q = {MUSCLE_Q:g} nA.m: OPM {pm:.0f} fT, SNR {pm / f.meg_per_channel_fT:.1f} "
        f"(floor {f.meg_per_channel_fT:.0f} fT);  best pair {bb:.1f} uV, "
        f"SNR {bb / f.eeg_per_channel_uV:.1f}"
    )


def section_placement(base) -> None:
    from inob.analysis.placement import compare_placement, nearest_array_info, optimise_array
    from inob.analysis.snr import compute_noise_floors, per_source_peak
    from inob.cli._common import apply_source_target
    from inob.physiology.profiles import profile_for

    print("\n== Placement (20 sensors, 30 mm apart)")
    for t in ("vagus", "spine", "muscle"):
        cfg = apply_source_target(base, t)
        meg = _load(cfg, "meg")
        L = meg.L_fT_per_nAm
        noise = compute_noise_floors(cfg).meg_per_channel_fT
        r = compare_placement(
            L,
            meg.coil_pos,
            meg.source_pos,
            source_idx=int(np.argmax(per_source_peak(L))),
            n_sensors=20,
            noise_floor=noise,
        )
        q = profile_for(cfg).event_moment_nAm
        opt = optimise_array(L, meg.coil_pos, n_sensors=20, noise_floor=noise, moment_nAm=q)
        near = nearest_array_info(
            L, meg.coil_pos, meg.source_pos, n_sensors=20, noise_floor=noise, moment_nAm=q
        )
        print(
            f"  {t:<7} one source: model-chosen / nearest amplitude {r.amplitude_ratio:.3f}; "
            f"whole structure: optimised / nearest information {opt.info_bits[-1] / near:.2f}"
        )


SECTIONS = {
    "noise": section_noise,
    "spine": section_spine,
    "vagus": section_vagus,
    "muscle": section_muscle,
    "placement": section_placement,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--target", choices=list(SECTIONS), action="append")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)

    from inob.config import load_config

    base = load_config(args.config)
    for name in args.target or list(SECTIONS):
        SECTIONS[name](base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
