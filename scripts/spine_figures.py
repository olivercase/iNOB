#!/usr/bin/env python3
"""The spinal-cord figures of the paper, from the leadfields already on disk.

The worked example in the paper asks four design questions of one target, the
cervical spinal cord. Two of the figures come straight from package commands
(`inob torso` for the field on the body, `inob source-models` for the three
source models); the other two compare several runs side by side, which no
single command does, so they are drawn here:

    forward_models   which forward model is needed: free-space Biot-Savart,
                     the single sphere (Sarvas) and the full FEM, along the cord
    sensor_design    which sensors and where: trials-to-detect along the cord
                     for each array, the sensor noise level, and how many
                     sensors an array needs

    python3 scripts/spine_figures.py                  # both, to outputs/paper/
    python3 scripts/spine_figures.py --only sensor_design
    python3 scripts/spine_figures.py --sweep-json outputs/paper/standoff.json

Needs `duneuro_leadfield_spine.npz`, `duneuro_eeg_leadfield_spine.npz` and
`duneuro_eeg_leadfield_wholebody_spine.npz` (see docs/REPRODUCE.md). No DUNEuro:
everything is read from those files. Every number printed is one the paper
quotes, computed by the same package functions `scripts/paper_numbers.py` uses.

The sensor stand-off needs a fresh DUNEuro solve per distance, so it is not
recomputed here. Run `python3 scripts/sensor_sweep.py --json FILE` on a machine
with DUNEuro and pass FILE as --sweep-json to add it as a fourth panel.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The cord's source strength (one stationary segment, nA·m); see the paper.
SPINE_Q = 5.11
#: Clinical somatosensory-evoked-potential averaging budget, trials.
CLINICAL_BUDGET = (500, 2000)
#: OPM noise settings compared in the noise-level panel, as (overrides, label).
#: Both QuSpin settings are manufacturer maxima: Neuro-1 triaxial (23) and
#: dual-axis (15 fT/√Hz).
OPM_PRESETS = (
    (["noise.opm_sensor=quspin_qzfm3"], "QuSpin\n3-axis\n23"),
    (["noise.opm_sensor=quspin_qzfm2"], "QuSpin\n2-axis\n15"),
    (["noise.opm_sensor=fieldline_v3"], "Field-\nLine v3\n15"),
    (["noise.opm_sensor=he4_wideband"], "He-4\n\n30"),
)
#: Alternative magnetometer for the same leadfield: the SQUID magnetoneurograph of the Japanese
#: magnetospinography studies (Sumiya 2017; Akaza 2021; Hashimoto 2022): under 4 fT/sqrt(Hz) with
#: a flat response over a 100-5,000 Hz recording band, so no roll-off. Effective floor =
#: 4 * sqrt(5000 * (atan(5000/5000) - atan(100/5000))) / gain(227 Hz) = 247 fT (paper Table 5).
SQUID_FLOOR_FT = 247.0
ARRAY_SIZES = (4, 8, 16, 32, 64)


def _trials(signal, sigma):
    """Averages to reach SNR 3, whole trials, at least one (vectorised)."""
    from inob.viz.detectability import required_trials

    n = np.vectorize(lambda s: required_trials(float(s), sigma))(np.atleast_1d(signal))
    return np.ceil(n - 1e-9)


def _levels(cfg, z_mm: np.ndarray) -> list[tuple[float, str]]:
    """(z, label) for each vertebral level the cord passes, for a top axis."""
    from inob.anatomy import vertebral_level

    out: dict[str, float] = {}
    for z in np.sort(z_mm):
        lv = vertebral_level(cfg.data.bone_dir, float(z))
        if lv is not None:
            out.setdefault(lv.short.split("/")[0], float(z))
    return [(z, name) for name, z in out.items()]


def _mark_levels(ax, levels, keep=("C1", "C4", "C7", "T4", "T8", "T12")) -> None:
    """Vertebral landmarks along the z axis, so 'z = 1393 mm' reads as C6."""
    for z, name in levels:
        short = name
        if short in keep:
            ax.axvline(z, color="#D8D8D8", lw=0.5, zorder=0)
            ax.text(
                z,
                1.0,
                short,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#6A6A6A",
            )


def forward_models(cfg, out: Path) -> None:
    """Biot-Savart vs Sarvas vs FEM, per source along the cord and per coil."""
    import matplotlib.pyplot as plt

    from inob.analysis.sarvas_compare import compare_sarvas_vs_fem, save_comparison_summary
    from inob.io.npz import load_leadfield
    from inob.viz.detectability import default_source_idx
    from inob.viz.style import NATURE_PALETTE as P
    from inob.viz.style import add_panel_label, apply_nature_style, save_figure

    r = compare_sarvas_vs_fem(cfg, Q_nAm=1.0)
    summary_path = out.with_suffix(".json")
    save_comparison_summary(r, summary_path)
    summary = json.loads(summary_path.read_text())
    band = summary["ratio_fem_to_sarvas_peak_band"]
    lo, hi = summary["ratio_fem_to_sarvas_peak_band_ci95"]

    # Tesla per nA·m -> fT per nA·m. Everything below is per unit moment.
    fem = np.abs(r.fem_T) * 1e15
    sar = np.abs(r.sarvas_T) * 1e15
    bio = np.abs(r.biot_T) * 1e15 if r.biot_T is not None else None
    z = r.source_pos_mm[:, 2]
    order = np.argsort(z)
    meg = load_leadfield(cfg.outputs.forward_npz)
    eeg = load_leadfield(cfg.outputs.forward_eeg_npz)
    idx = default_source_idx(meg, eeg)

    apply_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.3, 1, 1.3]})
    levels = _levels(cfg, z)

    ax = axes[0]
    if bio is not None:
        ax.plot(
            z[order],
            bio.max(axis=0)[order],
            color=P["grey"],
            lw=1.0,
            ls=":",
            label="free space (Biot–Savart)",
        )
    ax.plot(
        z[order], sar.max(axis=0)[order], color=P["blue"], lw=1.2, label="single sphere (Sarvas)"
    )
    ax.plot(z[order], fem.max(axis=0)[order], color=P["red"], lw=1.2, label="full FEM")
    ax.scatter(
        [z[idx]], [fem[:, idx].max()], s=14, color=P["glow"], zorder=5, edgecolor="k", lw=0.4
    )
    ax.set_yscale("log")
    ax.set_xlabel("source height z (mm)")
    ax.set_ylabel("best radial coil (fT per nA·m)")
    ax.set_ylim(0.03, 60)
    ax.legend(loc="lower left", fontsize=6, ncol=1)
    _mark_levels(ax, levels)
    add_panel_label(ax, "a", y=1.08)

    ax = axes[1]
    keep = (sar[:, idx] > 1e-3 * sar[:, idx].max()) & (fem[:, idx] > 0)
    ax.scatter(sar[keep, idx], fem[keep, idx], s=2, color=P["blue"], alpha=0.5, lw=0)
    lim = [
        min(sar[keep, idx].min(), fem[keep, idx].min()),
        max(sar[keep, idx].max(), fem[keep, idx].max()),
    ]
    ax.plot(lim, lim, color="k", lw=0.6, ls="--")
    ax.text(lim[1], lim[1], " y = x", fontsize=6.5, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sarvas (fT per nA·m)")
    ax.set_ylabel("FEM (fT per nA·m)")
    ax.set_title(f"every coil, source #{idx}", fontsize=8, fontweight="normal")
    add_panel_label(ax, "b", y=1.08)

    ax = axes[2]
    ratio = fem.max(axis=0) / np.maximum(sar.max(axis=0), 1e-30)
    ax.plot(z[order], ratio[order], color=P["purple"], lw=1.2)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.axhline(band, color=P["orange"], lw=0.8, ls="--")
    ax.text(
        0.98,
        0.36,
        f"benchmark {band:.2f}× (95% CI {lo:.2f}–{hi:.2f})",
        transform=ax.transAxes,
        fontsize=6.5,
        ha="right",
        color=P["orange"],
    )
    ax.set_yscale("log")
    ax.set_xlabel("source height z (mm)")
    ax.set_ylabel("FEM ÷ Sarvas, best coil")
    _mark_levels(ax, levels)
    add_panel_label(ax, "c", y=1.08)

    fig.tight_layout()
    save_figure(fig, out)
    print(
        f"forward models: FEM/Sarvas benchmark {band:.2f} (95% CI {lo:.2f}-{hi:.2f}); "
        f"per-source best-coil ratio median {np.median(ratio):.2f}, "
        f"range {ratio.min():.2f}-{ratio.max():.2f}; source #{idx} {ratio[idx]:.2f}"
    )
    print(f"  -> {out}")


def sensor_design(cfg, base_config: str, out: Path, sweep_json: Path | None) -> None:
    """Trials along the cord per array; noise level; array size."""
    import matplotlib.pyplot as plt

    from inob.analysis.placement import nearest_array_info, optimise_array
    from inob.analysis.snr import compute_noise_floors
    from inob.cli._common import apply_source_target
    from inob.config import load_config
    from inob.io.npz import load_leadfield
    from inob.sensors.layout import radial_channels
    from inob.sources.cap import longitudinal_leadfield
    from inob.viz.detectability import default_source_idx
    from inob.viz.style import NATURE_PALETTE as P
    from inob.viz.style import add_panel_label, apply_nature_style, save_figure

    meg = load_leadfield(cfg.outputs.forward_npz)
    eeg = load_leadfield(cfg.outputs.forward_eeg_npz)
    wb_path = cfg.outputs.forward_npz.parent / "duneuro_eeg_leadfield_wholebody_spine.npz"
    wb = load_leadfield(wb_path) if wb_path.exists() else None
    f = compute_noise_floors(cfg)
    idx = default_source_idx(meg, eeg)
    z = meg.source_pos[:, 2]
    order = np.argsort(z)
    # The moment along the cord, as `inob source-models` uses for its
    # "stationary" model, so panel a and the paper's text quote one number.
    long_meg = np.abs(longitudinal_leadfield(meg.L_fT_per_nAm, meg.source_pos)[0])
    rad = radial_channels(list(meg.channel_names))

    def bipolar(lf) -> np.ndarray:
        col = longitudinal_leadfield(lf.L_fT_per_nAm, lf.source_pos)[0]
        return col.max(axis=0) - col.min(axis=0)

    sm, se = f.meg_per_channel_fT, f.eeg_per_channel_uV
    curves = {
        "MXG, OPM triaxial": (_trials(long_meg.max(axis=0) * SPINE_Q, sm), P["blue"], "-"),
        "MXG, OPM radial only": (_trials(long_meg[rad].max(axis=0) * SPINE_Q, sm), P["blue"], "--"),
        "EXG, 32-contact paddle": (_trials(bipolar(eeg) * SPINE_Q, se), P["red"], "-"),
    }
    curves["MXG, SQUID triaxial (alternative)"] = (
        _trials(long_meg.max(axis=0) * SPINE_Q, SQUID_FLOOR_FT),
        P["teal"],
        ":",
    )
    if wb is not None:
        curves["EXG, 1,000-contact whole body"] = (
            _trials(bipolar(wb) * SPINE_Q, se),
            P["red"],
            "--",
        )

    n_panels = 4 if sweep_json else 3
    apply_nature_style()
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(2.45 * n_panels, 2.7),
        gridspec_kw={"width_ratios": [1.5, 1.2] + [1] * (n_panels - 2)},
    )

    ax = axes[0]
    ax.axhspan(*CLINICAL_BUDGET, color=P["glow"], alpha=0.25, lw=0)
    ax.text(
        z.min() + 4,
        CLINICAL_BUDGET[1] * 0.9,
        "clinical SSEP budget",
        fontsize=6,
        va="top",
        ha="left",
    )
    for name, (n, colour, ls) in curves.items():
        ax.plot(z[order], n[order], color=colour, ls=ls, lw=1.1, label=name)
    ax.axvline(z[idx], color=P["glow"], lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 1e5)
    ax.set_xlabel("source height z (mm)")
    ax.set_ylabel(f"trials to SNR 3 at {SPINE_Q} nA·m")
    ax.legend(loc="upper center", fontsize=6, ncol=2, bbox_to_anchor=(0.5, -0.22))
    _mark_levels(ax, _levels(cfg, z))
    add_panel_label(ax, "a", y=1.08)
    print(f"sensor design: source #{idx}, z = {z[idx]:.0f} mm, Q = {SPINE_Q} nA·m")
    for name, (n, _, _) in curves.items():
        print(
            f"  {name:<30} {int(n[idx]):>6} trials at #{idx}; "
            f"{int(n.min())}-{int(n.max())} along the cord"
        )

    ax = axes[1]
    peak = float(long_meg[:, idx].max()) * SPINE_Q
    names, n_pre = [], []
    for overrides, label in OPM_PRESETS:
        c = apply_source_target(load_config(base_config, overrides=overrides), "spine")
        sigma = compute_noise_floors(c).meg_per_channel_fT
        names.append(label)
        n_pre.append(int(_trials(peak, sigma)[0]))
        print(f"  {' '.join(overrides):<70} effective floor {sigma:6.1f} fT -> {n_pre[-1]} trials")
    names.append("SQUID\n(alt.)\n4")
    n_pre.append(int(_trials(peak, SQUID_FLOOR_FT)[0]))
    print(f"  SQUID floor {SQUID_FLOOR_FT:.0f} fT -> {n_pre[-1]} trials")
    bars = ax.bar(
        range(len(names)),
        n_pre,
        color=[P["blue"], P["blue"], P["teal"], P["purple"], P["grey"]],
    )
    bars[1].set_alpha(0.55)
    bars[-1].set_hatch("//")
    for b, n in zip(bars, n_pre, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, n, f"{n}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(names)), names, fontsize=5.5, rotation=0)
    ax.set_ylabel(f"trials to SNR 3, source #{idx}")
    ax.set_title("MXG noise level", fontsize=8, fontweight="normal")
    ax.set_xlabel("sensor; noise density (fT/√Hz)", fontsize=7)
    add_panel_label(ax, "b", y=1.08)

    ax = axes[2]
    q = SPINE_Q
    sigma = f.meg_per_channel_fT
    L = meg.L_fT_per_nAm
    opt = optimise_array(
        L, meg.coil_pos, n_sensors=max(ARRAY_SIZES), noise_floor=sigma, moment_nAm=q
    )
    near = [
        nearest_array_info(
            L, meg.coil_pos, meg.source_pos, n_sensors=n, noise_floor=sigma, moment_nAm=q
        )
        for n in ARRAY_SIZES
    ]
    got = [opt.info_bits[n - 1] for n in ARRAY_SIZES]
    ax.plot(ARRAY_SIZES, got, "o-", color=P["red"], ms=3, lw=1.1, label="information-optimised")
    ax.plot(ARRAY_SIZES, near, "^--", color=P["blue"], ms=3, lw=1.1, label="nearest the cord")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ARRAY_SIZES, [str(n) for n in ARRAY_SIZES])
    ax.set_xlabel("triaxial OPMs in the array")
    ax.set_ylabel("information about the cord (bits)")
    ax.legend(loc="upper left", fontsize=6)
    add_panel_label(ax, "c", y=1.08)
    for n, a, b in zip(ARRAY_SIZES, got, near, strict=True):
        print(
            f"  {n:>3} sensors: optimised {a:6.2f} bits, nearest {b:6.2f} bits, ratio {a / b:.2f}"
        )

    if sweep_json:
        rows = json.loads(Path(sweep_json).read_text())
        ax = axes[3]
        for preset, colour in (("quspin_qzfm3", P["blue"]), ("quspin_qzfm2", P["teal"])):
            sel = sorted((r for r in rows if r["preset"] == preset), key=lambda r: r["standoff_mm"])
            if sel:
                ax.plot(
                    [r["standoff_mm"] for r in sel],
                    [math.ceil(r["trials"] - 1e-9) for r in sel],
                    "o-",
                    color=colour,
                    ms=3,
                    lw=1.1,
                    label=preset.replace("quspin_qzfm", "Gen-"),
                )
        ax.set_xlabel("sensor stand-off from skin (mm)")
        ax.set_ylabel(f"trials to SNR 3, source #{idx}")
        ax.legend(loc="upper left", fontsize=6)
        add_panel_label(ax, "d", y=1.08)

    fig.tight_layout()
    save_figure(fig, out)
    print(f"  -> {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "paper")
    ap.add_argument("--only", choices=("forward_models", "sensor_design"), action="append")
    ap.add_argument(
        "--sweep-json", type=Path, help="sensor_sweep.py --json output (stand-off panel)"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)

    from inob.cli._common import apply_source_target
    from inob.config import load_config

    cfg = apply_source_target(load_config(args.config), "spine")
    todo = args.only or ["forward_models", "sensor_design"]
    if "forward_models" in todo:
        forward_models(cfg, args.out_dir / "spine_forward_models.png")
    if "sensor_design" in todo:
        sensor_design(cfg, args.config, args.out_dir / "spine_sensor_design.png", args.sweep_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
