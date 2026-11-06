#!/usr/bin/env python3
"""The remaining spinal-cord paper figures, from the leadfields already on disk.

`scripts/spine_figures.py` draws the forward-model and sensor-design figures.
This script draws the four that read the same leadfields and the `inob
source-models` result:

    trials_to_detect   trials to SNR 3 along the cord for the magnetic (MXG,
                       OPM) and electric (EXG) arrays, with the SQUID
                       magnetoneurograph of the Japanese studies as an
                       alternative to the OPM
    best_channel       the largest field any sensor position records per nA.m
    electrode_coverage the largest bipolar difference on the paddle and the
                       whole-body array
    source_models      stationary against ascending source, MXG against EXG

    python3 scripts/spine_paper_figures.py                 # all four, outputs/paper/
    python3 scripts/spine_paper_figures.py --only source_models

The source-model figure needs `inob source-models --json-out FILE`; if no
--source-models-json is given the command is run for you. Nothing here needs
DUNEuro. The figures use the vertebral-level labels of `scripts/spine_figures.py`
and add lumbar, thoracic and cervical bands.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spine_figures import SPINE_Q, SQUID_FLOOR_FT, _levels, _trials

REPO_ROOT = Path(__file__).resolve().parent.parent
#: Clinical somatosensory-evoked-potential averaging budget, trials.
CLINICAL_BUDGET = (500, 2000)
SHOWN_LEVELS = ("C1", "C4", "C7", "T4", "T8", "T12")


def _load(cfg):
    from inob.analysis.snr import compute_noise_floors
    from inob.io.npz import load_leadfield
    from inob.viz.detectability import default_source_idx

    meg = load_leadfield(cfg.outputs.forward_npz)
    eeg = load_leadfield(cfg.outputs.forward_eeg_npz)
    wb = load_leadfield(
        cfg.outputs.forward_npz.parent / "duneuro_eeg_leadfield_wholebody_spine.npz"
    )
    return meg, eeg, wb, compute_noise_floors(cfg), default_source_idx(meg, eeg)


def _bipolar(leadfield) -> np.ndarray:
    """Largest difference between any two contacts, for a moment along the cord."""
    from inob.sources.cap import longitudinal_leadfield

    col = longitudinal_leadfield(leadfield.L_fT_per_nAm, leadfield.source_pos)[0]
    return col.max(axis=0) - col.min(axis=0)


def _vertebral_levels(ax, cfg, z, fontsize):
    for zz, name in _levels(cfg, z):
        if name in SHOWN_LEVELS:
            ax.axvline(zz, color="#D8D8D8", lw=0.8, zorder=0)
            ax.text(
                zz,
                1.0,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color="#6A6A6A",
            )


def _regions(ax, cfg, z, fontsize):
    """Lumbar, thoracic and cervical bands, from the first z of the T12 and C7 levels."""
    levels = {name: zz for zz, name in _levels(cfg, z)}
    z_t12, z_c7 = levels["T12"], levels["C7"]
    for lo, hi, name in (
        (z.min(), z_t12, "lumbar"),
        (z_t12, z_c7, "thoracic"),
        (z_c7, z.max(), "cervical"),
    ):
        ax.annotate(
            "",
            xy=(lo + 2, 1.13),
            xytext=(hi - 2, 1.13),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops={"arrowstyle": "<->", "color": "0.35", "lw": 1},
        )
        ax.text(
            (lo + hi) / 2,
            1.15,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="0.25",
        )


def _tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def trials_to_detect(cfg, out: Path) -> None:
    import matplotlib.pyplot as plt

    from inob.sensors.layout import radial_channels
    from inob.sources.cap import longitudinal_leadfield
    from inob.viz.style import NATURE_PALETTE as P

    meg, eeg, wb, floors, idx = _load(cfg)
    z = meg.source_pos[:, 2]
    order = np.argsort(z)
    long_meg = np.abs(longitudinal_leadfield(meg.L_fT_per_nAm, meg.source_pos)[0])
    radial = radial_channels(list(meg.channel_names))
    sm, se = floors.meg_per_channel_fT, floors.eeg_per_channel_uV
    # (curve, colour, line style, y of its end label)
    curves = {
        "EXG, paddle": (_trials(_bipolar(eeg) * SPINE_Q, se), P["red"], "-", 330),
        "MXG, OPM radial only": (
            _trials(long_meg[radial].max(0) * SPINE_Q, sm),
            P["blue"],
            "--",
            130,
        ),
        "MXG, OPM triaxial": (_trials(long_meg.max(0) * SPINE_Q, sm), P["blue"], "-", 45),
        "MXG, SQUID triaxial (alt.)": (
            _trials(long_meg.max(0) * SPINE_Q, SQUID_FLOOR_FT),
            P["teal"],
            ":",
            12,
        ),
        "EXG, whole body": (_trials(_bipolar(wb) * SPINE_Q, se), P["red"], "--", 2.2),
    }
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.axhspan(*CLINICAL_BUDGET, color="0.5", alpha=0.15, lw=0)
    ax.text(
        z.min() + 5,
        CLINICAL_BUDGET[1] * 0.9,
        "clinical SSEP budget",
        fontsize=13,
        color="0.4",
        va="top",
    )
    for name, (trials, colour, style, label_y) in curves.items():
        ax.plot(z[order], trials[order], color=colour, ls=style, lw=2.4)
        ax.text(z.max() + 8, label_y, name, color=colour, fontsize=13, va="center")
    ax.axvline(z[idx], color="k", lw=1.1, ls=":")
    ax.text(z[idx] + 4, 7e4, "#72 (C6)", fontsize=12, va="top")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 1e5)
    _vertebral_levels(ax, cfg, z, 12)
    _regions(ax, cfg, z, 13)
    ax.set_xlabel("source height z (mm)", fontsize=15)
    ax.set_ylabel(f"trials to SNR 3 at {SPINE_Q} nA·m", fontsize=15)
    ax.tick_params(labelsize=13)
    _tidy(ax)
    fig.subplots_adjust(right=0.69, top=0.84, bottom=0.14, left=0.10)
    fig.savefig(out / "fig_spine_trials_to_detect.png", dpi=250)
    plt.close(fig)
    print(
        f"trials to detect at #{idx}: "
        + "; ".join(f"{n} {int(c[0][idx])}" for n, c in curves.items())
    )


def best_channel(cfg, out: Path) -> None:
    import matplotlib.pyplot as plt

    from inob.analysis.snr import per_source_peak
    from inob.sensors.layout import radial_channels
    from inob.viz.style import NATURE_PALETTE as P

    meg, _, _, _, idx = _load(cfg)
    z = meg.source_pos[:, 2]
    order = np.argsort(z)
    radial = radial_channels(list(meg.channel_names))
    triaxial = per_source_peak(meg.L_fT_per_nAm)
    radial_only = per_source_peak(meg.L_fT_per_nAm[radial])
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(z[order], triaxial[order], color=P["blue"], lw=2.2, label="triaxial")
    ax.plot(z[order], radial_only[order], color=P["blue"], lw=2.2, ls="--", label="radial only")
    ax.plot(z[idx], triaxial[idx], "o", mfc="white", mec=P["blue"], ms=9, mew=2)
    ax.text(z[idx] + 8, triaxial[idx] - 6, "#72", fontsize=13)
    _vertebral_levels(ax, cfg, z, 12)
    _regions(ax, cfg, z, 13)
    ax.set_ylim(0, 70)
    ax.set_xlabel("source height z (mm)")
    ax.set_ylabel("MXG maximum channel field\n(fT per nA·m)")
    ax.legend(frameon=False, loc="lower right", fontsize=13)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_spine_best_channel.png", dpi=250)
    plt.close(fig)
    print(f"maximum channel field at #{idx}: {triaxial[idx]:.1f} fT per nA.m")


def electrode_coverage(cfg, out: Path) -> None:
    import matplotlib.pyplot as plt

    from inob.viz.style import NATURE_PALETTE as P

    meg, eeg, wb, _, _ = _load(cfg)
    z = meg.source_pos[:, 2]
    order = np.argsort(z)
    paddle, whole = _bipolar(eeg), _bipolar(wb)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(z[order], whole[order], color=P["red"], ls="--", lw=2.4)
    ax.plot(z[order], paddle[order], color=P["red"], lw=2.4)
    ax.text(
        z.max() + 8,
        whole[order][-1],
        "EXG, 1,000-contact\nwhole body",
        color=P["red"],
        fontsize=13,
        va="center",
    )
    ax.text(
        z.max() + 8,
        paddle[order][-1] * 0.75,
        "EXG, 32-contact\npaddle",
        color=P["red"],
        fontsize=13,
        va="center",
    )
    ax.set_yscale("log")
    _vertebral_levels(ax, cfg, z, 12)
    _regions(ax, cfg, z, 13)
    ax.set_xlabel("source height z (mm)", fontsize=15)
    ax.set_ylabel("EXG maximum bipolar difference\n(µV per nA·m)", fontsize=14)
    ax.tick_params(labelsize=13)
    _tidy(ax)
    fig.subplots_adjust(right=0.72, top=0.84, bottom=0.15, left=0.15)
    fig.savefig(out / "fig_spine_electrode_coverage.png", dpi=250)
    plt.close(fig)


def source_models(result: dict, out: Path) -> None:
    """Stationary against ascending source, from `inob source-models --json-out`."""
    import matplotlib.pyplot as plt

    from inob.viz.style import NATURE_PALETTE as P

    models = [result["models"][k] for k in ("stationary", "ascending")]
    snr_m = [m["meg_single_trial_SNR"] for m in models]
    snr_e = [m["eeg_single_trial_SNR"] for m in models]
    trials_m = [m["meg_trials_for_SNR3"] for m in models]
    trials_e = [m["eeg_trials_for_SNR3"] for m in models]
    blue, red = P["blue"], P["red"]
    x = np.arange(2)
    width = 0.38
    labels = ["stationary\n(one segment)", "ascending\n(travelling volley)"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))

    ax = axes[0]
    ax.bar(x - width / 2, snr_m, width, color=blue, label="MXG (OPM, maximum-amplitude channel)")
    ax.bar(x + width / 2, snr_e, width, color=red, label="EXG (maximum bipolar difference)")
    ax.axhline(3, color="k", ls="--")
    ax.text(1.45, 3.3, "SNR = 3 (Rose criterion)", ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 10)
    ax.set_ylabel("single-trial SNR", fontsize=11)
    ax.set_title("a  single-trial SNR", loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(0.0, 0.58), frameon=False)

    ax = axes[1]
    ax.axhspan(*CLINICAL_BUDGET, color="#f5deb3", alpha=0.6, zorder=0)
    ax.bar(x - width / 2, trials_m, width, color=blue, zorder=3)
    ax.bar(x + width / 2, trials_e, width, color=red, zorder=3)
    ax.text(-0.45, 2300, "clinical budget 500–2000", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(10, 6000)
    ax.set_ylabel("trials to reach SNR 3", fontsize=11)
    ax.set_title("b  averaging cost", loc="left", fontsize=12)
    for a in axes[:2]:
        a.set_xticks(x, labels, fontsize=10)

    ax = axes[2]
    prop_m = models[1]["meg_fT"] / models[0]["meg_fT"]
    prop_e = models[1]["eeg_uV"] / models[0]["eeg_uV"]
    ratios = [snr_m[0] / snr_e[0], snr_m[1] / snr_e[1], prop_m, prop_e]
    ax.bar(range(4), ratios, color=[P["purple"], P["purple"], blue, red])
    ax.axhline(1, color="k", ls=":")
    ax.set_yscale("log")
    ax.set_xticks(
        range(4),
        [
            "MXG/EXG\nstationary",
            "MXG/EXG\nascending",
            "propagation\npenalty (MXG)",
            "propagation\npenalty (EXG)",
        ],
        rotation=30,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("ratio (×)", fontsize=11)
    ax.set_title("c  independent of source strength", loc="left", fontsize=12)

    fig.suptitle(
        f"spine: MXG vs EXG on the same cord activity — Q = {result['Q_nAm']} nA·m per active "
        f"source, source #{result['source_idx']} (z = {result['source_z_mm']:.0f} mm, C6)",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        f"{result['n_sources']} sources over {result['span_mm']:.0f} mm; transit "
        f"{result['transit_ms']:.1f} ms vs {result['ap_width_ms']} ms AP width. Panels a–b scale "
        "linearly with Q; panel c does not depend on it.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out / "fig5_spine_source_models.png", dpi=200)
    plt.close(fig)
    print(f"source models: trials MXG {trials_m}, EXG {trials_e}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/spine.yaml")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/paper"))
    ap.add_argument(
        "--only",
        choices=("trials_to_detect", "best_channel", "electrode_coverage", "source_models"),
        action="append",
    )
    ap.add_argument(
        "--source-models-json",
        type=Path,
        help="`inob source-models --json-out` result; run for you if omitted",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)

    from inob.cli._common import apply_source_target
    from inob.config import load_config

    cfg = apply_source_target(load_config(args.config), "spine")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = args.only or ["trials_to_detect", "best_channel", "electrode_coverage", "source_models"]
    if "trials_to_detect" in todo:
        trials_to_detect(cfg, args.out_dir)
    if "best_channel" in todo:
        best_channel(cfg, args.out_dir)
    if "electrode_coverage" in todo:
        electrode_coverage(cfg, args.out_dir)
    if "source_models" in todo:
        json_path = args.source_models_json
        if json_path is None:
            json_path = Path(tempfile.mkdtemp()) / "source_models.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "inob.cli.main",
                    "source-models",
                    "--config",
                    args.config,
                    "--source-target",
                    "spine",
                    "--json-out",
                    str(json_path),
                    "--no-figure",
                ],
                check=True,
                cwd=REPO_ROOT,
            )
        source_models(json.loads(json_path.read_text()), args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
