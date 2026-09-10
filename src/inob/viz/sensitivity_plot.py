"""MEG-vs-EEG conductivity-sensitivity comparison figure.

Reads the per-modality sweep JSON written by ``inob-sensitivity`` and
renders a grouped bar chart of the per-channel relative leadfield change
induced by perturbing each tissue's conductivity. The paper claim is the
visible asymmetry: MEG bars are near-flat while EEG bars are large for the
bone/skin perturbations — i.e. the magnetic forward is insensitive to the
dominant volume-conductor conductivity uncertainty.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inob.config import Config
from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
    save_figure,
)

logger = logging.getLogger(__name__)

# Which summary statistic to plot (the JSON also carries rms / p50).
_STAT = "p95_rel_change"
_STAT_LABEL = "95th-percentile relative change in |leadfield|"


def _load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("results", [])


def _key(r: dict) -> str:
    return f"{r['tissue']} ×{r['factor']:g}"


def render_sensitivity(
    cfg: Config,
    *,
    out_path: Path | None = None,
    dpi: int = 300,
) -> Path:
    """Render the MEG-vs-EEG sensitivity comparison bar figure.

    Requires ``sensitivity_meg.json`` and/or ``sensitivity_eeg.json`` under
    ``cfg.outputs.sensitivity_dir``. Missing modalities are simply omitted.
    """
    apply_nature_style()
    sdir = cfg.outputs.sensitivity_dir
    meg = _load_results(sdir / "sensitivity_meg.json")
    eeg = _load_results(sdir / "sensitivity_eeg.json")
    if not meg and not eeg:
        raise FileNotFoundError(f"no sensitivity_*.json under {sdir}; run inob-sensitivity first")

    # Union of perturbation keys, preserving config tissue/factor order.
    order: list[str] = []
    for r in (*meg, *eeg):
        k = _key(r)
        if k not in order:
            order.append(k)

    meg_map = {_key(r): r[_STAT] * 100 for r in meg}
    eeg_map = {_key(r): r[_STAT] * 100 for r in eeg}
    meg_vals = [meg_map.get(k, np.nan) for k in order]
    eeg_vals = [eeg_map.get(k, np.nan) for k in order]

    x = np.arange(len(order))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(order) + 1.5), 3.2))
    ax.bar(x - w / 2, meg_vals, w, label="OPM-MEG", color=NATURE_PALETTE["blue"])
    ax.bar(x + w / 2, eeg_vals, w, label="HD-EEG", color=NATURE_PALETTE["red"])

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel(f"{_STAT_LABEL} (%)")
    ax.set_title("Conductivity-perturbation sensitivity: MEG vs EEG")
    ax.legend(loc="upper left")

    # Annotate each bar with its value so small MEG bars stay legible.
    for xi, v in zip(x - w / 2, meg_vals, strict=True):
        if np.isfinite(v):
            ax.annotate(
                f"{v:.1f}",
                (xi, v),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                fontsize=6.5,
                color=NATURE_PALETTE["blue"],
            )
    for xi, v in zip(x + w / 2, eeg_vals, strict=True):
        if np.isfinite(v):
            ax.annotate(
                f"{v:.1f}",
                (xi, v),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                fontsize=6.5,
                color=NATURE_PALETTE["red"],
            )

    add_panel_label(ax, "a")
    fig.tight_layout()

    return save_figure(fig, out_path or (sdir / "sensitivity_comparison.png"), dpi=dpi)
