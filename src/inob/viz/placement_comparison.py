"""Figure: does the forward model beat a tape measure at placing sensors?

Three panels answering the comparison in :mod:`inob.analysis.placement`:
where each arm puts its sensors, how the gap behaves as the array grows,
and how it behaves as the naive arm's aim degrades.

The honest answer on the shipped anatomy is "barely", and the figure is
drawn to show that rather than to hide it — the two curves sitting on top of
each other is the result, and a reader should be able to see it at a glance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from inob.analysis.placement import (
    PlacementComparison,
    compare_placement,
    sweep_aiming_error,
    sweep_sensor_counts,
)
from inob.analysis.snr import compute_noise_floors, per_source_peak
from inob.config import Config, source_region_label, target_output
from inob.io.npz import load_leadfield
from inob.sensors.layout import require_triaxial
from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
    save_figure,
)

logger = logging.getLogger(__name__)

_INFORMED = NATURE_PALETTE["red"]
_NAIVE = NATURE_PALETTE["blue"]


def render_placement_comparison(
    cfg: Config,
    *,
    forward_npz: Path | None = None,
    source_idx: int = -1,
    n_sensors: int = 20,
    min_spacing_mm: float = 30.0,
    out_path: Path | None = None,
    dpi: int = 300,
) -> tuple[Path, list[PlacementComparison]]:
    """Render the placement comparison for one source.

    ``source_idx`` defaults to the best-coupled source, which is where a real
    study would aim and so the case the naive arm finds easiest.

    Returns the figure path and the sensor-count sweep behind panel b, so a
    caller can print the numbers rather than make the reader read them off.
    """
    apply_nature_style()
    npz = forward_npz or cfg.outputs.forward_npz
    lf = load_leadfield(npz)
    require_triaxial(lf.channel_names, "the placement comparison")
    leadfield = lf.L_fT_per_nAm
    noise = compute_noise_floors(cfg).meg_per_channel_fT

    if source_idx < 0:
        source_idx = int(np.argmax(per_source_peak(leadfield)))
    logger.info("placement comparison at source %d of %d", source_idx, leadfield.shape[1] // 3)

    headline = compare_placement(
        leadfield,
        lf.coil_pos,
        lf.source_pos,
        source_idx=source_idx,
        n_sensors=n_sensors,
        min_spacing_mm=min_spacing_mm,
        noise_floor=noise,
    )
    logger.info(
        "source %d: informed %.1f fT/nA·m (%.1f trials) vs naive %.1f (%.1f trials) "
        "— %.2fx amplitude, %.2fx trials",
        source_idx,
        headline.informed.amplitude,
        headline.informed.trials_to_detect,
        headline.naive.amplitude,
        headline.naive.trials_to_detect,
        headline.amplitude_ratio,
        headline.trials_ratio,
    )
    counts = sweep_sensor_counts(
        leadfield,
        lf.coil_pos,
        lf.source_pos,
        source_idx=source_idx,
        min_spacing_mm=min_spacing_mm,
        noise_floor=noise,
    )
    aiming = sweep_aiming_error(
        leadfield,
        lf.coil_pos,
        lf.source_pos,
        source_idx=source_idx,
        n_sensors=n_sensors,
        min_spacing_mm=min_spacing_mm,
        noise_floor=noise,
    )

    fig = plt.figure(figsize=(11.5, 3.8))
    ax_map = fig.add_subplot(1, 3, 1, projection="3d")
    ax_count = fig.add_subplot(1, 3, 2)
    ax_aim = fig.add_subplot(1, 3, 3)

    _draw_positions(ax_map, lf, headline)
    _draw_count_sweep(ax_count, counts)
    _draw_aiming_sweep(ax_aim, aiming)

    for ax, label in ((ax_map, "a"), (ax_count, "b"), (ax_aim, "c")):
        add_panel_label(ax, label)

    fig.suptitle(
        f"{source_region_label(cfg)}: model-informed vs nearest-to-source placement "
        f"({n_sensors} triaxial sensors, {min_spacing_mm:.0f} mm apart)",
        y=1.02,
    )
    # tight_layout leaves the 3-D axis's z-label overlapping panel b.
    fig.subplots_adjust(left=0.02, right=0.98, wspace=0.42, top=0.84)
    out = out_path or target_output(cfg, "placement_comparison.png")
    return save_figure(fig, out, dpi=dpi), counts


def _draw_positions(ax, lf, cmp_) -> None:
    n_pos = require_triaxial(lf.channel_names, "the placement comparison")
    positions = lf.coil_pos[:n_pos]
    ax.scatter(*positions.T, s=1.5, c=NATURE_PALETTE["stone"], alpha=0.35, linewidths=0)
    for arm, colour, marker in (
        (cmp_.informed, _INFORMED, "o"),
        (cmp_.naive, _NAIVE, "^"),
    ):
        picked = positions[arm.position_idx]
        ax.scatter(
            *picked.T,
            s=26,
            c=colour,
            marker=marker,
            depthshade=False,
            edgecolors="none",
            label=f"{arm.name} ({arm.amplitude:.0f} fT/nA·m)",
        )
    ax.scatter(
        *cmp_.source_pos,
        s=90,
        c=NATURE_PALETTE["glow"],
        marker="*",
        depthshade=False,
        edgecolors="none",
        label="source",
    )
    ax.set_xlabel("x (mm)", labelpad=-4)
    ax.set_ylabel("y (mm)", labelpad=-4)
    ax.set_zlabel("z (mm)", labelpad=-4)
    ax.tick_params(labelsize=5, pad=-2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    # Anterolateral, matching the other body figures (docs/COORDINATES.md).
    ax.view_init(elev=14, azim=42)
    # A flat array (every sensor at one height) has a zero extent, which
    # set_box_aspect divides by.
    extent = np.ptp(positions, axis=0)
    ax.set_box_aspect(np.where(extent > 0.0, extent, extent.max() or 1.0))
    ax.legend(loc="upper left", fontsize=6, frameon=False)
    ax.set_title("chosen positions", fontsize=8)


def _draw_count_sweep(ax, comparisons) -> None:
    counts = [c.n_sensors for c in comparisons]
    ax.plot(
        counts,
        [c.informed.amplitude for c in comparisons],
        "-o",
        ms=3.5,
        color=_INFORMED,
        label="informed",
    )
    ax.plot(
        counts,
        [c.naive.amplitude for c in comparisons],
        "--^",
        ms=3.5,
        color=_NAIVE,
        label="naive",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sensors in the array")
    ax.set_ylabel("matched-filter amplitude (fT/nA·m)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("against array size", fontsize=8)


def _draw_aiming_sweep(ax, aiming) -> None:
    errors = sorted(aiming)
    ax.plot(errors, [aiming[e] for e in errors], "-o", ms=3.5, color=_INFORMED)
    ax.axhline(1.0, lw=0.8, ls=":", color=NATURE_PALETTE["grey"])
    ax.set_xlabel("naive arm's aiming error (mm)")
    ax.set_ylabel("informed / naive amplitude")
    # Fixed to 1.0-2.0 on purpose. Auto-scaling a ratio that never leaves the
    # first few percent draws a dramatic-looking curve out of nothing; on this
    # scale a reader can see at a glance that there is almost no advantage to
    # be had, which is the result.
    ax.set_ylim(1.0, max(2.0, max(aiming.values()) * 1.05))
    ax.annotate(
        "no advantage",
        xy=(0.02, 1.0),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=6,
        color=NATURE_PALETTE["grey"],
    )
    ax.set_title("against localisation error", fontsize=8)
