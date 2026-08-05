"""Beautiful topoplot visualisations of MEG and EEG leadfields.

Designed to Nature Reviews figure standards (see :mod:`inob.viz.style`):

  * Top-left → bottom-right reading order.
  * Bold panel labels (a, b, c, …) in the top-left of each axes.
  * Saturated red/blue divergent palette (no red-green pairs).
  * Body context drawn neutrally (light skin tone) so signal pops.
  * 8 pt body text, no grid, minimal axis chrome.
  * Source unit anchored to a 1 nA·m dipole — so colour-bar values are
    directly readable as "fT" (MEG) and "µV" (EEG).

All four panels share a single source: the dipole position highlighted in
``glow`` (gold). Field amplitudes are the longitudinal (along-nerve) moment
— what a propagating compound action potential actually generates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from inob.config import Config, source_region_label
from inob.io.hdf5 import load_geometry, load_sensors
from inob.io.npz import load_leadfield
from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
    divergent_cmap,
    divergent_norm,
)

logger = logging.getLogger(__name__)


def _fmt_peak(value: float) -> str:
    """Format a peak amplitude with ~3 significant figures.

    A fixed ``.1f`` reads fine for a vagus/spine MEG peak (10s–1000s fT) but
    rounds a small EEG peak — muscle's common-average-referenced surface
    potential is ~0.005 µV — down to "0.0", making a real, if small, signal
    look like exactly zero in the legend.
    """
    if value == 0 or not np.isfinite(value):
        return f"{value:.1f}"
    from math import floor, log10
    decimals = max(0, 2 - floor(log10(abs(value))))
    return f"{value:.{decimals}f}"


@dataclass(frozen=True)
class TopoFrame:
    source_idx: int
    moment: str = "z"   # longitudinal moment along the cervical vagus


# Default to the longitudinal (Z) moment, the one a propagating CAP creates.
DEFAULT_MOMENT = "z"


# ── helpers ────────────────────────────────────────────────────────────────

def _column_for_source(L: np.ndarray, source_idx: int, moment: str) -> np.ndarray:
    """Slice the leadfield to a single (channel) vector for one source.

    ``moment="rms"`` averages over the three orthogonal moments;
    ``moment="x"|"y"|"z"`` returns that single moment's column (signed);
    ``moment="norm"`` returns the L2 norm across moments (always positive).
    """
    _C, three_S = L.shape
    if not (0 <= source_idx < three_S // 3):
        raise IndexError(f"source_idx {source_idx} out of range [0, {three_S // 3})")
    block = L[:, 3 * source_idx : 3 * source_idx + 3]
    if moment == "rms":
        return np.sqrt(np.mean(block ** 2, axis=1))
    if moment == "norm":
        return np.linalg.norm(block, axis=1)
    if moment in {"x", "y", "z"}:
        return block[:, "xyz".index(moment)]
    raise ValueError(f"unknown moment {moment!r}")


def _radial_channel_mask(labels: list[str]) -> np.ndarray:
    """Boolean mask selecting the R (radial) channels of a triaxial array."""
    n = len(labels)
    if n % 3 == 0 and labels[0].endswith("-R") and labels[n // 3].endswith("-T1"):
        mask = np.zeros(n, dtype=bool)
        mask[: n // 3] = True
        return mask
    return np.array([lab.endswith("-R") for lab in labels])


def _draw_skin_3d(
    ax, vertices: np.ndarray, faces: np.ndarray, *,
    alpha: float = 0.05, max_tris: int = 6_000,
    rng: np.random.Generator | None = None,
) -> None:
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(faces)
    if n > max_tris:
        idx = rng.choice(n, max_tris, replace=False)
        faces = faces[idx]
    coll = Poly3DCollection(
        vertices[faces], alpha=alpha,
        facecolor=NATURE_PALETTE["skin"], edgecolor="none",
    )
    ax.add_collection3d(coll)


def _grid_shape_from_labels(labels: tuple[str, ...]) -> tuple[int, int] | None:
    """Recover (rows, cols) from purely rectangular ``elec-RR-CC`` labels.

    Returns None when any label has a non-integer second/third token (e.g.
    paddle32's ``elec-head-00`` / ``elec-foot-00``); callers fall back to
    paddle reconstruction or a scatter plot.
    """
    rows, cols = -1, -1
    for lab in labels:
        parts = lab.split("-")
        if len(parts) < 3:
            return None
        try:
            r = int(parts[1])
            c = int(parts[2])
        except ValueError:
            return None
        rows = max(rows, r)
        cols = max(cols, c)
    return (rows + 1, cols + 1) if rows >= 0 else None


def _paddle_uv_from_labels(
    labels: tuple[str, ...], pitch_mm: float, head_offset_mm: float,
    foot_offset_mm: float,
) -> np.ndarray | None:
    """Reconstruct (u, v) coords for paddle32 channels in the patch frame.

    Returns ``(N, 2)`` mm or ``None`` if the labels are not paddle32.
    """
    has_head = any(lab.startswith("elec-head") for lab in labels)
    has_foot = any(lab.startswith("elec-foot") for lab in labels)
    if not (has_head or has_foot):
        return None
    rows = cols = -1
    for lab in labels:
        if lab.startswith("elec-head") or lab.startswith("elec-foot"):
            continue
        parts = lab.split("-")
        try:
            r = int(parts[1])
            c = int(parts[2])
        except (IndexError, ValueError):
            continue
        rows = max(rows, r)
        cols = max(cols, c)
    rows += 1
    cols += 1
    u = (np.arange(cols) - (cols - 1) / 2.0) * pitch_mm
    v = (np.arange(rows) - (rows - 1) / 2.0) * pitch_mm
    out = np.zeros((len(labels), 2), dtype=np.float64)
    for i, lab in enumerate(labels):
        if lab.startswith("elec-head"):
            out[i] = [0.0, v.max() + head_offset_mm]
        elif lab.startswith("elec-foot"):
            out[i] = [0.0, v.min() - foot_offset_mm]
        else:
            parts = lab.split("-")
            r = int(parts[1])
            c = int(parts[2])
            out[i] = [u[c], v[r]]
    return out


def _draw_paddle_silhouette(ax, uv: np.ndarray, *, contact_radius: float = 1.6,
                             head_radius_factor: float = 1.6) -> None:
    """Outline the PEDOT:PSS paddle as a teal substrate behind the contacts."""
    from matplotlib.patches import Circle

    head = uv[0]
    foot = uv[-1]
    body = uv[1:-1]
    body_xmin = float(body[:, 0].min()) - 4.0
    body_xmax = float(body[:, 0].max()) + 4.0
    body_ymin = float(body[:, 1].min()) - 4.0
    body_ymax = float(body[:, 1].max()) + 4.0

    # Body slab
    ax.add_patch(plt.Rectangle(
        (body_xmin, body_ymin),
        body_xmax - body_xmin, body_ymax - body_ymin,
        facecolor="#9FCFD3", edgecolor=NATURE_PALETTE["axis"],
        linewidth=0.4, alpha=0.55, zorder=1,
    ))
    # Head + foot connector slabs
    ax.add_patch(plt.Rectangle(
        (-2.0, body_ymax),
        4.0, head[1] - body_ymax,
        facecolor="#9FCFD3", edgecolor="none", alpha=0.55, zorder=1,
    ))
    ax.add_patch(plt.Rectangle(
        (-2.0, foot[1]),
        4.0, body_ymin - foot[1],
        facecolor="#9FCFD3", edgecolor="none", alpha=0.55, zorder=1,
    ))
    # Head and foot terminations (rounded)
    ax.add_patch(Circle((head[0], head[1]),
                        contact_radius * head_radius_factor + 1.2,
                        facecolor="#9FCFD3", edgecolor=NATURE_PALETTE["axis"],
                        linewidth=0.4, alpha=0.55, zorder=1))
    ax.add_patch(Circle((foot[0], foot[1]),
                        contact_radius + 1.2,
                        facecolor="#9FCFD3", edgecolor=NATURE_PALETTE["axis"],
                        linewidth=0.4, alpha=0.55, zorder=1))


def _draw_eeg_2d_topoplot(ax, electrodes, val: np.ndarray, cfg: Config, *,
                          vmin: float, vmax: float) -> None:
    """Render the EEG patch as a 2-D topoplot.

    Auto-detects layout from the channel labels: paddle32 → bicubic
    interpolation over the body grid plus head/foot circles drawn on the
    paddle silhouette; rectangular → imshow heatmap; anything else →
    scatter plot in the world frame.
    """
    pitch = cfg.electrodes.contact_pitch_mm
    head_off = cfg.electrodes.head_offset_mm
    foot_off = cfg.electrodes.foot_offset_mm
    paddle_uv = _paddle_uv_from_labels(electrodes.labels, pitch, head_off, foot_off)

    if paddle_uv is not None:
        # Body (rows × cols) interpolated heatmap; head/foot rendered as filled circles.
        rows = cols = -1
        for lab in electrodes.labels:
            if lab.startswith("elec-head") or lab.startswith("elec-foot"):
                continue
            parts = lab.split("-")
            r = int(parts[1])
            c = int(parts[2])
            rows = max(rows, r)
            cols = max(cols, c)
        rows += 1
        cols += 1
        body_idx = [i for i, lab in enumerate(electrodes.labels)
                    if not (lab.startswith("elec-head") or lab.startswith("elec-foot"))]
        body_uv = paddle_uv[body_idx]
        body_grid = np.full((rows, cols), np.nan)
        for i in body_idx:
            parts = electrodes.labels[i].split("-")
            body_grid[int(parts[1]), int(parts[2])] = val[i]

        _draw_paddle_silhouette(ax, paddle_uv)
        u_lo = body_uv[:, 0].min() - 0.5 * pitch
        u_hi = body_uv[:, 0].max() + 0.5 * pitch
        v_lo = body_uv[:, 1].min() - 0.5 * pitch
        v_hi = body_uv[:, 1].max() + 0.5 * pitch
        im = ax.imshow(
            body_grid, cmap=divergent_cmap(), vmin=vmin, vmax=vmax,
            extent=(u_lo, u_hi, v_lo, v_hi),
            aspect="equal", origin="lower", interpolation="bicubic", zorder=2,
        )
        # Plot circular contacts on top so positions are visible.
        from matplotlib.patches import Circle
        contact_r = 1.6
        for k, lab in enumerate(electrodes.labels):
            uv = paddle_uv[k]
            if lab.startswith("elec-head"):
                r = contact_r * 1.8
            else:
                r = contact_r
            ax.add_patch(Circle(
                (uv[0], uv[1]), r,
                facecolor=divergent_cmap()(
                    (val[k] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                ),
                edgecolor=NATURE_PALETTE["axis"], linewidth=0.4, zorder=3,
            ))
        ax.set_xlim(paddle_uv[:, 0].min() - 8.0, paddle_uv[:, 0].max() + 8.0)
        ax.set_ylim(paddle_uv[:, 1].min() - 8.0, paddle_uv[:, 1].max() + 8.0)
        ax.set_aspect("equal")
        ax.set_xlabel(f"u  (mm, patch frame)  ·  pitch {pitch:.1f} mm")
        ax.set_ylabel("v  (mm, along-body)")
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.85, fraction=0.04, pad=0.03)
        cb.set_label("µV  (1 nA·m source)", fontsize=8)
        cb.outline.set_visible(False)
        return

    # Rectangular path
    grid_shape = _grid_shape_from_labels(electrodes.labels)
    if grid_shape is not None:
        rows, cols = grid_shape
        Z = val.reshape(rows, cols)
        im = ax.imshow(
            Z, cmap=divergent_cmap(), vmin=vmin, vmax=vmax,
            aspect="equal", origin="lower", interpolation="bicubic",
        )
        ax.set_xlabel(f"Column  ·  {cols} contacts @ {pitch:.1f} mm pitch")
        ax.set_ylabel(f"Row  ·  {rows} contacts")
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.85, fraction=0.04, pad=0.03)
        cb.set_label("µV  (1 nA·m source)", fontsize=8)
        cb.outline.set_visible(False)
        return

    im = ax.scatter(
        electrodes.coilpos[:, 0], electrodes.coilpos[:, 2],
        c=val, cmap=divergent_cmap(), s=80, vmin=vmin, vmax=vmax,
        edgecolor=NATURE_PALETTE["axis"], linewidths=0.3,
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_aspect("equal")
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.85, fraction=0.04, pad=0.03)
    cb.set_label("µV  (1 nA·m source)", fontsize=8)
    cb.outline.set_visible(False)


def _format_3d_axis(ax, *, src: np.ndarray, pos_for_lim: np.ndarray | None = None) -> None:
    """Trim chrome from a 3-D matplotlib axis and centre on the cervical region."""
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("white")
        axis.set_tick_params(labelsize=6.5, pad=-2)
    ax.set_xlabel("X (mm)", labelpad=-3)
    ax.set_ylabel("Y (mm)", labelpad=-3)
    ax.set_zlabel("Z (mm)", labelpad=-3)
    z_lo, z_hi = src[2] - 220.0, src[2] + 220.0
    if pos_for_lim is not None and len(pos_for_lim):
        ax.set_xlim(pos_for_lim[:, 0].min() - 30, pos_for_lim[:, 0].max() + 30)
        ax.set_ylim(pos_for_lim[:, 1].min() - 30, pos_for_lim[:, 1].max() + 30)
    ax.set_zlim(z_lo, z_hi)
    ax.view_init(elev=14, azim=42)


# ── single-modality panels (also used by the dual figure) ──────────────────

def render_meg_topoplot(
    cfg: Config, *, source_idx: int = -1, ax=None,
    show_skin: bool = True, title: str | None = None,
    moment: str = DEFAULT_MOMENT, panel_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """3-D MEG topoplot for a single source position. Returns (fig, ax)."""
    apply_nature_style()
    lf = load_leadfield(cfg.outputs.forward_npz)
    if source_idx < 0:
        source_idx = lf.source_pos.shape[0] // 2

    radial = _radial_channel_mask(list(lf.channel_names))
    L_chan = _column_for_source(lf.L_fT_per_nAm, source_idx, moment=moment)
    pos = lf.coil_pos[radial]
    val = L_chan[radial]
    src = lf.source_pos[source_idx]

    # Crop the view to a ±220 mm Z slab centred on the source.
    keep = (pos[:, 2] >= src[2] - 220.0) & (pos[:, 2] <= src[2] + 220.0)
    pos = pos[keep]
    val = val[keep]

    if ax is None:
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if show_skin:
        try:
            geom = load_geometry(cfg.outputs.geometry_mat)
            skin = geom.compartments.get("mesh_skin")
            if skin is not None:
                _draw_skin_3d(ax, skin.vertices, skin.faces, alpha=0.05)
        except Exception as e:
            logger.debug("skin draw skipped: %s", e)

    vmin, vmax = divergent_norm(val)
    sc = ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=val, cmap=divergent_cmap(), s=44, vmin=vmin, vmax=vmax,
        edgecolor=NATURE_PALETTE["axis"], linewidths=0.25, depthshade=False,
    )
    ax.scatter(
        [src[0]], [src[1]], [src[2]],
        s=160, c=NATURE_PALETTE["glow"],
        edgecolor=NATURE_PALETTE["axis"], linewidths=0.8, marker="*",
    )
    _format_3d_axis(ax, src=src, pos_for_lim=pos)
    ax.set_title(title or "MEG  ·  radial OPM coils")
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.06, fraction=0.04)
    cb.set_label("fT  (1 nA·m source)", fontsize=8)
    cb.outline.set_visible(False)
    if panel_label:
        add_panel_label(ax, panel_label)
    return fig, ax


def render_eeg_topoplot(
    cfg: Config, *, source_idx: int = -1, figure: plt.Figure | None = None,
    with_3d_context: bool = True, title: str | None = None,
    moment: str = DEFAULT_MOMENT,
) -> plt.Figure:
    """Render an EEG topoplot: 2-D grid heatmap + optional 3-D context."""
    apply_nature_style()
    lf = load_leadfield(cfg.outputs.forward_eeg_npz)
    electrodes = load_sensors(cfg.outputs.electrodes_mat)
    if source_idx < 0:
        source_idx = lf.source_pos.shape[0] // 2
    val = _column_for_source(lf.L_fT_per_nAm, source_idx, moment=moment)
    src = lf.source_pos[source_idx]

    if figure is None:
        figure = plt.figure(figsize=(11, 5)) if with_3d_context else plt.figure(figsize=(5.5, 5))

    if with_3d_context:
        gs = GridSpec(1, 2, figure=figure, width_ratios=[1.1, 1.0],
                      left=0.05, right=0.96, top=0.92, bottom=0.10, wspace=0.25)
        ax2d = figure.add_subplot(gs[0, 0])
        ax3d = figure.add_subplot(gs[0, 1], projection="3d")
    else:
        ax2d = figure.add_subplot(111)
        ax3d = None

    vmin, vmax = divergent_norm(val)
    _draw_eeg_2d_topoplot(ax2d, electrodes, val, cfg, vmin=vmin, vmax=vmax)
    ax2d.set_title(title or "EEG  ·  HD PEDOT:PSS patch")

    if ax3d is not None:
        try:
            geom = load_geometry(cfg.outputs.geometry_mat)
            skin = geom.compartments.get("mesh_skin")
            if skin is not None:
                _draw_skin_3d(ax3d, skin.vertices, skin.faces, alpha=0.04)
        except Exception:
            pass
        ax3d.scatter(
            electrodes.coilpos[:, 0], electrodes.coilpos[:, 1], electrodes.coilpos[:, 2],
            c=val, cmap=divergent_cmap(), vmin=vmin, vmax=vmax,
            s=24, edgecolor=NATURE_PALETTE["axis"], linewidths=0.25,
            depthshade=False,
        )
        ax3d.scatter(
            [src[0]], [src[1]], [src[2]],
            s=130, c=NATURE_PALETTE["glow"],
            edgecolor=NATURE_PALETTE["axis"], linewidths=0.8, marker="*",
        )
        _format_3d_axis(ax3d, src=src, pos_for_lim=electrodes.coilpos)
        ax3d.set_title("Patch placement on neck")

    return figure


# ── the headline dual figure ───────────────────────────────────────────────

def render_dual_topoplot(
    cfg: Config, *, source_idx: int = -1, out_path: Path | None = None,
    dpi: int = 300, moment: str = DEFAULT_MOMENT,
) -> Path:
    """Nature Reviews-styled dual MEG/EEG figure (4 panels, single source).

    Panel layout
    ------------
        a  Anatomy + sensor placement context
        b  MEG topoplot (3-D OPM array, longitudinal moment)
        c  EEG topoplot (HD-EMG patch heatmap, longitudinal moment)
        d  Channel amplitude distributions (MEG | EEG)
    """
    apply_nature_style()

    meg_lf = load_leadfield(cfg.outputs.forward_npz)
    eeg_lf = load_leadfield(cfg.outputs.forward_eeg_npz)
    sensors = load_sensors(cfg.outputs.sensors_mat)
    electrodes = load_sensors(cfg.outputs.electrodes_mat)

    if source_idx < 0:
        source_idx = meg_lf.source_pos.shape[0] // 2
    src = meg_lf.source_pos[source_idx]

    radial = _radial_channel_mask(list(meg_lf.channel_names))
    meg_val = _column_for_source(meg_lf.L_fT_per_nAm, source_idx, moment=moment)[radial]
    meg_pos = meg_lf.coil_pos[radial]
    keep = (meg_pos[:, 2] >= src[2] - 220.0) & (meg_pos[:, 2] <= src[2] + 220.0)
    meg_pos = meg_pos[keep]
    meg_val = meg_val[keep]

    eeg_val = _column_for_source(eeg_lf.L_fT_per_nAm, source_idx, moment=moment)

    # ── figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 11.5))
    gs = GridSpec(
        2, 2, figure=fig,
        left=0.04, right=0.97, top=0.93, bottom=0.06,
        hspace=0.30, wspace=0.20,
    )

    # ── panel a: anatomy + sensor placement context ────────────────────────
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    try:
        geom = load_geometry(cfg.outputs.geometry_mat)
        skin = geom.compartments.get("mesh_skin")
        if skin is not None:
            _draw_skin_3d(ax_a, skin.vertices, skin.faces, alpha=0.05)
        vagus = geom.compartments.get("mesh_vagus_left")
        if vagus is not None:
            _draw_skin_3d(ax_a, vagus.vertices, vagus.faces, alpha=0.65,
                          max_tris=12_000)
    except Exception:
        pass
    ax_a.scatter(
        sensors.coilpos[:: 3, 0], sensors.coilpos[:: 3, 1], sensors.coilpos[:: 3, 2],
        c=NATURE_PALETTE["blue"], s=4, alpha=0.5, depthshade=False,
        label=f"OPM array (n={len(sensors.coilpos) // 3})",
    )
    ax_a.scatter(
        electrodes.coilpos[:, 0], electrodes.coilpos[:, 1], electrodes.coilpos[:, 2],
        c=NATURE_PALETTE["red"], s=10, depthshade=False,
        edgecolor=NATURE_PALETTE["axis"], linewidths=0.2,
        label=f"HD-EMG patch (n={len(electrodes.coilpos)})",
    )
    # All vagus sources along the polyline
    ax_a.plot(
        meg_lf.source_pos[:, 0], meg_lf.source_pos[:, 1], meg_lf.source_pos[:, 2],
        c=NATURE_PALETTE["axis"], lw=0.6, alpha=0.5,
    )
    ax_a.scatter(
        [src[0]], [src[1]], [src[2]],
        s=180, c=NATURE_PALETTE["glow"],
        edgecolor=NATURE_PALETTE["axis"], linewidths=0.8, marker="*",
        label="Source", depthshade=False,
    )
    _format_3d_axis(ax_a, src=src, pos_for_lim=sensors.coilpos[:: 3])
    ax_a.set_title("Anatomy and sensor placement")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=7,
                handletextpad=0.5, borderpad=0.3, labelspacing=0.4)
    add_panel_label(ax_a, "a")

    # ── panel b: MEG topoplot ──────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    try:
        if skin is not None:
            _draw_skin_3d(ax_b, skin.vertices, skin.faces, alpha=0.05)
    except Exception:
        pass
    vmin_b, vmax_b = divergent_norm(meg_val)
    sc_b = ax_b.scatter(
        meg_pos[:, 0], meg_pos[:, 1], meg_pos[:, 2],
        c=meg_val, cmap=divergent_cmap(), s=46, vmin=vmin_b, vmax=vmax_b,
        edgecolor=NATURE_PALETTE["axis"], linewidths=0.25, depthshade=False,
    )
    ax_b.scatter(
        [src[0]], [src[1]], [src[2]],
        s=160, c=NATURE_PALETTE["glow"],
        edgecolor=NATURE_PALETTE["axis"], linewidths=0.8, marker="*",
    )
    _format_3d_axis(ax_b, src=src, pos_for_lim=meg_pos)
    ax_b.set_title("MEG topoplot  ·  radial OPM coils")
    cb_b = fig.colorbar(sc_b, ax=ax_b, shrink=0.55, pad=0.06, fraction=0.04)
    cb_b.set_label("fT  (1 nA·m source)", fontsize=8)
    cb_b.outline.set_visible(False)
    add_panel_label(ax_b, "b")

    # ── panel c: EEG topoplot ──────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    vmin_c, vmax_c = divergent_norm(eeg_val)
    _draw_eeg_2d_topoplot(ax_c, electrodes, eeg_val, cfg, vmin=vmin_c, vmax=vmax_c)
    ax_c.set_title(f"EEG topoplot  ·  PEDOT:PSS paddle  ·  {len(eeg_val)} contacts")
    add_panel_label(ax_c, "c")

    # ── panel d: channel-amplitude distributions ───────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    bins = 60
    meg_norm = meg_val / np.abs(meg_val).max() if np.abs(meg_val).max() else meg_val
    eeg_norm = eeg_val / np.abs(eeg_val).max() if np.abs(eeg_val).max() else eeg_val
    ax_d.hist(meg_norm, bins=bins, color=NATURE_PALETTE["blue"], alpha=0.65,
              label=f"MEG  ·  peak |L| = {_fmt_peak(np.abs(meg_val).max())} fT")
    ax_d.hist(eeg_norm, bins=bins, color=NATURE_PALETTE["red"], alpha=0.65,
              label=f"EEG  ·  peak |L| = {_fmt_peak(np.abs(eeg_val).max())} µV")
    ax_d.axvline(0, color=NATURE_PALETTE["axis"], lw=0.6)
    ax_d.set_xlabel("Channel amplitude  /  modality peak")
    ax_d.set_ylabel("Sensor count")
    ax_d.set_title("Channel-amplitude distributions  ·  normalised")
    ax_d.legend(loc="upper left", fontsize=7.5, handlelength=1.2)
    add_panel_label(ax_d, "d")

    fig.suptitle(
        f"Dual-modality forward model of a single {source_region_label(cfg)} source "
        f"(z = {src[2]:.0f} mm, longitudinal moment)",
        fontsize=11, fontweight="bold", y=0.985,
    )

    _tag = source_region_label(cfg).replace(" + ", "_").replace(" ", "_")
    out = out_path or cfg.outputs.base / f"dual_topoplot_{_tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    logger.info("[saved] %s", out)
    plt.close(fig)
    return out


def render_meg_montage(
    cfg: Config, *, n_sources: int = 5, out_path: Path | None = None, dpi: int = 220,
) -> Path:
    """MEG montage: ``n_sources`` evenly-spaced source positions along the vagus."""
    apply_nature_style()
    lf = load_leadfield(cfg.outputs.forward_npz)
    n_total = lf.source_pos.shape[0]
    indices = np.linspace(0, n_total - 1, n_sources).round().astype(int)

    fig = plt.figure(figsize=(3.4 * n_sources, 4.5))
    panel_letters = "abcdefghij"
    for k, idx in enumerate(indices):
        ax = fig.add_subplot(1, n_sources, k + 1, projection="3d")
        render_meg_topoplot(
            cfg, source_idx=int(idx), ax=ax, show_skin=True,
            title=f"z = {lf.source_pos[idx, 2]:.0f} mm",
            panel_label=panel_letters[k] if k < len(panel_letters) else None,
        )
    fig.suptitle(
        f"MEG topoplot montage along the {source_region_label(cfg)}",
        fontsize=11, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _tag = source_region_label(cfg).replace(" + ", "_").replace(" ", "_")
    out = out_path or cfg.outputs.base / f"meg_topoplot_montage_{_tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    logger.info("[saved] %s", out)
    plt.close(fig)
    return out
