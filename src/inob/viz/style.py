"""Nature Reviews-inspired figure style.

A small, opinionated styling layer:

  * `NATURE_PALETTE`            — accent + background colours (hex).
  * `divergent_cmap()`          — blue-white-red cmap built from the palette.
  * `apply_nature_style()`      — set matplotlib rcParams (8 pt body, no grid,
                                    minor-tick chrome stripped).
  * `add_panel_label(ax, "a")`  — bold panel labels in the top-left, the
                                    Nature Reviews convention.
  * `save_figure(fig, path)`    — mkdir + savefig + log + close, in one place.

Design principles (from the Nature Reviews "Guide to designing figures"):

  Hierarchy   most important elements highly saturated; context neutral.
  Clarity     label first instances; no ambiguous arrows; minimal chrome.
  Accessibility  blue/red divergent (never red/green); 8 pt minimum text;
                 black text where possible.
  Consistency      same colours mean the same thing across figures.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

# Backends that can render without a window server, so are safe on any thread.
_NON_INTERACTIVE_BACKENDS = frozenset(
    {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
    }
)

NATURE_PALETTE: dict[str, str] = {
    # neutral context
    "skin": "#E8DCC4",  # body silhouette / muscle / fat
    "stone": "#C9C2B5",
    "grey": "#8A8A8A",
    "panel_bg": "#FFFFFF",
    "axis": "#1A1A1A",
    # main accents
    "red": "#C0392B",  # positive lobe
    "blue": "#2C4A78",  # negative lobe
    "glow": "#F2B33C",  # source / focal element
    # extended palette for categorical (use sparingly)
    "olive": "#7A8C3A",
    "teal": "#3A7A78",
    "purple": "#6A4A8A",
    "orange": "#D46B2A",
}


def divergent_cmap() -> LinearSegmentedColormap:
    """Blue → white → red divergent map using Nature's accent reds/blues."""
    return LinearSegmentedColormap.from_list(
        "nature_div",
        [NATURE_PALETTE["blue"], "#FFFFFF", NATURE_PALETTE["red"]],
        N=256,
    )


def sequential_cmap() -> LinearSegmentedColormap:
    """Stone → red sequential map for amplitude magnitudes."""
    return LinearSegmentedColormap.from_list(
        "nature_seq",
        ["#FFFFFF", NATURE_PALETTE["stone"], NATURE_PALETTE["red"]],
        N=256,
    )


def ensure_headless_backend() -> None:
    """Force a non-interactive backend when rendering off the main thread.

    Every figure this package produces is written straight to a PNG, so a GUI
    backend is never needed. It is actively harmful off-thread: macOS' default
    backend raises "Cannot create a GUI FigureManager outside the main thread",
    which broke the visualisation stage of every run driven from the browser
    GUI (which solves on a worker thread) while the identical CLI run passed.

    Only switches when both conditions hold — an interactive backend AND a
    non-main thread — so an interactive session keeps whatever it chose.
    """
    if threading.current_thread() is threading.main_thread():
        return
    backend = mpl.get_backend().lower()
    if backend in _NON_INTERACTIVE_BACKENDS:
        return
    logger.debug("switching matplotlib backend %s → Agg (off-main-thread render)", backend)
    mpl.use("Agg", force=True)


def apply_nature_style() -> None:
    """Install Nature-leaning matplotlib rcParams. Idempotent."""
    ensure_headless_backend()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "axes.labelcolor": NATURE_PALETTE["axis"],
            "axes.edgecolor": NATURE_PALETTE["axis"],
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.titlepad": 6.0,
            "xtick.color": NATURE_PALETTE["axis"],
            "ytick.color": NATURE_PALETTE["axis"],
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "figure.facecolor": NATURE_PALETTE["panel_bg"],
            "axes.facecolor": NATURE_PALETTE["panel_bg"],
            "savefig.facecolor": NATURE_PALETTE["panel_bg"],
            "savefig.dpi": 300,
        }
    )


def add_panel_label(ax: Any, label: str, *, x: float = -0.04, y: float = 1.04) -> None:
    """Add a bold panel label (a, b, c, …) to an axes, in Nature style.

    Works for both 2-D and 3-D axes (delegates to ``text2D`` for the latter).
    """
    text_fn = getattr(ax, "text2D", ax.text)
    text_fn(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=NATURE_PALETTE["axis"],
    )


def save_figure(fig: Any, out: Any, *, dpi: int = 300) -> Any:
    """Write ``fig`` to ``out``, log it, close it, and return the path.

    Every render_* function ends with the same four lines; keeping them here
    means the output directory is always created and the figure is always
    closed (matplotlib leaks figures otherwise in long pipeline runs).
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    logger.info("[saved] %s", out)
    plt.close(fig)
    return out


def divergent_norm(values: np.ndarray, *, pct_clip: float | None = None) -> tuple[float, float]:
    """Return (vmin, vmax) symmetric about zero for a divergent colour map.

    ``pct_clip`` (0-100), if given, scales to that percentile of ``|values|``
    instead of the exact max. A few near-field-dominated outliers (e.g. one
    source a few mm from a sensor) can otherwise stretch the scale so far
    that every other value renders as white — clipping lets a handful of
    outliers saturate at the colour extreme instead of washing out the rest.
    Default (``None``) keeps the exact abs-max behaviour used everywhere today.
    """
    if values.size == 0:
        return -1.0, 1.0
    v = (
        float(np.max(np.abs(values)))
        if pct_clip is None
        else float(np.percentile(np.abs(values), pct_clip))
    )
    v = v if v > 0 else 1.0
    return -v, v
