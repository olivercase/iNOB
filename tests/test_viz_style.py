"""Tests for inob.viz.style (palette, cmaps, rcParams, panel labels)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
    divergent_cmap,
    divergent_norm,
    sequential_cmap,
)


def test_palette_hex_colours() -> None:
    assert set(NATURE_PALETTE) >= {"red", "blue", "skin", "axis", "panel_bg"}
    for name, hexcode in NATURE_PALETTE.items():
        assert hexcode.startswith("#"), name
        assert len(hexcode) == 7, name


def test_divergent_cmap_endpoints() -> None:
    cmap = divergent_cmap()
    assert isinstance(cmap, LinearSegmentedColormap)
    lo = np.array(cmap(0.0))
    hi = np.array(cmap(1.0))
    mid = np.array(cmap(0.5))
    # blue at 0, red at 1, white-ish in the middle.
    assert lo[2] > lo[0]  # more blue than red channel
    assert hi[0] > hi[2]  # more red than blue channel
    assert mid[0] > 0.9 and mid[1] > 0.9 and mid[2] > 0.9


def test_sequential_cmap_is_distinct_object() -> None:
    seq = sequential_cmap()
    div = divergent_cmap()
    assert isinstance(seq, LinearSegmentedColormap)
    assert seq.name != div.name
    lo = np.array(seq(0.0))
    hi = np.array(seq(1.0))
    assert not np.allclose(lo, hi)


def test_apply_nature_style_sets_rcparams() -> None:
    apply_nature_style()
    assert matplotlib.rcParams["font.size"] == 8.0
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.spines.right"] is False
    assert matplotlib.rcParams["axes.grid"] is False
    assert matplotlib.rcParams["legend.frameon"] is False


def test_apply_nature_style_idempotent() -> None:
    apply_nature_style()
    first = dict(matplotlib.rcParams)
    apply_nature_style()
    second = dict(matplotlib.rcParams)
    assert first["font.size"] == second["font.size"]
    assert first["axes.labelcolor"] == second["axes.labelcolor"]


def test_add_panel_label_2d_axes() -> None:
    fig, ax = plt.subplots()
    try:
        add_panel_label(ax, "a")
        texts = [t.get_text() for t in ax.texts]
        assert "a" in texts
        panel_text = ax.texts[-1]
        assert panel_text.get_fontweight() == "bold"
    finally:
        plt.close(fig)


def test_add_panel_label_custom_position() -> None:
    fig, ax = plt.subplots()
    try:
        add_panel_label(ax, "b", x=0.1, y=0.9)
        panel_text = ax.texts[-1]
        assert panel_text.get_position() == (0.1, 0.9)
    finally:
        plt.close(fig)


def test_add_panel_label_3d_axes_uses_text2d() -> None:
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        add_panel_label(ax, "c")
        # text2D-created annotations are still tracked in ax.texts
        texts = [t.get_text() for t in ax.texts]
        assert "c" in texts
    finally:
        plt.close(fig)


def test_divergent_norm_empty() -> None:
    assert divergent_norm(np.array([])) == (-1.0, 1.0)


def test_divergent_norm_symmetric() -> None:
    values = np.array([-3.0, 1.0, 2.5])
    lo, hi = divergent_norm(values)
    assert lo == -3.0
    assert hi == 3.0


def test_divergent_norm_all_zero() -> None:
    lo, hi = divergent_norm(np.zeros(5))
    assert lo == -1.0
    assert hi == 1.0
