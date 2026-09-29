"""What a magnetometer array's channel labels say about its layout.

A generated array is written in one of two layouts, and the labels are
the record of which:

``triaxial``
    Every position carries three coils, stored in blocks — all N ``-R``
    channels, then all N ``-T1``, then all N ``-T2`` — so position ``i``
    is channels ``i``, ``N + i`` and ``2N + i``.

``single_axis``
    One ``-R`` coil per position (``sensors.axes: radial``, or an imported
    array with one coil at each position).

Anything else — an imported dual-axis array, or one whose positions carry
different numbers of coils — is ``other``: every channel is still a valid
forward-model output, but nothing may assume a fixed number of coils per
position. Code that needs the block layout asks :func:`require_triaxial`
instead of dividing the channel count by three, so a single-axis array
fails with a sentence rather than silently reading the wrong channels.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

TRIAXIAL = "triaxial"
SINGLE_AXIS = "single_axis"
OTHER = "other"


def array_layout(labels: Sequence[str]) -> str:
    """Classify a channel list as ``triaxial``, ``single_axis`` or ``other``."""
    labels = [str(lab) for lab in labels]
    n = len(labels)
    if n == 0:
        return OTHER
    if n % 3 == 0:
        k = n // 3
        blocks = (labels[:k], labels[k : 2 * k], labels[2 * k :])
        if all(
            all(lab.endswith(suffix) for lab in block)
            for block, suffix in zip(blocks, ("-R", "-T1", "-T2"), strict=True)
        ):
            return TRIAXIAL
    if all(lab.endswith("-R") for lab in labels):
        return SINGLE_AXIS
    return OTHER


def radial_channels(labels: Sequence[str]) -> np.ndarray:
    """Indices of the ``-R`` channels, one per position, in position order.

    Raises when there are none, which is the case for an imported array
    whose coils were not labelled radial (see ``inob sensors --help``).
    """
    idx = np.array([i for i, lab in enumerate(labels) if str(lab).endswith("-R")], dtype=int)
    if idx.size == 0:
        raise ValueError(
            "this sensor array has no radial (-R) channels, so there is no "
            "single field component to map; regenerate it with "
            "`inob sensors` or import one with one or three coils per position"
        )
    return idx


def require_triaxial(labels: Sequence[str], what: str) -> int:
    """Number of positions of a triaxial block-layout array, else raise.

    ``what`` names the analysis in the message, e.g. "placement comparison".
    """
    layout = array_layout(labels)
    if layout != TRIAXIAL:
        described = {
            SINGLE_AXIS: "a single-axis array (one coil per position)",
            OTHER: "an array without three coils at every position",
        }[layout]
        raise ValueError(
            f"{what} needs a triaxial array (R, T1 and T2 at every position); "
            f"this leadfield was solved on {described}. Regenerate the array "
            "with sensors.axes=triaxial and re-run the forward solve."
        )
    return len(labels) // 3


def position_channels(labels: Sequence[str], coil_pos: np.ndarray) -> np.ndarray:
    """One channel index per distinct sensor position, in position order.

    For the two generated layouts this is the ``-R`` block; for anything
    else it is the first channel found at each position.
    """
    layout = array_layout(labels)
    if layout == TRIAXIAL:
        return np.arange(len(labels) // 3)
    if layout == SINGLE_AXIS:
        return np.arange(len(labels))
    pos = np.round(np.asarray(coil_pos, dtype=float), 3)
    _, first = np.unique(pos, axis=0, return_index=True)
    return np.sort(first)
