"""Re-reference an electrode leadfield after the fact.

The saved electrode leadfield is common-average referenced (see
:mod:`inob.forward.eeg`). A surface potential is only defined up to a
per-source additive constant, so the reference is a choice about the readout,
not about the physics, and changing it is linear on the leadfield rows:

    L_ref[:, cols(s)] = L[:, cols(s)] - w(s) @ L[:, cols(s)]

where ``w(s)`` is a weight vector over electrodes summing to one — all equal
for the average, a single 1 for an electrode reference. Because the stored
leadfield differs from the raw potentials by a per-column constant, and any
such ``w`` removes that constant exactly, every reference built from the
array's own contacts is recovered without re-solving.

What this cannot do is reference to a point that is not an electrode (the nose,
a mastoid, a distant limb) on a patch that has none there: that potential was
never solved. The nearest-to-a-point option picks the array contact closest to
such a point, and the result says how far away that contact actually is.

Reference-free readouts are unaffected by any of this. A bipolar difference
between two contacts subtracts the same constant from both terms, so
:func:`inob.analysis.snr.per_source_best_bipolar` returns identical numbers
under every reference here — and, since a referenced channel is itself a
difference of two array potentials (or of one and an average), it is an upper
bound on the referenced single-channel peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Reference kinds :func:`parse_reference` understands.
REFERENCE_KINDS = ("average", "electrode", "farthest", "nearest")


@dataclass(frozen=True)
class EegReference:
    """A parsed reference choice.

    ``kind`` is one of :data:`REFERENCE_KINDS`. ``index`` is the electrode row
    for ``electrode``; ``point_mm`` the (x, y, z) target for ``nearest``.
    """

    kind: str = "average"
    index: int | None = None
    point_mm: tuple[float, float, float] | None = None

    @property
    def label(self) -> str:
        """The spec string that parses back to this reference."""
        if self.kind == "electrode":
            return f"electrode:{self.index}"
        if self.kind == "nearest" and self.point_mm is not None:
            return "nearest:" + ",".join(f"{v:g}" for v in self.point_mm)
        return self.kind


def parse_reference(spec: str | EegReference | None) -> EegReference:
    """Parse ``average``, ``electrode:N``, ``farthest`` or ``nearest:x,y,z``.

    ``None`` is the stored common average. A bare integer is taken as an
    electrode index, so ``--eeg-reference 5`` and ``electrode:5`` agree.
    """
    if spec is None:
        return EegReference()
    if isinstance(spec, EegReference):
        return spec
    text = str(spec).strip().lower()
    kind, _, arg = text.partition(":")
    if kind in ("average", "avg", "car") and not arg:
        return EegReference("average")
    if kind == "farthest" and not arg:
        return EegReference("farthest")
    if kind.isdigit() and not arg:
        return EegReference("electrode", index=int(kind))
    if kind == "electrode" and arg.isdigit():
        return EegReference("electrode", index=int(arg))
    if kind == "nearest" and arg:
        try:
            pt = tuple(float(v) for v in arg.split(","))
        except ValueError:
            pt = ()
        if len(pt) == 3 and all(np.isfinite(pt)):
            return EegReference("nearest", point_mm=pt)  # type: ignore[arg-type]
    raise ValueError(
        f"reference {spec!r}: use average, electrode:N, farthest, or nearest:x,y,z (mm)"
    )


def reference_electrodes(
    ref: EegReference,
    electrode_pos: np.ndarray,
    source_pos: np.ndarray,
) -> np.ndarray | None:
    """Reference electrode row per source, shape (S,), or ``None`` for average.

    ``farthest`` is chosen per source — the contact with the greatest
    Euclidean distance from that source — so it can differ between sources.
    The others are one row for all sources.
    """
    ref = parse_reference(ref)
    elec = np.asarray(electrode_pos, dtype=float)
    src = np.asarray(source_pos, dtype=float)
    n_src = src.shape[0]
    if ref.kind == "average":
        return None
    if ref.kind == "electrode":
        assert ref.index is not None
        if not 0 <= ref.index < elec.shape[0]:
            raise ValueError(
                f"reference electrode {ref.index} out of range (0..{elec.shape[0] - 1})"
            )
        return np.full(n_src, ref.index, dtype=int)
    if ref.kind == "nearest":
        assert ref.point_mm is not None
        d = np.linalg.norm(elec - np.asarray(ref.point_mm), axis=1)
        return np.full(n_src, int(np.argmin(d)), dtype=int)
    if ref.kind == "farthest":
        d = np.linalg.norm(elec[:, None, :] - src[None, :, :], axis=2)  # (C, S)
        return np.argmax(d, axis=0).astype(int)
    raise ValueError(f"unknown reference kind {ref.kind!r}")


def rereference(
    L: np.ndarray,
    ref: EegReference | str | None,
    *,
    electrode_pos: np.ndarray,
    source_pos: np.ndarray,
) -> np.ndarray:
    """Return ``L`` (C, 3S) re-referenced to ``ref``; the input is not modified.

    Linear on the rows: each source's three columns have the reference
    potential (one row, or the row mean) subtracted. Re-applying ``average``
    to an already average-referenced leadfield is the identity.
    """
    L = np.asarray(L, dtype=float)
    C, three_S = L.shape
    if three_S % 3 != 0:
        raise ValueError(f"L second dim {three_S} not divisible by 3")
    if np.asarray(electrode_pos).shape[0] != C:
        raise ValueError(f"{np.asarray(electrode_pos).shape[0]} electrode positions for {C} rows")
    if np.asarray(source_pos).shape[0] != three_S // 3:
        raise ValueError(
            f"{np.asarray(source_pos).shape[0]} source positions for {three_S // 3} sources"
        )
    rows = reference_electrodes(parse_reference(ref), electrode_pos, source_pos)
    if rows is None:
        return L - L.mean(axis=0, keepdims=True)
    cols_ref = L[np.repeat(rows, 3), np.arange(three_S)]  # (3S,)
    return L - cols_ref[None, :]
