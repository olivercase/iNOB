"""Harmonic field correction basis -- the regressor for sources OUTSIDE the array.

Port of ``cfa_hfc_basis.m``, generalised to **any** order.

The MATLAB original hard-codes the degree-1, 2 and 3 harmonic polynomials by
hand, which caps it at order 3 (15 terms). That cap was reasonable against 85
single-axis channels. Against a triaxial array's 255 channels it is not: order 5
is 35 regressors, comfortably conditioned, and the cardiac field's higher-degree
content is exactly what the near-field geometry leaves behind.

So the basis is built rather than tabulated. The harmonic polynomials of degree l
are the null space of the Laplacian restricted to homogeneous degree-l
polynomials -- a ``(l+1)(l+2)/2`` -dimensional monomial space mapped to degree
``l-2``, whose kernel has the required dimension ``2l+1``. Which orthogonal basis
of that kernel comes out is arbitrary and irrelevant: only the SPAN is used, and
the projector ``I - B B^+`` is basis-independent. The construction is verified
against the hand-tabulated MATLAB polynomials at orders 1-3 by comparing
projectors, in ``tests/test_cfa_hfc.py``.

In the current-free region the sensors occupy, ``B = -grad(V)`` with V harmonic.
A source outside the array contributes only the REGULAR solid harmonics, so V is
a sum of harmonic polynomials of degree 1..L and an oriented magnetometer i
measures ``o_i . grad(psi_j)(p_i)``. Those are the columns. There are ``2l+1``
harmonics of degree l, so ``sum_{l=1..L} (2l+1) = L(L+2)`` columns: 3, 8, 15.

Order 1 is standard HFC (a uniform field), which is what ``spm_opm_hfc``
implements. Orders 2-3 are the AMM-family extension.

**Why order matters here.** The cardiac field is not far-field. The heart sits
~0.3 m from an array of ~0.1 m radius, so the degree-l term falls only as
``(0.1/0.3)^l``: order 1 leaves a large fraction of the artefact behind. Order-1
HFC therefore cannot remove the CFA however well the array is calibrated -- it is
defeated by geometry before calibration is ever the binding constraint. Raising
the order is what makes precision the limiting factor, which is the regime the
2x2 exists to probe.

**And its cost.** The columns are removed from the data, so they are removed from
any brain signal lying in their span too. A near source (the insula HEF) is
mostly not in that span, but the overlap is non-zero and grows with order -- the
real trade-off, and why order is a reported parameter and not a hidden constant.
"""
from __future__ import annotations

import numpy as np

__all__ = ["harmonic_basis", "harmonic_projector", "n_terms"]


def n_terms(order: int) -> int:
    return order * (order + 2)


def _monomials(l: int) -> list[tuple[int, int, int]]:
    return [(a, b, l - a - b) for a in range(l + 1) for b in range(l - a + 1)]


def _harmonic_polynomials(l: int) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """Homogeneous harmonic polynomials of degree ``l``.

    Returns ``(monomials, coeffs)`` with ``coeffs`` of shape ``(2l+1, n_mono)``:
    the null space of the Laplacian on degree-l homogeneous polynomials.
    """
    mono = _monomials(l)
    if l < 2:
        return mono, np.eye(len(mono))
    lower = _monomials(l - 2)
    row = {m: i for i, m in enumerate(lower)}
    lap = np.zeros((len(lower), len(mono)))
    for j, (a, b, c) in enumerate(mono):
        for k, e in enumerate((a, b, c)):
            if e >= 2:
                m = [a, b, c]
                m[k] -= 2
                lap[row[tuple(m)], j] += e * (e - 1)
    _, sv, vh = np.linalg.svd(lap)
    null_dim = len(mono) - int((sv > sv.max() * 1e-10).sum())
    if null_dim != 2 * l + 1:
        raise RuntimeError(f"degree {l}: found {null_dim} harmonics, expected {2*l+1}")
    return mono, vh[len(mono) - null_dim:]


def _grad_at(mono, coeffs: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Gradients ``(n_poly, n_pts, 3)`` of each polynomial at points ``q``."""
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    out = np.zeros((len(coeffs), len(q), 3))
    for j, (a, b, c) in enumerate(mono):
        for axis, e in enumerate((a, b, c)):
            if e == 0:
                continue
            pw = [a, b, c]
            pw[axis] -= 1
            term = e * x ** pw[0] * y ** pw[1] * z ** pw[2]
            out[:, :, axis] += coeffs[:, j][:, None] * term[None, :]
    return out


def harmonic_basis(chanpos: np.ndarray, chanori: np.ndarray, order: int = 1,
                   origin: np.ndarray | None = None) -> np.ndarray:
    """``(n_chan, order*(order+2))`` design matrix, unit-norm columns."""
    if order < 1 or order != int(order):
        raise ValueError("order must be a positive integer")
    order = int(order)
    p = np.asarray(chanpos, float)
    o = np.asarray(chanori, float)
    org = p.mean(0) if origin is None else np.asarray(origin, float)
    q = p - org
    # Characteristic array radius: keeps the higher-degree columns O(1) before
    # the final normalisation, so the design stays conditioned across orders.
    r0 = float(np.linalg.norm(q, axis=1).max()) or 1.0
    q = q / r0
    cols = []
    for l in range(1, order + 1):
        mono, coeffs = _harmonic_polynomials(l)
        grads = _grad_at(mono, coeffs, q)                  # (2l+1, n_chan, 3)
        cols.append(np.einsum("pnc,nc->np", grads, o))
    b = np.concatenate(cols, axis=1)
    return b / np.maximum(np.linalg.norm(b, axis=0), np.finfo(float).eps)


def harmonic_projector(chanpos: np.ndarray, chanori: np.ndarray,
                       order: int = 1) -> np.ndarray:
    """``I - B B^+``: removes everything an external source can produce to ``order``."""
    b = harmonic_basis(chanpos, chanori, order)
    return np.eye(len(b)) - b @ np.linalg.pinv(b)
