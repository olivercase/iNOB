"""The generalised harmonic basis must span exactly what the pack's tabulation does.

``cfa_hfc_basis.m`` hand-writes the degree-1/2/3 harmonic polynomials. This
module builds them instead, as the null space of the Laplacian on homogeneous
polynomials, which removes the order-3 cap. Only the SPAN is used -- the basis is
projected out -- so the test compares PROJECTORS, not columns: any two bases of
the same subspace give the same projector, and that is the invariant that matters.
"""
from __future__ import annotations

import numpy as np
import pytest

from inob.cfa.hfc import harmonic_basis, harmonic_projector, n_terms


def _matlab_basis(chanpos: np.ndarray, chanori: np.ndarray, order: int) -> np.ndarray:
    """Verbatim transcription of the gradients tabulated in cfa_hfc_basis.m."""
    q = chanpos - chanpos.mean(0)
    q = q / (np.linalg.norm(q, axis=1).max() or 1.0)
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    n = len(q)
    one, nil = np.ones(n), np.zeros(n)
    g = [np.stack([one, nil, nil], 1), np.stack([nil, one, nil], 1),
         np.stack([nil, nil, one], 1)]
    if order >= 2:
        g += [np.stack([y, x, nil], 1), np.stack([nil, z, y], 1),
              np.stack([z, nil, x], 1), np.stack([2 * x, -2 * y, nil], 1),
              np.stack([-2 * x, -2 * y, 4 * z], 1)]
    if order >= 3:
        g += [
            np.stack([-6 * x * z, -6 * y * z, 6 * z**2 - 3 * (x**2 + y**2)], 1),
            np.stack([4 * z**2 - 3 * x**2 - y**2, -2 * x * y, 8 * x * z], 1),
            np.stack([-2 * x * y, 4 * z**2 - x**2 - 3 * y**2, 8 * y * z], 1),
            np.stack([2 * x * z, -2 * y * z, x**2 - y**2], 1),
            np.stack([y * z, x * z, x * y], 1),
            np.stack([3 * (x**2 - y**2), -6 * x * y, nil], 1),
            np.stack([6 * x * y, 3 * (x**2 - y**2), nil], 1),
        ]
    b = np.stack([np.einsum("ij,ij->i", chanori, gi) for gi in g], axis=1)
    return b / np.maximum(np.linalg.norm(b, axis=0), np.finfo(float).eps)


@pytest.fixture
def array():
    rng = np.random.default_rng(3)
    pos = rng.normal(size=(120, 3))
    pos = 100.0 * pos / np.linalg.norm(pos, axis=1, keepdims=True)
    ori = rng.normal(size=(120, 3))
    return pos, ori / np.linalg.norm(ori, axis=1, keepdims=True)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_spans_the_same_subspace_as_the_pack(array, order):
    pos, ori = array
    ours = harmonic_basis(pos, ori, order)
    theirs = _matlab_basis(pos, ori, order)
    proj = lambda b: b @ np.linalg.pinv(b)
    assert np.allclose(proj(ours), proj(theirs), atol=1e-9)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
def test_dimension_and_rank(array, order):
    b = harmonic_basis(*array, order)
    assert b.shape[1] == n_terms(order) == order * (order + 2)
    assert np.linalg.matrix_rank(b) == b.shape[1]


def test_columns_are_harmonic_gradients(array):
    """Each basis field must be curl- and divergence-free: it is grad of a harmonic."""
    pos, _ = array
    eye = np.repeat(np.eye(3), len(pos), axis=0)
    p3 = np.tile(pos, (3, 1))
    b = harmonic_basis(p3, eye, 3)                 # world x/y/z components
    f = b.reshape(3, len(pos), -1)                 # (component, point, term)
    h = 1e-3 * float(np.linalg.norm(pos, axis=1).max())
    div = np.zeros((len(pos), f.shape[2]))
    for c in range(3):
        step = np.zeros(3); step[c] = h
        plus = harmonic_basis(np.tile(pos + step, (3, 1)), eye, 3).reshape(3, len(pos), -1)
        minus = harmonic_basis(np.tile(pos - step, (3, 1)), eye, 3).reshape(3, len(pos), -1)
        div += (plus[c] - minus[c]) / (2 * h)
    scale = np.abs(f).max()
    assert np.abs(div).max() < 1e-4 * scale / h * h, "basis fields must be divergence-free"


def test_projector_is_an_orthogonal_projection(array):
    p = harmonic_projector(*array, 3)
    assert np.allclose(p @ p, p, atol=1e-9)
    assert np.allclose(p, p.T, atol=1e-12)


def test_rejects_bad_order(array):
    with pytest.raises(ValueError):
        harmonic_basis(*array, 0)
