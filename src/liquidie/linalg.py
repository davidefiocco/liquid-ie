"""Vectorised matrix operations on stacks of small matrices.

All functions operate on arrays of shape ``(N, n, n)`` where *N* is the
number of grid points and *n* is the number of species.  Specialised
fast paths exist for n=1 and n=2; the general case falls back to
``numpy.linalg.inv``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def invv(a: NDArray[np.floating]) -> NDArray[np.floating]:
    """Element-wise matrix inverse for a stack of (n x n) matrices.

    Parameters
    ----------
    a : array, shape (N, n, n)

    Returns
    -------
    Array of shape (N, n, n) containing the inverses.
    """
    n = a.shape[1]
    if n == 1:
        return 1.0 / a
    if n == 2:
        out = np.zeros_like(a)
        det = a[:, 0, 0] * a[:, 1, 1] - a[:, 1, 0] * a[:, 0, 1]
        out[:, 0, 0] = a[:, 1, 1] / det
        out[:, 0, 1] = -a[:, 0, 1] / det
        out[:, 1, 0] = -a[:, 1, 0] / det
        out[:, 1, 1] = a[:, 0, 0] / det
        return out
    return np.linalg.inv(a)


def dotve(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Batch matrix-matrix product for stacks of (n x n) matrices.

    Parameters
    ----------
    a, b : arrays of shape (N, n, n)

    Returns
    -------
    Array of shape (N, n, n) with ``result[q] = a[q] @ b[q]``.
    """
    return np.einsum("qij,qjk->qik", a, b)


def dotvbs(
    b: NDArray[np.floating],
    s: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Batch product of a full matrix with a diagonal (scalar-per-species) matrix.

    *b* has shape ``(N, n, n)``, *s* has shape ``(n, n)``.
    Computes ``b[q] @ s`` for each grid point *q*.
    """
    return np.einsum("qij,jk->qik", b, s)
