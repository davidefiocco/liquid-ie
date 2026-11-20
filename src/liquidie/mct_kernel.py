"""Numba-accelerated MCT memory-kernel matrix element computation.

Implements the mode-coupling theory kernel M(q) following
Nauroth & Kob, Phys. Rev. E 55, 657 (1997).
"""

from __future__ import annotations

import numpy as np
import numba
from numba import prange
from numpy.typing import NDArray


@numba.njit(cache=True, parallel=True)
def get_m(
    n_species: int,
    n_pts: int,
    c: NDArray,
    dk: float,
    x: NDArray,
    n_total: float,
    m: NDArray,
    inv_t: float,
    f: NDArray,
) -> NDArray:
    """Compute the MCT memory-kernel matrix M(q).

    Parameters
    ----------
    n_species
        Number of species.
    n_pts
        Number of grid points in reciprocal space.
    c
        Direct correlation function, shape ``(n_pts, n_species, n_species)``.
    dk
        Reciprocal-space grid spacing.
    x
        Mole fractions, shape ``(n_species,)``.
    n_total
        Total number density.
    m
        Masses, shape ``(n_species,)``.
    inv_t
        Inverse temperature 1/T.
    f
        Non-ergodicity parameter, shape ``(n_pts, n_species, n_species)``.

    Returns
    -------
    M : array, shape ``(n_pts, n_species, n_species)``

    Notes
    -----
    Both the k-integral and, in bipolar coordinates, the p-integral over
    ``[|q - k|, q + k]`` use the trapezoidal rule on the integer grid. The
    p limits are always grid nodes, so they get half weight; the k cutoff at
    ``n_pts - 1`` gets half weight as well.
    """
    pi = np.pi
    d = np.eye(n_species)
    M = np.zeros((n_pts, n_species, n_species))
    k_last = n_pts - 1

    # q = 0 contribution (Eq. from Phys. Rev. E 55, 657)
    for i in range(n_species):
        for j in range(n_species):
            for ki in range(1, n_pts):
                w_k = 0.5 if ki == k_last else 1.0
                for alpha in range(n_species):
                    for beta in range(n_species):
                        for alphap in range(n_species):
                            for betap in range(n_species):
                                v_i = (
                                    d[i, beta] * c[ki, i, alpha]
                                    - d[i, alpha] * c[ki, i, beta]
                                )
                                v_j = (
                                    d[j, betap] * c[ki, j, alphap]
                                    - d[j, alphap] * c[ki, j, betap]
                                )
                                M[0, i, j] += (
                                    w_k
                                    * ki**4
                                    * v_i
                                    * v_j
                                    * f[ki, alpha, alphap]
                                    * f[ki, beta, betap]
                                )
            M[0, i, j] *= dk**5 / (
                inv_t * n_total * m[i] * x[j] * 3.0 * (2.0 * pi) ** 2
            )

    # q != 0 contributions (parallel over q)
    for q in prange(1, n_pts):
        q2 = float(q * q)
        for ki in range(1, n_pts):
            ki2 = float(ki * ki)
            w_k = 0.5 if ki == k_last else 1.0
            p_lo = abs(q - ki)
            p_hi = min(q + ki, k_last)
            if p_hi <= p_lo:
                continue
            for p in range(p_lo, p_hi + 1):
                w = w_k * (0.5 if p == p_lo or p == p_hi else 1.0)
                p2 = float(p * p)
                fac1 = q2 - p2 + ki2
                fac2 = q2 + p2 - ki2
                for i in range(n_species):
                    for j in range(n_species):
                        for alpha in range(n_species):
                            for beta in range(n_species):
                                for alphap in range(n_species):
                                    for betap in range(n_species):
                                        v_i = (
                                            fac1 * d[i, beta] * c[ki, i, alpha]
                                            + fac2 * d[i, alpha] * c[p, i, beta]
                                        )
                                        v_j = (
                                            fac1 * d[j, betap] * c[ki, j, alphap]
                                            + fac2 * d[j, alphap] * c[p, j, betap]
                                        )
                                        M[q, i, j] += (
                                            w
                                            * ki
                                            * p
                                            * v_i
                                            * v_j
                                            * f[ki, alpha, alphap]
                                            * f[p, beta, betap]
                                        )
        for i in range(n_species):
            for j in range(n_species):
                M[q, i, j] *= dk**5 / (
                    inv_t * 32.0 * pi**2 * n_total * x[j] * m[i] * q**3
                )

    return M
