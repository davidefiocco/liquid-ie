"""Mode-coupling theory (MCT) driver.

Computes the non-ergodicity parameter F(q) from the static structure
factor S(q) and direct correlation function c(q) produced by the OZ
solver, following the MCT equations in Nauroth & Kob, Phys. Rev. E 55, 657 (1997).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton_krylov

from liquidie.config import Config
from liquidie.linalg import dotve, invv
from liquidie.mct_kernel import get_m
from liquidie.solver import SolverResult

logger = logging.getLogger("liquidie")


def run_mct(
    result: SolverResult,
    *,
    config: Config | None = None,
    masses: NDArray[np.floating] | None = None,
    method: Literal["picard", "newton_krylov"] = "picard",
    n_iterations: int = 3,
    tolerance: float = 1e-8,
    strict: bool = False,
) -> NDArray[np.floating]:
    """Run the MCT iteration for the non-ergodicity parameter F(q).

    Parameters
    ----------
    result
        Output from the OZ solver.  Must carry ``density`` and
        ``temperature`` metadata (automatically set by :func:`solve`)
        unless *config* is provided.
    config
        Optional override.  When given, density and temperature are
        read from the config instead of from *result*.
    masses
        Per-species masses, shape ``(n_species,)``.
        Defaults to all ones.
    method
        ``"picard"`` for simple fixed-point iteration, or
        ``"newton_krylov"`` for scipy's Newton-Krylov (LGMRES) solver.
    n_iterations
        Number of Picard iterations (only used when *method* is
        ``"picard"``).
    tolerance
        Convergence tolerance for Newton-Krylov.
    strict
        If True, raise ``FloatingPointError`` on non-finite values
        instead of silently replacing them with zero.

    Returns
    -------
    F : array, shape ``(n_pts, n_species, n_species)``
        Non-ergodicity parameter.
    """
    if config is not None:
        if config.system.n_species is None:
            raise ValueError("n_species must be set (provide at least one density)")
        rho_vec = np.array(config.system.density)
        temperature = config.system.temperature
    else:
        if result.density is None or result.temperature is None:
            raise ValueError(
                "SolverResult must carry density and temperature metadata. "
                "Either pass config=... or use a result produced by solve()."
            )
        rho_vec = np.asarray(result.density)
        temperature = result.temperature

    n_species = result.n_species
    n_pts = len(result.k)
    dk = result.k[1] - result.k[0]

    inv_t = 1.0 / temperature
    x = rho_vec / rho_vec.sum()
    n_total = rho_vec.sum()

    if masses is None:
        masses = np.ones(n_species)

    s_k = result.s_k

    # Recompute c from S via c_{ij}(q) = delta_{ij}/x_i - S^{-1}_{ij}(q)
    inv_s = invv(s_k)
    c = np.eye(n_species)[None, :, :] / x[None, :, None] - inv_s

    def compute_f(f_in: NDArray) -> NDArray:
        """Single MCT map: F_in -> F_new."""
        m_matrix = get_m(n_species, n_pts, c, dk, x, n_total, masses, inv_t, f_in)

        # N_{ij}(q) = m_i / (T * x_i) * M_{ij}(q)
        n_matrix = (masses[None, :, None] * inv_t / x[None, :, None]) * m_matrix

        q_arr = np.arange(n_pts) * dk

        # F = (q^2 I + S N)^{-1}  S N S
        q2_eye = np.outer(q_arr**2, np.eye(n_species)).reshape(-1, n_species, n_species)
        sn = dotve(s_k, n_matrix)
        f_new = dotve(dotve(dotve(invv(q2_eye + sn), s_k), n_matrix), s_k)
        if not np.all(np.isfinite(f_new)):
            if strict:
                raise FloatingPointError("Non-finite values in MCT map F(q)")
            logger.warning("Non-finite values in MCT map F(q); NaN/Inf replaced with 0")
            f_new = np.nan_to_num(f_new)
        return f_new

    # Initial guess
    f = np.ones_like(c)

    if method == "picard":
        for iteration in range(n_iterations):
            f = compute_f(f)
            if logger.isEnabledFor(logging.INFO):
                residual = np.max(np.abs(compute_f(f) - f))
                logger.info(
                    "Picard %d/%d  |F_new - F|_max = %.6e",
                    iteration + 1,
                    n_iterations,
                    residual,
                )
    elif method == "newton_krylov":

        def residual(f_flat: NDArray) -> NDArray:
            f_3d = f_flat.reshape(-1, n_species, n_species)
            f_old = f_3d.copy()
            f_new = compute_f(f_3d)
            return (f_new - f_old).flatten()

        counter = [0]

        def nk_callback(x: NDArray, f: NDArray = None) -> None:
            counter[0] += 1
            logger.info("MCT Newton-Krylov iteration %4d", counter[0])

        f_sol = newton_krylov(
            residual,
            f.flatten(),
            verbose=logger.isEnabledFor(logging.DEBUG),
            f_tol=tolerance,
            callback=nk_callback,
        )
        f = f_sol.reshape(-1, n_species, n_species)
    else:
        raise ValueError(f"Unknown method {method!r}; use 'picard' or 'newton_krylov'")

    return f


def write_mct_results(
    f: NDArray[np.floating],
    k: NDArray[np.floating],
    n_species: int,
    output_dir: Path,
) -> None:
    """Write MCT non-ergodicity parameter to text files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_species):
        for j in range(i, n_species):
            np.savetxt(
                output_dir / f"f{i}{j}.dat",
                np.column_stack((k, f[:, i, j])),
            )
