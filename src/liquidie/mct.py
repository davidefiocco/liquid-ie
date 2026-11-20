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
from scipy.integrate import trapezoid
from scipy.optimize import NoConvergence, newton_krylov

from liquidie.config import Config
from liquidie.linalg import dotve, invv
from liquidie.mct_kernel import get_m
from liquidie.solver import SolverResult

logger = logging.getLogger("liquidie")


def _f_zero_q(
    f: NDArray,
    n_zero: NDArray,
    s_zero: NDArray,
    c: NDArray,
    dc: NDArray,
    k: NDArray,
    x: NDArray,
    n_total: float,
    dk: float,
) -> NDArray:
    """Exact q -> 0 limit of the MCT map, from the vertex expanded in q.

    ``F(q) = S - (S^{-1} + N(q)/q^2)^{-1}`` with ``N(q) = N_0 + q^2 N_2 + ...``.
    Momentum conservation makes ``N_0 x = 0`` along the total-density mode;
    ``N_0`` is positive on arrested concentration modes. With ``U`` spanning
    the null space of ``N_0`` and ``G = S^{-1} + N_2``, the limit is
    ``F(0) = S - U (U^T G U)^{-1} U^T``. Along ``x`` the vertex is
    ``q [c(k) + k mu^2 c'(k)] + O(q^2)``, so that ``x^T N_2 x = kappa``,

        kappa = 1/(8 pi^2 n) int dk k^2 [2 tr(FcFc) + 4/3 k tr(FcFc')
                                         + 2/5 k^2 tr(Fc'Fc')].

    ``N_2`` is kept along ``x`` only. That is exact for one component, for a
    mixture whose concentration modes are all arrested (``U = x``), and for
    a liquid; it drops the O(q^2) kernel of concentration modes that stay
    mobile in a partially arrested mixture.
    """
    fc = np.einsum("qab,qbc->qac", f, c)
    fdc = np.einsum("qab,qbc->qac", f, dc)
    integrand = k**2 * (
        2.0 * np.einsum("qab,qba->q", fc, fc)
        + 4.0 / 3.0 * k * np.einsum("qab,qba->q", fc, fdc)
        + 0.4 * k**2 * np.einsum("qab,qba->q", fdc, fdc)
    )
    kappa = trapezoid(integrand, k) / (8.0 * np.pi**2 * n_total)

    s_inv = np.linalg.inv(s_zero)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (n_zero + n_zero.T))
    # A mode adds lambda/q^2 to S^{-1}; below this it is invisible even at q = dk.
    threshold = 1e-8 * dk**2 * np.linalg.norm(s_inv, 2)
    u = eigvecs[:, eigvals <= threshold]
    if u.shape[1] == 0:
        u = (x / np.linalg.norm(x))[:, None]
    g = s_inv + kappa * np.outer(x, x) / (x @ x) ** 2
    return s_zero - u @ np.linalg.solve(u.T @ g @ u, u.T)


def run_mct(
    result: SolverResult,
    *,
    config: Config | None = None,
    masses: NDArray[np.floating] | None = None,
    method: Literal["picard", "newton_krylov"] = "picard",
    n_iterations: int = 1000,
    tolerance: float = 1e-8,
    k_max: float | None = None,
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
        Maximum number of Picard iterations (only used when *method* is
        ``"picard"``).  The loop stops early once successive iterates
        change by less than *tolerance*.
    tolerance
        Convergence tolerance for both solvers.
    k_max
        Optional upper wavevector cutoff.  When set, the MCT kernel is
        evaluated only on the ``result.k <= k_max`` slice (matching
        ``result.k[:len(F)]``), decoupling the O(N^3) MCT cost from a fine
        real-space OZ grid ``dr``.
    strict
        If True, raise ``FloatingPointError`` on non-finite values or on
        failure to converge, instead of warning and continuing.

    Returns
    -------
    F : array, shape ``(n_q, n_species, n_species)``
        Non-ergodicity parameter, *unnormalised*: ``F(q) = S(q) f(q)``.
        Divide by ``result.s_k[:len(F)]`` to recover the ``f(q)`` that
        is conventionally plotted and tabulated in the literature.
        ``F(0)`` is the analytical q -> 0 limit (see :func:`_f_zero_q`).
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
    if k_max is not None:
        if k_max <= 0:
            raise ValueError(f"k_max must be positive, got {k_max}")
        n_pts = int(np.sum(result.k <= k_max))
    else:
        n_pts = len(result.k)
    if n_pts < 3:
        raise ValueError(f"MCT needs at least 3 wavevectors, got {n_pts}")
    dk = result.k[1] - result.k[0]

    inv_t = 1.0 / temperature
    x = rho_vec / rho_vec.sum()
    n_total = rho_vec.sum()

    if masses is None:
        masses = np.ones(n_species)

    s_k = result.s_k[:n_pts]

    # Recompute c from S via c_{ij}(q) = delta_{ij}/x_i - S^{-1}_{ij}(q)
    inv_s = invv(s_k)
    c = np.eye(n_species)[None, :, :] / x[None, :, None] - inv_s
    dc = np.gradient(c, dk, axis=0, edge_order=2)
    k = np.arange(n_pts) * dk
    q2_eye = np.outer(k[1:] ** 2, np.eye(n_species)).reshape(-1, n_species, n_species)
    s_pos = s_k[1:]
    n_scale = masses[None, :, None] * inv_t / x[None, :, None]

    def compute_f_pos(f_in: NDArray) -> tuple[NDArray, NDArray]:
        """Single MCT map for q > 0; also returns N(q=0)."""
        m_matrix = get_m(n_species, n_pts, c, dk, x, n_total, masses, inv_t, f_in)

        # N_{ij}(q) = m_i / (T * x_i) * M_{ij}(q)
        n_matrix = n_scale * m_matrix

        # F = (q^2 I + S N)^{-1}  S N S
        sn = dotve(s_pos, n_matrix[1:])
        f_pos_new = dotve(dotve(invv(q2_eye + sn), sn), s_pos)
        if not np.all(np.isfinite(f_pos_new)):
            if strict:
                raise FloatingPointError("Non-finite values in MCT map F(q)")
            logger.warning("Non-finite values in MCT map F(q); NaN/Inf replaced with 0")
            f_pos_new = np.nan_to_num(f_pos_new)
        return f_pos_new, n_matrix[0]

    # Starting from the maximal admissible F = S makes the monotone iteration
    # descend onto the largest fixed point, which is the physical one.
    # The kernel weights the q = 0 node by zero, so iterate over q > 0 and
    # evaluate the exact q -> 0 limit once after convergence.
    f = s_k.copy()
    f[0] = 0.0
    n_zero = np.zeros((n_species, n_species))

    if method == "picard":
        delta = float("inf")
        for iteration in range(n_iterations):
            f_pos_new, n_zero = compute_f_pos(f)
            delta = float(np.max(np.abs(f_pos_new - f[1:])))
            f[1:] = f_pos_new
            logger.info(
                "Picard %d/%d  |F_new - F|_max = %.6e",
                iteration + 1,
                n_iterations,
                delta,
            )
            if delta < tolerance:
                break
        else:
            message = (
                f"MCT Picard iteration did not converge in {n_iterations} "
                f"iterations (last change {delta:.3e} > tolerance {tolerance:.3e}). "
                "F(q) may be an unconverged transient; raise n_iterations."
            )
            if strict:
                raise FloatingPointError(message)
            logger.warning(message)
    elif method == "newton_krylov":

        def residual(f_pos_flat: NDArray) -> NDArray:
            f_3d = np.zeros((n_pts, n_species, n_species))
            f_3d[1:] = f_pos_flat.reshape(n_pts - 1, n_species, n_species)
            f_pos_new, _ = compute_f_pos(f_3d)
            return (f_pos_new - f_3d[1:]).flatten()

        counter = [0]

        def nk_callback(x: NDArray, f: NDArray = None) -> None:
            counter[0] += 1
            logger.info("MCT Newton-Krylov iteration %4d", counter[0])

        try:
            f_pos_sol = newton_krylov(
                residual,
                f[1:].flatten(),
                verbose=logger.isEnabledFor(logging.DEBUG),
                f_tol=tolerance,
                callback=nk_callback,
            )
        except NoConvergence as exc:
            message = (
                f"MCT Newton-Krylov did not converge in {counter[0]} iterations "
                f"(tolerance={tolerance:.3e}). Try method='picard'."
            )
            if strict:
                raise FloatingPointError(message) from exc
            raise RuntimeError(message) from exc
        f[1:] = f_pos_sol.reshape(n_pts - 1, n_species, n_species)
        _, n_zero = compute_f_pos(f)
    else:
        raise ValueError(f"Unknown method {method!r}; use 'picard' or 'newton_krylov'")

    f[0] = _f_zero_q(f, n_zero, s_k[0], c, dc, k, x, n_total, dk)
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
