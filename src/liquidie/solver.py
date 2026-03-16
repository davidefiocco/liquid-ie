"""Ornstein-Zernike equation solver using Newton-Krylov iteration.

Solves the OZ equation for multi-component liquid systems in the form
    gamma(k) = [ (I*k - c(k)*rho)^{-1} * I*k - I ] * c(k)
closed by a user-specified closure relation, using scipy's Newton-Krylov
solver (LGMRES variant).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from scipy.optimize import newton_krylov

from liquidie.config import Config
from liquidie.expressions import (
    apply_closure_vec,
    build_closure,
    build_potential,
    generate_potential_grid,
)
from liquidie.linalg import dotvbs, dotve, invv
from liquidie.transforms import sft

logger = logging.getLogger("liquidie")


@dataclass
class SolverResult:
    """Container for all OZ solver outputs."""

    r: NDArray[np.floating]
    k: NDArray[np.floating]
    gamma_r: NDArray[np.floating]
    gamma_k: NDArray[np.floating]
    c_r: NDArray[np.floating]
    c_k: NDArray[np.floating]
    rdf: NDArray[np.floating]
    h_k: NDArray[np.floating]
    s_k: NDArray[np.floating]
    n_species: int

    @classmethod
    def from_directory(cls, path: Path, n_species: int) -> "SolverResult":
        """Reconstruct a SolverResult from ``.dat`` files written by :func:`write_results`."""
        gamma_data = np.loadtxt(path / "gamma.dat")
        r = gamma_data[:, 0]
        n_pts_r = len(r)
        gamma_r = gamma_data[:, 1:].reshape(n_pts_r, n_species, n_species)

        first_s = np.loadtxt(path / "s00.dat")
        k = first_s[:, 0]
        n_pts = len(k)

        s_k = np.zeros((n_pts, n_species, n_species))
        c_k = np.zeros((n_pts, n_species, n_species))
        h_k = np.zeros((n_pts, n_species, n_species))
        rdf = np.zeros((n_pts_r, n_species, n_species))

        for i in range(n_species):
            for j in range(i, n_species):
                s_data = np.loadtxt(path / f"s{i}{j}.dat")
                s_k[:, i, j] = s_data[:, 1]
                s_k[:, j, i] = s_data[:, 1]

                c_data = np.loadtxt(path / f"c{i}{j}.dat")
                c_k[:, i, j] = c_data[:, 1]
                c_k[:, j, i] = c_data[:, 1]

                h_data = np.loadtxt(path / f"h{i}{j}.dat")
                h_k[:, i, j] = h_data[:, 1]
                h_k[:, j, i] = h_data[:, 1]

                rdf_data = np.loadtxt(path / f"rdf{i}{j}.dat")
                rdf[:, i, j] = rdf_data[:, 1]
                rdf[:, j, i] = rdf_data[:, 1]

        return cls(
            r=r,
            k=k,
            gamma_r=gamma_r,
            gamma_k=np.zeros((n_pts, n_species, n_species)),
            c_r=np.zeros_like(gamma_r),
            c_k=c_k,
            rdf=rdf,
            h_k=h_k,
            s_k=s_k,
            n_species=n_species,
        )


def solve(
    config: Config,
    *,
    strict: bool = False,
) -> SolverResult:
    """Run the full OZ solve and return all correlation functions.

    Parameters
    ----------
    config
        Validated configuration object.
    strict
        If True, raise ``FloatingPointError`` on non-finite values
        instead of silently replacing them with zero.
    """
    if config.system.n_species is None:
        raise ValueError("n_species must be set (provide at least one density)")
    n_species = config.system.n_species
    inv_t = 1.0 / config.system.temperature

    rho_vec = np.array(config.system.density)
    rho = np.diag(rho_vec)

    closure_fn = build_closure(config.solver.closure, config.solver.closure_params)
    potential_fn = build_potential(config.potential.expression)

    # --- Grid and potential ---
    r = np.arange(0, config.grid.r_max, config.grid.dr)
    n_pts = len(r)

    phi = generate_potential_grid(
        potential_fn,
        r,
        config.potential.sigma,
        config.potential.epsilon,
        n_species,
    )

    # --- Initial guess (or restart) ---
    if config.restart.enabled:
        restart_data = np.loadtxt(config.restart.file)
        r = restart_data[:, 0]
        n_pts = len(r)
        gam = restart_data[:, 1:].reshape(n_pts, n_species, n_species)
    else:
        gam = np.zeros((n_pts, n_species, n_species))

    ones_nn = np.ones((n_species, n_species))

    # gamma_r = gamma(r) * r  (the r-multiplied form used internally)
    gam_r = gam * np.outer(r, ones_nn).reshape(-1, n_species, n_species)

    # --- Picard / OZ step ---
    def picard(
        r_: NDArray,
        gam_r_: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        cr = apply_closure_vec(closure_fn, r_, gam_r_, phi, inv_t, strict=strict)
        k_, ck = sft(r_, cr)
        ck[1:] = 4.0 * np.pi * ck[1:]
        ck[0] = 0.0

        gam_k = np.zeros_like(ck)
        ik = (
            np.outer(np.eye(n_species), k_)
            .reshape(n_species, n_species, -1)
            .transpose(2, 0, 1)
        )
        gam_k[1:] = (
            dotve(
                dotve(invv(ik[1:] - dotvbs(ck[1:], rho)), ik[1:]),
                ck[1:],
            )
            - ck[1:]
        )
        gam_k[0] = 0.0

        r_new, gam_r_out = sft(k_, gam_k)
        gam_r_out[1:] = gam_r_out[1:] / (2.0 * np.pi**2)
        gam_r_out[0] = 0.0

        return r_new, k_, ck, gam_k, cr, gam_r_out

    # --- Newton-Krylov residual ---
    def residual(gam_r_flat: NDArray) -> NDArray:
        gam_r_ = gam_r_flat.reshape(-1, n_species, n_species)
        gam_r_old = gam_r_.copy()
        _, _, _, _, _, gam_r_new = picard(r, gam_r_)
        return (gam_r_new - gam_r_old).flatten()

    # --- Solve ---
    counter = [0]

    def callback(x: NDArray, f: NDArray = None) -> None:
        counter[0] += 1
        logger.info("OZ Newton-Krylov iteration %4d", counter[0])

    gam_r_sol = newton_krylov(
        residual,
        gam_r.flatten(),
        verbose=logger.isEnabledFor(logging.DEBUG),
        f_tol=config.solver.tolerance,
        callback=callback,
    )
    gam_r_sol = gam_r_sol.reshape(-1, n_species, n_species)

    # Final Picard step to get all functions
    r, k, ck, gam_k, cr, gam_r_sol = picard(r, gam_r_sol)

    # --- Convert from r-multiplied form back to physical functions ---
    r_broadcast = np.outer(r, ones_nn).reshape(-1, n_species, n_species)
    k_broadcast = np.outer(k, ones_nn).reshape(-1, n_species, n_species)

    # Real-space: gamma(r) = gamma_r(r) / r,  c(r) = c_r(r) / r
    gamma = np.zeros_like(gam_r_sol)
    c = np.zeros_like(cr)
    gamma[1:] = gam_r_sol[1:] / r_broadcast[1:]
    c[1:] = cr[1:] / r_broadcast[1:]
    gamma[0] = trapezoid(gam_k * k_broadcast, x=k, axis=0) / (2.0 * np.pi**2)
    c[0] = trapezoid(ck * k_broadcast, x=k, axis=0) / (2.0 * np.pi**2)

    rdf = c + gamma + 1.0

    # Reciprocal-space: gamma(k) = gamma_k(k) / k,  c(k) = c_k(k) / k
    gamma_kspace = np.zeros_like(gam_k)
    c_kspace = np.zeros_like(ck)
    gamma_kspace[1:] = gam_k[1:] / k_broadcast[1:]
    c_kspace[1:] = ck[1:] / k_broadcast[1:]
    gamma_kspace[0] = 4.0 * np.pi * trapezoid(gam_r_sol * r_broadcast, x=r, axis=0)
    c_kspace[0] = 4.0 * np.pi * trapezoid(cr * r_broadcast, x=r, axis=0)

    h_k = c_kspace + gamma_kspace

    # Structure factor S(k)
    rho_total = np.trace(rho)
    s_k = np.zeros_like(h_k)
    for i in range(n_species):
        for j in range(i, n_species):
            s_ij = (
                rho[i, i] / rho_total * np.eye(n_species)[i, j]
                + rho[i, i] * rho[j, j] / rho_total * h_k[:, i, j]
            )
            s_k[:, i, j] = s_ij
            s_k[:, j, i] = s_ij

    return SolverResult(
        r=r,
        k=k,
        gamma_r=gamma,
        gamma_k=gamma_kspace,
        c_r=c,
        c_k=c_kspace,
        rdf=rdf,
        h_k=h_k,
        s_k=s_k,
        n_species=n_species,
    )


def write_results(result: SolverResult, output_dir: Path) -> None:
    """Write solver outputs to text files in *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n = result.n_species

    # gamma.dat (restart file)
    gamma_flat = result.gamma_r.transpose(2, 1, 0).reshape(n * n, -1)
    np.savetxt(
        output_dir / "gamma.dat",
        np.vstack((result.r, gamma_flat)).T,
    )

    for i in range(n):
        for j in range(i, n):
            np.savetxt(
                output_dir / f"rdf{i}{j}.dat",
                np.column_stack((result.r, result.rdf[:, i, j])),
            )
            np.savetxt(
                output_dir / f"c{i}{j}.dat",
                np.column_stack((result.k, result.c_k[:, i, j])),
            )
            np.savetxt(
                output_dir / f"h{i}{j}.dat",
                np.column_stack((result.k, result.h_k[:, i, j])),
            )
            np.savetxt(
                output_dir / f"s{i}{j}.dat",
                np.column_stack((result.k, result.s_k[:, i, j])),
            )
