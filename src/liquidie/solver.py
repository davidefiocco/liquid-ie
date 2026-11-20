"""Ornstein-Zernike equation solver using Newton-Krylov iteration.

Solves the OZ equation for multi-component liquid systems in the form
    gamma(k) = [ (I*k - c(k)*rho)^{-1} * I*k - I ] * c(k)
closed by a user-specified closure relation, using scipy's Newton-Krylov
solver (LGMRES variant).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from scipy.optimize import NoConvergence, newton_krylov

from liquidie.config import Config
from liquidie.expressions import (
    apply_closure_vec,
    build_closure,
    build_potential,
    find_potential_jumps,
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
    density: NDArray[np.floating] | None = None
    temperature: float | None = None

    @classmethod
    def from_directory(cls, path: Path, n_species: int | None = None) -> "SolverResult":
        """Reconstruct a SolverResult from ``.dat`` files written by :func:`write_results`.

        Derived fields ``c_r`` and ``gamma_k`` are recomputed from the
        loaded data using the OZ decomposition identities
        ``c(r) = g(r) - gamma(r) - 1`` and ``gamma(k) = h(k) - c(k)``.
        If ``metadata.json`` is present, ``density``, ``temperature``, and
        (when omitted) ``n_species`` are restored automatically.
        """
        path = Path(path)
        density: NDArray[np.floating] | None = None
        temperature: float | None = None
        meta_path = path / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_n = meta.get("n_species")
            if n_species is None:
                n_species = meta_n
            elif meta_n is not None and meta_n != n_species:
                raise ValueError(
                    f"n_species={n_species} does not match metadata.json "
                    f"(n_species={meta_n})"
                )
            if meta.get("density") is not None:
                density = np.asarray(meta["density"], dtype=float)
            if meta.get("temperature") is not None:
                temperature = float(meta["temperature"])

        if n_species is None:
            raise ValueError(
                "n_species must be provided when metadata.json is not present"
            )

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

        c_r = rdf - gamma_r - 1.0
        gamma_k = h_k - c_k

        return cls(
            r=r,
            k=k,
            gamma_r=gamma_r,
            gamma_k=gamma_k,
            c_r=c_r,
            c_k=c_k,
            rdf=rdf,
            h_k=h_k,
            s_k=s_k,
            n_species=n_species,
            density=density,
            temperature=temperature,
        )

    def squeeze(self) -> "SolverResult":
        """Return a copy with correlation arrays squeezed to 1-D.

        Only valid for single-component systems (``n_species == 1``).

        Raises
        ------
        ValueError
            If ``n_species > 1``.
        """
        if self.n_species != 1:
            raise ValueError(f"squeeze() requires n_species == 1, got {self.n_species}")
        return SolverResult(
            r=self.r,
            k=self.k,
            gamma_r=self.gamma_r[:, 0, 0],
            gamma_k=self.gamma_k[:, 0, 0],
            c_r=self.c_r[:, 0, 0],
            c_k=self.c_k[:, 0, 0],
            rdf=self.rdf[:, 0, 0],
            h_k=self.h_k[:, 0, 0],
            s_k=self.s_k[:, 0, 0],
            n_species=1,
            density=self.density,
            temperature=self.temperature,
        )


def solve(
    config: Config,
    *,
    strict: bool = False,
    gamma_initial: NDArray[np.floating] | None = None,
) -> SolverResult:
    """Run the full OZ solve and return all correlation functions.

    Parameters
    ----------
    config
        Validated configuration object.
    strict
        If True, raise ``FloatingPointError`` on non-finite values
        instead of silently replacing them with zero.
    gamma_initial
        Optional initial guess for gamma(r), shape ``(N, n_species, n_species)``.
        When provided, this overrides both the zero-initialisation default and
        any file-based restart.  Useful for warm-starting from a previous
        solution at a nearby state point (density ramping, etc.).
    """
    if config.system.n_species is None:
        raise ValueError("n_species must be set (provide at least one density)")
    n_species = config.system.n_species
    inv_t = 1.0 / config.system.temperature

    rho_vec = np.array(config.system.density)
    rho = np.diag(rho_vec)

    closure_fn = build_closure(config.solver.closure, config.solver.closure_params)
    potential_fn = build_potential(
        config.potential.expression, config.potential.potential_params
    )

    # --- Grid and initial guess (or restart) ---
    if gamma_initial is not None:
        r = np.arange(0, config.grid.r_max, config.grid.dr)
        n_pts = len(r)
        gam = np.asarray(gamma_initial)
        if gam.shape != (n_pts, n_species, n_species):
            raise ValueError(
                f"gamma_initial shape {gam.shape} does not match "
                f"expected ({n_pts}, {n_species}, {n_species})"
            )
    elif config.restart.enabled:
        restart_data = np.loadtxt(config.restart.file)
        r = restart_data[:, 0]
        n_pts = len(r)
        expected_cols = 1 + n_species * n_species
        if restart_data.ndim != 2 or restart_data.shape[1] != expected_cols:
            raise ValueError(
                f"Restart file has shape {restart_data.shape}, "
                f"expected ({n_pts}, {expected_cols})"
            )
        gam = restart_data[:, 1:].reshape(n_pts, n_species, n_species)
    else:
        r = np.arange(0, config.grid.r_max, config.grid.dr)
        n_pts = len(r)
        gam = np.zeros((n_pts, n_species, n_species))

    # --- Potential and symbolic discontinuity nodes ---
    phi = generate_potential_grid(
        potential_fn,
        r,
        config.potential.sigma,
        config.potential.epsilon,
        n_species,
    )
    # Where a piecewise potential jumps on a grid node, the transforms use the
    # mean of the two one-sided closure values (trapezoidal rule for a step),
    # which is second order in dr. Reported c(r) and g(r) hold the outer limit,
    # e.g. g(sigma+).
    jumps = find_potential_jumps(
        config.potential.expression,
        potential_fn,
        r,
        config.potential.sigma,
        config.potential.epsilon,
        n_species,
        inv_t,
        config.potential.potential_params,
    )
    if not jumps.is_empty:
        phi[jumps.idx, jumps.i, jumps.j] = jumps.phi_outer

    ones_nn = np.ones((n_species, n_species))

    # gamma_r = gamma(r) * r  (the r-multiplied form used internally)
    gam_r = gam * np.outer(r, ones_nn).reshape(-1, n_species, n_species)

    # --- Picard / OZ step ---
    def picard(
        r_: NDArray,
        gam_r_: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        cr = apply_closure_vec(closure_fn, r_, gam_r_, phi, inv_t, strict=strict)
        cr_quad = cr
        if not jumps.is_empty:
            cr_in = np.asarray(
                closure_fn(
                    gam_r_[jumps.idx, jumps.i, jumps.j],
                    inv_t,
                    jumps.phi_inner,
                    r_[jumps.idx],
                ),
                dtype=float,
            )
            if not np.all(np.isfinite(cr_in)):
                if strict:
                    raise FloatingPointError(
                        "Non-finite values in closure evaluation at potential jump node"
                    )
                logger.warning(
                    "Non-finite values in closure evaluation at potential jump node; "
                    "NaN/Inf replaced with 0"
                )
                cr_in = np.nan_to_num(cr_in)
            cr_quad = cr.copy()
            cr_quad[jumps.idx, jumps.i, jumps.j] = 0.5 * (
                cr[jumps.idx, jumps.i, jumps.j] + cr_in
            )
        k_, ck = sft(r_, cr_quad)
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

        return r_new, k_, ck, gam_k, cr, cr_quad, gam_r_out

    # --- Newton-Krylov residual ---
    def residual(gam_r_flat: NDArray) -> NDArray:
        gam_r_ = gam_r_flat.reshape(-1, n_species, n_species)
        gam_r_old = gam_r_.copy()
        *_, gam_r_new = picard(r, gam_r_)
        return (gam_r_new - gam_r_old).flatten()

    # --- Solve ---
    counter = [0]

    def callback(x: NDArray, f: NDArray = None) -> None:
        counter[0] += 1
        logger.info("OZ Newton-Krylov iteration %4d", counter[0])

    # Skip Newton-Krylov if the initial guess already satisfies the tolerance
    # (also avoids scipy's 0/0 RuntimeWarning on exact warm starts).
    initial_res = residual(gam_r.flatten())
    if (
        np.all(np.isfinite(initial_res))
        and float(np.max(np.abs(initial_res))) <= config.solver.tolerance
    ):
        gam_r_sol = gam_r
    else:
        nk_kwargs: dict[str, object] = dict(
            verbose=logger.isEnabledFor(logging.DEBUG),
            f_tol=config.solver.tolerance,
            callback=callback,
        )
        if config.solver.maxiter is not None:
            nk_kwargs["maxiter"] = config.solver.maxiter
        try:
            gam_r_sol = newton_krylov(
                residual,
                gam_r.flatten(),
                **nk_kwargs,
            )
        except NoConvergence as exc:
            raise RuntimeError(
                f"OZ Newton-Krylov solver did not converge in {counter[0]} "
                f"iterations (tolerance={config.solver.tolerance:.3e}). "
                "Try density/temperature ramping via gamma_initial, increasing "
                "solver.maxiter, or adjusting the grid."
            ) from exc
        except OverflowError as exc:
            raise RuntimeError(
                "OZ solver overflowed during iteration (exponential blow-up in "
                "closure). Try approaching this state point via density/temperature "
                "ramping with gamma_initial."
            ) from exc
        gam_r_sol = gam_r_sol.reshape(-1, n_species, n_species)

    # Final Picard step to get all functions
    r, k, ck, gam_k, cr, cr_quad, gam_r_sol = picard(r, gam_r_sol)

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
    c_kspace[0] = 4.0 * np.pi * trapezoid(cr_quad * r_broadcast, x=r, axis=0)
    # Take gamma(0) from the OZ relation rather than a real-space quadrature,
    # which converges too slowly to keep S(0) on the compressibility route.
    c_zero = c_kspace[0]
    gamma_kspace[0] = np.linalg.solve(np.eye(n_species) - c_zero @ rho, c_zero) - c_zero

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

    # A non-positive-definite S(k) is thermodynamically impossible, so it flags
    # the spurious roots the iteration can land on when the grid is too coarse.
    s_min_eig = float(np.min(np.linalg.eigvalsh(s_k)))
    if s_min_eig <= 0.0:
        message = (
            f"Unphysical structure factor: min eigenvalue of S(k) = {s_min_eig:.4g} <= 0. "
            "The solver converged to a spurious root; retry with a smaller dr, "
            "a larger r_max, or density ramping via gamma_initial."
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)

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
        density=rho_vec,
        temperature=config.system.temperature,
    )


def write_results(result: SolverResult, output_dir: Path) -> None:
    """Write solver outputs to text files in *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = result.n_species

    # gamma.dat (restart file; columns in row-major (i, j) order)
    gamma_flat = result.gamma_r.reshape(len(result.r), n * n)
    np.savetxt(
        output_dir / "gamma.dat",
        np.column_stack((result.r, gamma_flat)),
    )

    metadata = {
        "n_species": n,
        "density": (
            [float(x) for x in result.density] if result.density is not None else None
        ),
        "temperature": (
            float(result.temperature) if result.temperature is not None else None
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
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
