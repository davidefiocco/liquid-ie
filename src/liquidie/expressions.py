"""SymPy-based expression parsing for closures and potentials.

Both closures and pair potentials are specified as strings that are either
a known preset name (e.g. ``"PY"``, ``"hard_sphere"``) or an arbitrary SymPy
math expression.  The shared pipeline is:

1. Look up the string in a preset registry; expand if matched.
2. Parse via ``sympy.sympify``.
3. Validate that all free symbols belong to an allowed set.
4. Convert to a fast NumPy callable via ``sympy.lambdify``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy
from numpy.typing import NDArray
from sympy.core.relational import Relational

logger = logging.getLogger("liquidie")

# ---------------------------------------------------------------------------
# Preset registries
# ---------------------------------------------------------------------------

KNOWN_CLOSURES: dict[str, str] = {
    "PY": "(r + gamma_r) * (exp(-inv_t * phi) - 1)",
    "HNC": "r * exp(-inv_t * phi + gamma_r / r) - gamma_r - r",
    "MS": "r * exp(-inv_t * phi + sqrt(1 + 2*gamma_r/r) - 1) - gamma_r - r",
    "BPGG": "r * exp(-inv_t * phi + (1 + s*gamma_r/r)**(1/s) - 1) - gamma_r - r",
}

HARD_CORE_HEIGHT = 1e30

KNOWN_POTENTIALS: dict[str, str] = {
    "hard_sphere": f"Piecewise(({HARD_CORE_HEIGHT:.0e}, r < sigma), (0, True))",
    "lennard_jones": "4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)",
    "inverse_power_law": "epsilon * (sigma / r)**n",
    "gaussian_core": "epsilon * exp(-(r / sigma)**2)",
}

CLOSURE_SYMBOLS = {"r", "gamma_r", "phi", "inv_t"}
POTENTIAL_SYMBOLS = {"r", "sigma", "epsilon"}


def list_potentials() -> list[str]:
    """Return the names of all built-in potential presets."""
    return list(KNOWN_POTENTIALS)


def list_closures() -> list[str]:
    """Return the names of all built-in closure presets."""
    return list(KNOWN_CLOSURES)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def _parse_expression(
    spec: str,
    registry: dict[str, str],
    allowed_symbols: set[str],
    params: dict[str, float] | None = None,
    params_hint: str = "params",
) -> tuple[sympy.Expr, dict[str, sympy.Symbol]]:
    """Expand a preset name or raw string into a validated SymPy expression."""
    params = params or {}
    expr_str = registry.get(spec, spec)

    all_names = allowed_symbols | set(params)
    sym_locals = {name: sympy.Symbol(name) for name in all_names}
    try:
        expr = sympy.sympify(expr_str, locals=sym_locals)
    except (sympy.SympifyError, SyntaxError) as exc:
        raise ValueError(f"Cannot parse expression: {expr_str!r}") from exc

    if params:
        expr = expr.subs({sym_locals[k]: v for k, v in params.items()})

    free = {str(s) for s in expr.free_symbols}
    unexpected = free - allowed_symbols
    if unexpected:
        hint = f"set unresolved params via {params_hint}"
        lower_map = {k.lower(): k for k in registry}
        if spec.lower() in lower_map and spec not in registry:
            hint = f"did you mean preset {lower_map[spec.lower()]!r}?"
        raise ValueError(
            f"Expression contains unknown symbol(s) {unexpected}. "
            f"Allowed: {allowed_symbols} ({hint})"
        )

    return expr, sym_locals


def build_expression(
    spec: str,
    registry: dict[str, str],
    allowed_symbols: set[str],
    params: dict[str, float] | None = None,
    *,
    params_hint: str = "params",
) -> Callable[..., Any]:
    """Parse *spec* into a NumPy-vectorised callable.

    Parameters
    ----------
    spec
        Either a key in *registry* or a raw SymPy expression string.
    registry
        Mapping of shortcut names to canonical expression strings.
    allowed_symbols
        Set of permitted free-symbol names.  Any other symbol in the
        parsed expression raises ``ValueError``.
    params
        Optional mapping of parameter names to scalar values.  These
        symbols are substituted into the expression at build time so
        the returned callable has the same signature as when *params*
        is empty.

    Returns
    -------
    A callable whose positional arguments correspond to *allowed_symbols*
    in sorted order, accepting and returning NumPy arrays.
    """
    expr, sym_locals = _parse_expression(
        spec, registry, allowed_symbols, params, params_hint=params_hint
    )
    ordered_syms = sorted(allowed_symbols)
    sym_objects = [sym_locals[name] for name in ordered_syms]
    fn = sympy.lambdify(sym_objects, expr, modules="numpy")
    return fn


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------


def build_closure(
    spec: str,
    params: dict[str, float] | None = None,
) -> Callable[..., Any]:
    """Build a closure callable from a preset name or SymPy expression.

    The returned function has signature ``f(gamma_r, inv_t, phi, r)``
    (arguments in sorted alphabetical order).  Extra *params* are
    substituted at build time.
    """
    return build_expression(
        spec,
        KNOWN_CLOSURES,
        CLOSURE_SYMBOLS,
        params,
        params_hint="closure_params",
    )


def build_potential(
    spec: str,
    params: dict[str, float] | None = None,
) -> Callable[..., Any]:
    """Build a potential callable from a preset name or SymPy expression.

    The returned function has signature ``f(epsilon, r, sigma)``
    (arguments in sorted alphabetical order).  Extra *params* (e.g.
    exponent ``n`` for an inverse-power-law potential) are substituted
    at build time.
    """
    return build_expression(
        spec,
        KNOWN_POTENTIALS,
        POTENTIAL_SYMBOLS,
        params,
        params_hint="potential_params",
    )


# ---------------------------------------------------------------------------
# Vectorised application helpers
# ---------------------------------------------------------------------------


def apply_closure_vec(
    closure_fn: Callable[..., Any],
    r: NDArray[np.floating],
    gamma_r: NDArray[np.floating],
    phi: NDArray[np.floating],
    inv_t: float,
    *,
    strict: bool = False,
) -> NDArray[np.floating]:
    """Apply *closure_fn* element-wise over all species pairs.

    Parameters
    ----------
    closure_fn
        Callable with signature ``(gamma_r, inv_t, phi, r)``.
    r, gamma_r, phi
        Arrays of shape ``(N,)`` or ``(N, n_species, n_species)``.
    inv_t
        Inverse temperature (scalar).
    strict
        If True, raise ``FloatingPointError`` on non-finite values
        instead of replacing them with zero.

    Returns
    -------
    Array of same shape as *gamma_r*.
    """
    n_species = gamma_r.shape[1]
    cr = np.zeros_like(gamma_r)
    # Every closure but PY divides by r, so r=0 is skipped; r*c(r) vanishes there.
    inner = r > 0
    for i in range(n_species):
        for j in range(n_species):
            vals = closure_fn(gamma_r[inner, i, j], inv_t, phi[inner, i, j], r[inner])
            if not np.all(np.isfinite(vals)):
                if strict:
                    raise FloatingPointError(
                        f"Non-finite values in closure evaluation for pair ({i},{j})"
                    )
                logger.warning(
                    "Non-finite values in closure evaluation for pair (%d,%d); "
                    "NaN/Inf replaced with 0",
                    i,
                    j,
                )
                vals = np.nan_to_num(vals)
            cr[inner, i, j] = vals
    return cr


def generate_potential_grid(
    potential_fn: Callable[..., Any],
    r: NDArray[np.floating],
    sigma: NDArray[np.floating],
    epsilon: NDArray[np.floating],
    n_species: int,
) -> NDArray[np.floating]:
    """Evaluate the potential on the radial grid for every species pair.

    Parameters
    ----------
    potential_fn
        Callable with signature ``(epsilon, r, sigma)``.
    r
        1-D radial grid, shape ``(N,)``.
    sigma, epsilon
        Flattened ``n_species x n_species`` parameter arrays.
    n_species
        Number of species.

    Returns
    -------
    Array of shape ``(N, n_species, n_species)``.
    """
    sigma_mat = np.asarray(sigma).reshape(n_species, n_species)
    eps_mat = np.asarray(epsilon).reshape(n_species, n_species)
    n_pts = len(r)
    phi = np.zeros((n_pts, n_species, n_species))
    inner = r > 0
    has_zero = bool(np.any(~inner))
    for i in range(n_species):
        for j in range(n_species):
            phi[inner, i, j] = potential_fn(eps_mat[i, j], r[inner], sigma_mat[i, j])
            if has_zero:
                with np.errstate(divide="ignore", invalid="ignore"):
                    phi[~inner, i, j] = potential_fn(
                        eps_mat[i, j], r[~inner], sigma_mat[i, j]
                    )
    return phi


@dataclass(frozen=True)
class PotentialJumps:
    """Grid nodes where a piecewise pair potential jumps in ``exp(-inv_t*phi)``."""

    idx: NDArray[np.intp]
    i: NDArray[np.intp]
    j: NDArray[np.intp]
    phi_inner: NDArray[np.floating]
    phi_outer: NDArray[np.floating]

    @property
    def is_empty(self) -> bool:
        return len(self.idx) == 0


def find_potential_jumps(
    spec: str,
    potential_fn: Callable[..., Any],
    r: NDArray[np.floating],
    sigma: NDArray[np.floating],
    epsilon: NDArray[np.floating],
    n_species: int,
    inv_t: float,
    params: dict[str, float] | None = None,
) -> PotentialJumps:
    """Locate grid nodes where a ``Piecewise``/``Heaviside`` potential jumps.

    Breakpoints in *r* are extracted symbolically from the SymPy expression;
    smooth potentials return an empty :class:`PotentialJumps` without probing
    the grid.  For each symbolic breakpoint that lands on a grid node ``r > 0``,
    the one-sided limits ``phi(r_b-)`` and ``phi(r_b+)`` are evaluated and
    kept whenever ``exp(-inv_t * phi)`` actually jumps across the node.
    """
    empty = PotentialJumps(
        idx=np.empty(0, dtype=np.intp),
        i=np.empty(0, dtype=np.intp),
        j=np.empty(0, dtype=np.intp),
        phi_inner=np.empty(0, dtype=float),
        phi_outer=np.empty(0, dtype=float),
    )
    expr, sym_locals = _parse_expression(
        spec, KNOWN_POTENTIALS, POTENTIAL_SYMBOLS, params
    )
    r_sym = sym_locals["r"]

    equations: list[sympy.Expr] = []
    for pw in expr.atoms(sympy.Piecewise):
        for _, cond in pw.args:
            for rel in cond.atoms(Relational):
                if r_sym in rel.free_symbols:
                    equations.append(rel.lhs - rel.rhs)
    for hv in expr.atoms(sympy.Heaviside):
        arg = hv.args[0]
        if r_sym in arg.free_symbols:
            equations.append(arg)

    if not equations or len(r) < 2:
        return empty

    bp_syms = [sym_locals["epsilon"], sym_locals["sigma"]]
    bp_fns: list[Callable[[float, float], Any]] = []
    seen_sols: set[sympy.Expr] = set()
    for eq in equations:
        try:
            sols = sympy.solve(eq, r_sym)
        except (NotImplementedError, ValueError):
            continue
        for sol in sols:
            if r_sym in sol.free_symbols or sol in seen_sols:
                continue
            seen_sols.add(sol)
            bp_fns.append(sympy.lambdify(bp_syms, sol, modules="numpy"))

    if not bp_fns:
        return empty

    sigma_mat = np.asarray(sigma, dtype=float).reshape(n_species, n_species)
    eps_mat = np.asarray(epsilon, dtype=float).reshape(n_species, n_species)
    dr = float(r[1] - r[0])
    tol = 1e-9 * dr
    offset = 1e-9 * dr

    idxs: list[int] = []
    is_: list[int] = []
    js_: list[int] = []
    phi_in_list: list[float] = []
    phi_out_list: list[float] = []

    for i in range(n_species):
        for j in range(n_species):
            eps_ij = float(eps_mat[i, j])
            sig_ij = float(sigma_mat[i, j])
            seen_idx: set[int] = set()
            for bp_fn in bp_fns:
                try:
                    raw = complex(bp_fn(eps_ij, sig_ij))
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                if abs(raw.imag) > 1e-12 or not np.isfinite(raw.real) or raw.real <= 0:
                    continue
                r_b = raw.real
                idx = int(np.rint((r_b - r[0]) / dr))
                if idx <= 0 or idx >= len(r) or idx in seen_idx:
                    continue
                on_node = abs(r[idx] - r_b) <= tol
                with np.errstate(all="ignore"):
                    p_in = float(
                        np.asarray(
                            potential_fn(eps_ij, np.array([r_b - offset]), sig_ij)
                        ).ravel()[0]
                    )
                    p_out = float(
                        np.asarray(
                            potential_fn(eps_ij, np.array([r_b + offset]), sig_ij)
                        ).ravel()[0]
                    )
                    b_in = float(np.exp(-inv_t * p_in))
                    b_out = float(np.exp(-inv_t * p_out))
                if (
                    np.isfinite(b_in)
                    and np.isfinite(b_out)
                    and not np.isclose(b_in, b_out, rtol=1e-6, atol=1e-6)
                ):
                    if not on_node:
                        logger.warning(
                            "Potential discontinuity at r=%g for pair (%d,%d) falls "
                            "between grid nodes (dr=%g); accuracy drops to first order "
                            "in dr. Choose dr to divide %g for second-order accuracy.",
                            r_b,
                            i,
                            j,
                            dr,
                            r_b,
                        )
                        continue
                    seen_idx.add(idx)
                    idxs.append(idx)
                    is_.append(i)
                    js_.append(j)
                    phi_in_list.append(p_in)
                    phi_out_list.append(p_out)

    if not idxs:
        return empty
    return PotentialJumps(
        idx=np.array(idxs, dtype=np.intp),
        i=np.array(is_, dtype=np.intp),
        j=np.array(js_, dtype=np.intp),
        phi_inner=np.array(phi_in_list, dtype=float),
        phi_outer=np.array(phi_out_list, dtype=float),
    )
