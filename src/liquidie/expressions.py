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
from typing import Any

import numpy as np
import sympy
from numpy.typing import NDArray

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
}

CLOSURE_SYMBOLS = {"r", "gamma_r", "phi", "inv_t"}
POTENTIAL_SYMBOLS = {"r", "sigma", "epsilon"}

# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_expression(
    spec: str,
    registry: dict[str, str],
    allowed_symbols: set[str],
    params: dict[str, float] | None = None,
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
        raise ValueError(
            f"Expression contains unknown symbol(s) {unexpected}. "
            f"Allowed: {allowed_symbols} (set unresolved params via closure_params)"
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
    return build_expression(spec, KNOWN_CLOSURES, CLOSURE_SYMBOLS, params)


def build_potential(spec: str) -> Callable[..., Any]:
    """Build a potential callable from a preset name or SymPy expression.

    The returned function has signature ``f(epsilon, r, sigma)``
    (arguments in sorted alphabetical order).
    """
    return build_expression(spec, KNOWN_POTENTIALS, POTENTIAL_SYMBOLS)


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
    for i in range(n_species):
        for j in range(n_species):
            vals = closure_fn(gamma_r[:, i, j], inv_t, phi[:, i, j], r)
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
            cr[:, i, j] = vals
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
    for i in range(n_species):
        for j in range(n_species):
            phi[:, i, j] = potential_fn(eps_mat[i, j], r, sigma_mat[i, j])
    return phi
