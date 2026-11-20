"""Quantitative MCT checks against the exact Percus-Yevick hard-sphere S(q).

Feeding the analytical PY structure factor straight into :func:`run_mct`
isolates the MCT discretisation from the OZ solver.

References
----------
- Wertheim, Phys. Rev. Lett. 10, 321 (1963); Thiele, J. Chem. Phys. 39, 474 (1963)
- Franosch, Fuchs, Gotze, Mayr, Singh, Phys. Rev. E 55, 7153 (1997):
  eta_c = 0.51591 for PY hard spheres
- Nauroth & Kob, Phys. Rev. E 55, 657 (1997)
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from liquidie.mct import run_mct
from liquidie.solver import SolverResult

DK = 0.4
N_PTS = 100

_NODES, _WEIGHTS = leggauss(200)
_R = 0.5 * (_NODES + 1.0)
_W = 0.5 * _WEIGHTS


def py_ck(k: NDArray, eta: float, derivative: bool = False) -> NDArray:
    """Exact PY hard-sphere c(k) for sigma = 1, or its k-derivative."""
    lam1 = (1 + 2 * eta) ** 2 / (1 - eta) ** 4
    lam2 = -((1 + eta / 2) ** 2) / (1 - eta) ** 4
    c_r = -lam1 - 6 * eta * lam2 * _R - 0.5 * eta * lam1 * _R**3
    kr = np.outer(k, _R)
    safe = np.where(kr > 0, kr, 1.0)
    if derivative:
        kernel = np.where(kr > 0, (np.cos(safe) * safe - np.sin(safe)) / safe**2, 0.0)
        kernel = kernel * _R
    else:
        kernel = np.where(kr > 0, np.sin(safe) / safe, 1.0)
    return 4 * np.pi * (kernel * (_R**2 * c_r * _W)).sum(axis=1)


def exact_py_result(
    eta: float, x: NDArray | None = None, dk: float = DK, n_pts: int = N_PTS
) -> SolverResult:
    """SolverResult carrying the exact PY S(q).

    With mole fractions *x*, the fluid is split into identical species, for
    which S_ij = x_i delta_ij + x_i x_j (S - 1) exactly.
    """
    k = np.arange(n_pts) * dk
    rho = 6 * eta / np.pi
    s_one = 1.0 / (1.0 - rho * py_ck(k, eta))
    x = np.array([1.0]) if x is None else np.asarray(x)
    n = len(x)
    s_k = (
        x[None, :, None] * np.eye(n)[None]
        + np.outer(x, x)[None] * (s_one - 1)[:, None, None]
    )
    zeros = np.zeros_like(s_k)
    return SolverResult(
        r=np.arange(n_pts) * np.pi / (n_pts * dk),
        k=k,
        gamma_r=zeros,
        gamma_k=zeros,
        c_r=zeros,
        c_k=zeros,
        rdf=zeros,
        h_k=zeros,
        s_k=s_k,
        n_species=n,
        density=rho * x,
        temperature=1.0,
    )


def converged_f(result: SolverResult, **kwargs) -> NDArray:
    return run_mct(result, tolerance=1e-11, n_iterations=20000, strict=True, **kwargs)


class TestCriticalPackingFraction:
    """eta_c from exact PY S(q) brackets the literature value 0.51591."""

    def test_liquid_below_eta_c(self):
        result = exact_py_result(0.513)
        f = converged_f(result)[:, 0, 0]
        assert np.max(np.abs(f)) < 1e-8

    def test_glass_above_eta_c(self):
        result = exact_py_result(0.518)
        f = converged_f(result)[:, 0, 0] / result.s_k[:, 0, 0]
        assert np.max(f) > 0.5


class TestZeroWavevector:
    """F(q=0) is the finite q -> 0 limit, not zero, in a one-component glass."""

    ETA = 0.53

    @pytest.fixture(scope="class")
    def glass(self):
        result = exact_py_result(self.ETA)
        return result, converged_f(result)[:, 0, 0]

    def test_bounded_by_structure_factor(self, glass):
        result, f = glass
        assert 0.0 < f[0] <= result.s_k[0, 0, 0]

    def test_matches_exact_limit(self, glass):
        """F(0) = S(0) - 1/(1/S(0) + K(0)), with the vertex expanded to O(q).

        For one component V(q, k) = q [c(k) + k mu^2 c'(k)] + O(q^2), so
        K(0) = rho/(8 pi^2) int k^2 F(k)^2 [2c^2 + 4/3 k c c' + 2/5 k^2 c'^2] dk.
        Here c'(k) is analytic, whereas run_mct differentiates c(k) on the
        grid at second order, which differs by about 0.2% at dk = 0.4.
        """
        result, f = glass
        k = result.k
        rho = 6 * self.ETA / np.pi
        c = py_ck(k, self.ETA)
        dc = py_ck(k, self.ETA, derivative=True)
        integrand = k**2 * f**2 * (2 * c**2 + 4 / 3 * k * c * dc + 0.4 * k**2 * dc**2)
        k0 = rho / (8 * np.pi**2) * trapezoid(integrand, k)
        s0 = result.s_k[0, 0, 0]
        np.testing.assert_allclose(f[0], s0 - 1.0 / (1.0 / s0 + k0), rtol=5e-3)

    def test_liquid_is_zero(self):
        f = converged_f(exact_py_result(0.45))[:, 0, 0]
        assert abs(f[0]) < 1e-10


class TestMixtureInvariants:
    """A one-component glass split into two identical species.

    The collective F must be reproduced, F must be symmetric with
    0 <= F <= S as matrices, and it must not depend on the masses.
    """

    ETA = 0.53
    X = np.array([0.3, 0.7])

    @pytest.fixture(scope="class")
    def single(self):
        return converged_f(exact_py_result(self.ETA))[:, 0, 0]

    @pytest.fixture(scope="class")
    def binary(self):
        result = exact_py_result(self.ETA, x=self.X)
        f_light = converged_f(result, masses=np.array([1.0, 1.0]))
        f_heavy = converged_f(result, masses=np.array([1.0, 5.0]))
        return result, f_light, f_heavy

    def test_species_splitting(self, single, binary):
        _, f_light, _ = binary
        np.testing.assert_allclose(f_light.sum(axis=(1, 2)), single, atol=1e-8)

    def test_symmetry(self, binary):
        _, f_light, _ = binary
        np.testing.assert_allclose(f_light[:, 0, 1], f_light[:, 1, 0], atol=1e-12)

    def test_bounds(self, binary):
        result, f_light, _ = binary
        sym = 0.5 * (f_light + f_light.transpose(0, 2, 1))
        assert np.min(np.linalg.eigvalsh(sym)) > -1e-10
        assert np.min(np.linalg.eigvalsh(result.s_k - sym)) > -1e-10

    def test_mass_independence(self, binary):
        _, f_light, f_heavy = binary
        np.testing.assert_allclose(f_heavy, f_light, atol=1e-8)


class TestMctKMax:
    """k_max truncates the wavevector grid cleanly without losing accuracy."""

    def test_k_max_matches_full_grid(self):
        full_result = exact_py_result(0.53, dk=0.4, n_pts=150)  # k up to 59.6
        short_result = exact_py_result(0.53, dk=0.4, n_pts=101)  # k up to 40.0
        f_full = converged_f(full_result)
        f_short = converged_f(short_result)
        f_cut = converged_f(full_result, k_max=40.0)
        assert f_cut.shape == (101, 1, 1)
        # Slicing via k_max is identical to running on the truncated grid
        np.testing.assert_allclose(f_cut, f_short, rtol=1e-12, atol=1e-12)
        # Physically close to the wider k-range grid
        np.testing.assert_allclose(f_cut, f_full[:101], atol=1e-2)

    def test_k_max_invalid_raises(self):
        result = exact_py_result(0.45, dk=0.4, n_pts=20)
        with pytest.raises(ValueError, match="k_max must be positive"):
            run_mct(result, k_max=-1.0)
        with pytest.raises(ValueError, match="at least 3 wavevectors"):
            run_mct(result, k_max=0.5)


class TestBinaryGlassMct:
    """Converged non-trivial binary hard-sphere glass (delta = 0.6, equimolar)."""

    def test_binary_glass_invariants_and_nk_agreement(self):
        from liquidie.config import Config
        from liquidie.solver import solve

        # Total packing fraction eta = 0.56 with sigma = (1.0, 0.6), x = (0.5, 0.5)
        eta = 0.56
        rho_each = 6.0 * eta / (np.pi * (1.0**3 + 0.6**3))
        cfg = Config(
            system={"temperature": 1.0, "density": [rho_each, rho_each]},
            grid={"dr": 0.02, "r_max": 10.24},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0, 1.0, 1.0, 1.0],
                "sigma": [1.0, 0.8, 0.8, 0.6],
            },
            solver={"closure": "PY", "tolerance": 1e-10},
        )
        oz = solve(cfg, strict=True)
        f_picard = run_mct(
            oz,
            method="picard",
            k_max=35.0,
            tolerance=1e-9,
            n_iterations=5000,
            strict=True,
        )
        f_nk = run_mct(
            oz,
            method="newton_krylov",
            k_max=35.0,
            tolerance=1e-9,
            strict=True,
        )

        # Arrested state (non-zero plateau)
        assert np.max(f_picard[:, 0, 0]) > 0.2
        assert np.max(f_picard[:, 1, 1]) > 0.1
        # Symmetry and matrix bounds 0 <= F <= S
        np.testing.assert_allclose(f_picard[:, 0, 1], f_picard[:, 1, 0], atol=1e-12)
        s_cut = oz.s_k[: len(f_picard)]
        sym = 0.5 * (f_picard + f_picard.transpose(0, 2, 1))
        assert np.min(np.linalg.eigvalsh(sym)) > -1e-10
        assert np.min(np.linalg.eigvalsh(s_cut - sym)) > -1e-10
        # Picard and Newton-Krylov converge to the same glass fixed point
        np.testing.assert_allclose(f_nk, f_picard, atol=1e-7)
