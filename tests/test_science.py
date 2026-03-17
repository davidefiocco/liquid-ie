"""Scientific validation tests for the LiquidIE OZ solver.

These tests validate numerical solutions against known analytical results
from statistical mechanics, ensuring the solver produces physically correct
correlation functions for well-studied model systems.
"""

import math

import numpy as np
import pytest

from scipy.integrate import trapezoid

from liquidie.config import Config
from liquidie.mct import run_mct
from liquidie.solver import solve, write_results, SolverResult

# Module-level cache to avoid re-solving for the same eta across multiple tests
_RESULT_CACHE: dict[float, SolverResult] = {}


def _get_solver_result(eta: float) -> SolverResult:
    """Solve PY hard-sphere OZ for given packing fraction; cache result."""
    if eta not in _RESULT_CACHE:
        rho = 6 * eta / math.pi  # sigma=1
        config = Config(
            system={"temperature": 1.0, "density": [rho]},
            grid={"dr": 0.005, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-10},
        )
        _RESULT_CACHE[eta] = solve(config)
    return _RESULT_CACHE[eta]


@pytest.fixture(scope="module")
def cached_solve():
    """Return a function that returns (and caches) solver results by eta."""
    return _get_solver_result


class TestPYHardSphere:
    """Validate PY hard-sphere OZ solution against the exact analytical Wertheim-Thiele solution.

    References
    ----------
    - Wertheim, Phys. Rev. Lett. 10, 321 (1963)
    - Thiele, J. Chem. Phys. 39, 474 (1963)
    """

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_contact_value(self, eta, cached_solve):
        """Contact value g(sigma+) = (1 + eta/2) / (1 - eta)^2 within 3%."""
        result = cached_solve(eta)
        rdf = result.rdf[:, 0, 0]
        peak_value = np.max(rdf)
        analytical = (1 + eta / 2) / (1 - eta) ** 2
        np.testing.assert_allclose(peak_value, analytical, rtol=0.03)

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_compressibility_sum_rule(self, eta, cached_solve):
        """S(0) via the compressibility route: S(0) = 1/(1 - rho * c_hat(0)).

        For PY hard spheres c(r) = 0 for r > sigma, so c_hat(0) converges
        quickly with grid refinement.
        Analytical result: (1 - eta)^4 / (1 + 2*eta)^2.
        """
        result = cached_solve(eta)
        rho = 6 * eta / math.pi
        c0 = result.c_k[0, 0, 0]
        s0_compressibility = 1.0 / (1.0 - rho * c0)
        analytical = (1 - eta) ** 4 / (1 + 2 * eta) ** 2
        np.testing.assert_allclose(s0_compressibility, analytical, rtol=0.03)

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_direct_correlation_zero_outside_core(self, eta, cached_solve):
        """c(r) = 0 for r > sigma (PY property for hard spheres)."""
        result = cached_solve(eta)
        r, c_r = result.r, result.c_r[:, 0, 0]
        dr = 0.005
        sigma = 1.0
        outside_mask = r > sigma + 3 * dr
        np.testing.assert_array_less(
            np.abs(c_r[outside_mask]),
            0.01,
            err_msg=f"|c(r)| should be < 0.01 for r > {sigma + 3 * dr}",
        )

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_core_exclusion(self, eta, cached_solve):
        """g(r) ≈ 0 inside the hard core (0.1 < r < 0.9)."""
        result = cached_solve(eta)
        r, rdf = result.r, result.rdf[:, 0, 0]
        core_mask = (r > 0.1) & (r < 0.9)
        np.testing.assert_array_less(
            np.abs(rdf[core_mask]),
            0.05,
            err_msg="|g(r)| should be < 0.05 inside core",
        )


class TestMSHardSphere:
    """Validate Martynov-Sarkisov closure for hard spheres.

    MS should produce more accurate contact values than HNC at high
    packing fractions while sharing the same qualitative behaviour.
    The analytical PY contact value (1 + eta/2) / (1 - eta)^2 serves
    as a reference; MS is known to lie between PY and HNC.

    References
    ----------
    - Martynov & Sarkisov, Mol. Phys. 49, 1495 (1983)
    - Ballone et al., Mol. Phys. 59, 275 (1986)
    """

    @pytest.fixture(scope="class")
    def ms_result(self):
        rho = 6 * 0.3 / math.pi
        config = Config(
            system={"temperature": 1.0, "density": [rho]},
            grid={"dr": 0.005, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "MS", "tolerance": 1e-10},
        )
        return solve(config)

    @pytest.fixture(scope="class")
    def bpgg_s2_result(self):
        """BPGG with s=2 should give bitwise identical results to MS."""
        rho = 6 * 0.3 / math.pi
        config = Config(
            system={"temperature": 1.0, "density": [rho]},
            grid={"dr": 0.005, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={
                "closure": "BPGG",
                "closure_params": {"s": 2.0},
                "tolerance": 1e-10,
            },
        )
        return solve(config)

    def test_core_exclusion(self, ms_result):
        """g(r) ~ 0 inside the hard core."""
        r, rdf = ms_result.r, ms_result.rdf[:, 0, 0]
        core_mask = (r > 0.1) & (r < 0.9)
        np.testing.assert_array_less(
            np.abs(rdf[core_mask]),
            0.05,
            err_msg="MS: |g(r)| should be < 0.05 inside core",
        )

    def test_contact_value(self, ms_result):
        """MS contact value should be within 5% of PY analytical."""
        eta = 0.3
        rdf = ms_result.rdf[:, 0, 0]
        peak = np.max(rdf)
        analytical_py = (1 + eta / 2) / (1 - eta) ** 2
        np.testing.assert_allclose(peak, analytical_py, rtol=0.05)

    def test_structure_factor_positive(self, ms_result):
        """S(k) >= 0 for all k (thermodynamic stability)."""
        sk = ms_result.s_k[1:, 0, 0]
        np.testing.assert_array_less(-0.01, sk, err_msg="S(k) should be non-negative")

    def test_rdf_approaches_unity(self, ms_result):
        """g(r) -> 1 at large r."""
        r, rdf = ms_result.r, ms_result.rdf[:, 0, 0]
        mean_dev = np.mean(np.abs(rdf[r > 8.0] - 1.0))
        assert mean_dev < 0.02, f"|g(r)-1| mean = {mean_dev:.4f}, expected < 0.02"

    def test_bpgg_s2_matches_ms(self, ms_result, bpgg_s2_result):
        """BPGG(s=2) produces bitwise identical g(r) to MS preset."""
        np.testing.assert_array_equal(
            ms_result.rdf,
            bpgg_s2_result.rdf,
            err_msg="BPGG(s=2) should match MS exactly",
        )


class TestBPGGHardSphere:
    """Validate the BPGG closure family at intermediate s values.

    BPGG(s) interpolates between HNC (s=1) and MS (s=2).  The boundary
    cases are covered by TestMSHardSphere.test_bpgg_s2_matches_ms and
    test_expressions.py.  This class exercises the *interior* of the
    family: s=1 full-solve equivalence with HNC, intermediate s=1.5,
    and monotonicity of the contact value across the s parameter.

    State point: hard spheres, eta = 0.3, PY analytical contact value
    g(sigma+) = (1 + eta/2) / (1 - eta)^2 ~ 2.3469.

    References
    ----------
    - Ballone, Pastore, Galli, Gazzillo, Mol. Phys. 59, 275 (1986)
    """

    ETA = 0.3
    RHO = 6 * 0.3 / math.pi
    GRID = {"dr": 0.005, "r_max": 20.0}
    TOL = 1e-10

    @staticmethod
    def _solve_bpgg(rho, grid, s, tol):
        config = Config(
            system={"temperature": 1.0, "density": [rho]},
            grid=grid,
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={
                "closure": "BPGG",
                "closure_params": {"s": s},
                "tolerance": tol,
            },
        )
        return solve(config)

    @pytest.fixture(scope="class")
    def hnc_result(self):
        config = Config(
            system={"temperature": 1.0, "density": [self.RHO]},
            grid=self.GRID,
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "HNC", "tolerance": self.TOL},
        )
        return solve(config)

    @pytest.fixture(scope="class")
    def ms_result(self):
        config = Config(
            system={"temperature": 1.0, "density": [self.RHO]},
            grid=self.GRID,
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "MS", "tolerance": self.TOL},
        )
        return solve(config)

    @pytest.fixture(scope="class")
    def bpgg_s1_result(self):
        return self._solve_bpgg(self.RHO, self.GRID, s=1.0, tol=self.TOL)

    @pytest.fixture(scope="class")
    def bpgg_s1p5_result(self):
        return self._solve_bpgg(self.RHO, self.GRID, s=1.5, tol=self.TOL)

    # -- BPGG(s=1) == HNC full solve --

    def test_bpgg_s1_matches_hnc_rdf(self, hnc_result, bpgg_s1_result):
        """Full OZ solve: BPGG(s=1) g(r) matches HNC within solver tolerance.

        Not bitwise identical because SymPy evaluates (1+x)^1 - 1 via a
        different FP operation sequence than plain x, but the converged
        solutions agree to the solver tolerance.
        """
        np.testing.assert_allclose(
            bpgg_s1_result.rdf,
            hnc_result.rdf,
            atol=1e-9,
            err_msg="BPGG(s=1) should match HNC to solver tolerance",
        )

    def test_bpgg_s1_matches_hnc_sk(self, hnc_result, bpgg_s1_result):
        """Full OZ solve: BPGG(s=1) S(k) matches HNC within solver tolerance."""
        np.testing.assert_allclose(
            bpgg_s1_result.s_k,
            hnc_result.s_k,
            atol=1e-9,
            err_msg="BPGG(s=1) S(k) should match HNC to solver tolerance",
        )

    # -- BPGG(s=1.5) physics sanity --

    def test_bpgg_s1p5_core_exclusion(self, bpgg_s1p5_result):
        """g(r) ~ 0 inside the hard core for intermediate s."""
        r, rdf = bpgg_s1p5_result.r, bpgg_s1p5_result.rdf[:, 0, 0]
        core_mask = (r > 0.1) & (r < 0.9)
        np.testing.assert_array_less(
            np.abs(rdf[core_mask]),
            0.05,
            err_msg="BPGG(s=1.5): |g(r)| should be < 0.05 inside core",
        )

    def test_bpgg_s1p5_contact_between_hnc_and_ms(
        self, hnc_result, ms_result, bpgg_s1p5_result
    ):
        """BPGG(s=1.5) contact value lies strictly between HNC (s=1) and MS (s=2)."""
        peak_hnc = np.max(hnc_result.rdf[:, 0, 0])
        peak_ms = np.max(ms_result.rdf[:, 0, 0])
        peak_1p5 = np.max(bpgg_s1p5_result.rdf[:, 0, 0])
        lo, hi = sorted([peak_hnc, peak_ms])
        assert lo < peak_1p5 < hi, (
            f"BPGG(s=1.5) peak {peak_1p5:.6f} should be between "
            f"HNC {peak_hnc:.6f} and MS {peak_ms:.6f}"
        )

    def test_bpgg_s1p5_contact_physically_reasonable(self, bpgg_s1p5_result):
        """BPGG(s=1.5) contact value is in a physically reasonable range.

        HNC overestimates the HS contact value, so BPGG(s=1.5)
        (closer to HNC than MS) is expected to overshoot the PY
        analytical value.  We just check it's in a sane range.
        """
        peak = np.max(bpgg_s1p5_result.rdf[:, 0, 0])
        assert 2.0 < peak < 3.5, (
            f"BPGG(s=1.5) contact value {peak:.4f} outside reasonable range [2.0, 3.5]"
        )

    def test_bpgg_s1p5_structure_factor_positive(self, bpgg_s1p5_result):
        """S(k) >= 0 for all k (thermodynamic stability)."""
        sk = bpgg_s1p5_result.s_k[1:, 0, 0]
        np.testing.assert_array_less(
            -0.01, sk, err_msg="BPGG(s=1.5): S(k) should be non-negative"
        )

    def test_bpgg_s1p5_rdf_approaches_unity(self, bpgg_s1p5_result):
        """g(r) -> 1 at large r."""
        r, rdf = bpgg_s1p5_result.r, bpgg_s1p5_result.rdf[:, 0, 0]
        mean_dev = np.mean(np.abs(rdf[r > 8.0] - 1.0))
        assert mean_dev < 0.02, f"|g(r)-1| mean = {mean_dev:.4f}, expected < 0.02"

    # -- Monotonicity across s --

    def test_contact_monotonicity(self, hnc_result, bpgg_s1p5_result, ms_result):
        """Contact value varies monotonically from s=1 (HNC) through s=1.5 to s=2 (MS).

        For hard spheres at eta=0.3, HNC overestimates the contact value
        relative to MS, so the sequence should be strictly decreasing or
        strictly increasing -- no reversal.
        """
        peaks = [
            np.max(hnc_result.rdf[:, 0, 0]),
            np.max(bpgg_s1p5_result.rdf[:, 0, 0]),
            np.max(ms_result.rdf[:, 0, 0]),
        ]
        diffs = [peaks[1] - peaks[0], peaks[2] - peaks[1]]
        assert (diffs[0] > 0 and diffs[1] > 0) or (diffs[0] < 0 and diffs[1] < 0), (
            f"Contact values should be monotonic in s: "
            f"s=1 -> {peaks[0]:.6f}, s=1.5 -> {peaks[1]:.6f}, s=2 -> {peaks[2]:.6f}"
        )


class TestHNCLennardJones:
    """Validate HNC closure for the Lennard-Jones fluid at Verlet's state II.

    State point: T* = 2.74, rho* = 0.844. MD gives g(r) first peak ~ 2.50
    at r ~ 1.0 sigma. HNC typically underestimates this by 5-10%.

    References
    ----------
    - Verlet, Phys. Rev. 165, 201 (1968) — MD reference data
    - Hansen & McDonald, Theory of Simple Liquids, 4th ed.
    - Lado, Phys. Rev. A 8, 2548 (1973)
    """

    @pytest.fixture(scope="class")
    def hnc_result(self):
        config = Config(
            system={"temperature": 2.74, "density": [0.844]},
            grid={"dr": 0.01, "r_max": 20.0},
            potential={
                "expression": "lennard_jones",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "HNC", "tolerance": 1e-8},
        )
        return solve(config)

    @pytest.fixture(scope="class")
    def py_result(self):
        config = Config(
            system={"temperature": 2.74, "density": [0.844]},
            grid={"dr": 0.01, "r_max": 20.0},
            potential={
                "expression": "lennard_jones",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        return solve(config)

    def test_core_exclusion(self, hnc_result):
        """g(r) ~ 0 inside the LJ repulsive core (r < 0.85 sigma)."""
        r, rdf = hnc_result.r, hnc_result.rdf[:, 0, 0]
        core_mask = (r > 0.1) & (r < 0.85)
        np.testing.assert_array_less(
            np.abs(rdf[core_mask]),
            0.01,
            err_msg="g(r) should be < 0.01 in repulsive core",
        )

    def test_first_peak_position(self, hnc_result):
        """First peak of g(r) between 0.95 and 1.15 sigma."""
        r, rdf = hnc_result.r, hnc_result.rdf[:, 0, 0]
        mask = (r > 0.85) & (r < 2.0)
        peak_r = r[mask][np.argmax(rdf[mask])]
        assert 0.95 <= peak_r <= 1.15, f"Peak at r={peak_r}, expected [0.95, 1.15]"

    def test_first_peak_height(self, hnc_result):
        """g(r) first peak height between 2.0 and 2.8.

        MD gives ~2.50; HNC underestimates by 5-10%.
        """
        r, rdf = hnc_result.r, hnc_result.rdf[:, 0, 0]
        mask = (r > 0.85) & (r < 2.0)
        peak = np.max(rdf[mask])
        assert 2.0 <= peak <= 2.8, f"Peak height {peak:.3f}, expected [2.0, 2.8]"

    def test_structure_factor_peak_position(self, hnc_result):
        """S(k) first peak near k*sigma ~ 6.5-7.5."""
        k, sk = hnc_result.k, hnc_result.s_k[:, 0, 0]
        mask = k > 2.0
        peak_k = k[mask][np.argmax(sk[mask])]
        assert 6.0 <= peak_k <= 8.0, f"S(k) peak at k={peak_k}, expected [6.0, 8.0]"

    def test_structure_factor_positive(self, hnc_result):
        """S(k) >= 0 for all k (thermodynamic stability)."""
        sk = hnc_result.s_k[1:, 0, 0]
        np.testing.assert_array_less(-0.01, sk, err_msg="S(k) should be non-negative")

    def test_rdf_approaches_unity(self, hnc_result):
        """g(r) -> 1 for large r (mean deviation < 0.02 for r > 8)."""
        r, rdf = hnc_result.r, hnc_result.rdf[:, 0, 0]
        mean_dev = np.mean(np.abs(rdf[r > 8.0] - 1.0))
        assert mean_dev < 0.02, f"|g(r)-1| mean = {mean_dev:.4f}, expected < 0.02"

    def test_py_hnc_consistency(self, hnc_result, py_result):
        """PY and HNC first peaks agree within 5% at this state point."""
        for res, label in [(hnc_result, "HNC"), (py_result, "PY")]:
            r, rdf = res.r, res.rdf[:, 0, 0]
            mask = (r > 0.85) & (r < 2.0)
            peak = np.max(rdf[mask])
            assert peak > 2.0, f"{label} peak {peak:.3f} too low"
        r_h, rdf_h = hnc_result.r, hnc_result.rdf[:, 0, 0]
        r_p, rdf_p = py_result.r, py_result.rdf[:, 0, 0]
        mask_h = (r_h > 0.85) & (r_h < 2.0)
        mask_p = (r_p > 0.85) & (r_p < 2.0)
        peak_hnc = np.max(rdf_h[mask_h])
        peak_py = np.max(rdf_p[mask_p])
        np.testing.assert_allclose(peak_hnc, peak_py, rtol=0.05)


class TestPYBinaryHardSphere:
    """Validate PY binary hard-sphere OZ solution against Lebowitz exact contact values.

    Binary additive hard spheres: sigma_1=1.0, sigma_2=0.6, equimolar, eta=0.3.
    Lebowitz (1964) derived exact analytical contact values for the PY closure.

    References
    ----------
    - Lebowitz, Phys. Rev. 133, A895 (1964)
    - Ashcroft & Langreth, Phys. Rev. 156, 685 (1967)
    """

    @pytest.fixture(scope="class")
    def binary_result(self):
        # sigma_1=1.0, sigma_2=0.6, eta=0.3
        # xi_3 = (pi/6) * (rho_1*sigma_1^3 + rho_2*sigma_2^3), rho_1=rho_2=rho_total/2
        # 0.3 = (pi/12) * rho_total * (1.0 + 0.216) => rho_total = 0.3*12/(pi*1.216)
        rho_total = 0.3 * 12 / (math.pi * (1.0 + 0.6**3))
        rho_1 = rho_2 = rho_total / 2
        config = Config(
            system={"temperature": 1.0, "density": [rho_1, rho_2]},
            grid={"dr": 0.005, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0, 1.0, 1.0, 1.0],
                "sigma": [1.0, 0.8, 0.8, 0.6],
            },
            solver={"closure": "PY", "tolerance": 1e-10},
        )
        return solve(config)

    def test_contact_values(self, binary_result):
        """Contact values g_ij(sigma_ij) match Lebowitz analytical within 5%."""
        r, rdf = binary_result.r, binary_result.rdf
        sigma_1, sigma_2 = 1.0, 0.6
        rho_1 = rho_2 = 0.3 * 12 / (math.pi * (1.0 + 0.6**3)) / 2
        xi_2 = (math.pi / 6) * (rho_1 * sigma_1**2 + rho_2 * sigma_2**2)
        xi_3 = 0.3
        denom = (1 - xi_3) ** 2

        pairs = [
            (0, 0, 1.0, sigma_1, sigma_1),
            (0, 1, 0.8, sigma_1, sigma_2),
            (1, 0, 0.8, sigma_2, sigma_1),
            (1, 1, 0.6, sigma_2, sigma_2),
        ]
        for i, j, sigma_ij, s_i, s_j in pairs:
            mask = (r > sigma_ij - 0.1) & (r < sigma_ij + 0.5)
            peak_value = np.max(rdf[mask, i, j])
            analytical = 1 / (1 - xi_3) + 3 * s_i * s_j / (s_i + s_j) * xi_2 / denom
            np.testing.assert_allclose(peak_value, analytical, rtol=0.05)

    def test_direct_correlation_zero_outside_core(self, binary_result):
        """|c_ij(r)| < 0.01 for r > sigma_ij + 3*dr (PY property)."""
        r, c_r = binary_result.r, binary_result.c_r
        dr = 0.005
        sigma = [[1.0, 0.8], [0.8, 0.6]]
        for i in range(2):
            for j in range(2):
                sigma_ij = sigma[i][j]
                outside_mask = r > sigma_ij + 3 * dr
                np.testing.assert_array_less(
                    np.abs(c_r[outside_mask, i, j]),
                    0.01,
                    err_msg=f"|c_{i}{j}(r)| should be < 0.01 for r > {sigma_ij + 3 * dr}",
                )

    def test_core_exclusion(self, binary_result):
        """g_ij(r) < 0.05 for 0.1 < r < 0.9*sigma_ij (inside core)."""
        r, rdf = binary_result.r, binary_result.rdf
        sigma = [[1.0, 0.8], [0.8, 0.6]]
        for i in range(2):
            for j in range(2):
                sigma_ij = sigma[i][j]
                core_mask = (r > 0.1) & (r < 0.9 * sigma_ij)
                np.testing.assert_array_less(
                    np.abs(rdf[core_mask, i, j]),
                    0.05,
                    err_msg=f"|g_{i}{j}(r)| should be < 0.05 inside core",
                )

    def test_symmetry(self, binary_result):
        """g_12(r) ≈ g_21(r) and c_12(r) ≈ c_21(r) within 1e-10."""
        rdf = binary_result.rdf
        c_r = binary_result.c_r
        np.testing.assert_allclose(rdf[:, 0, 1], rdf[:, 1, 0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(c_r[:, 0, 1], c_r[:, 1, 0], rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# Testbed 4A – Square-well fluid with PY closure
# ---------------------------------------------------------------------------

SW_POTENTIAL = "Piecewise((10000, r < sigma), (-epsilon, r < 1.5 * sigma), (0, True))"


class TestPYSquareWell:
    """Validate PY closure for the square-well potential (lambda = 1.5).

    Tests a custom Piecewise potential expression with a hard core (r < sigma)
    and an attractive well (-epsilon for sigma < r < 1.5*sigma).

    References
    ----------
    - Barker & Henderson, J. Chem. Phys. 47, 4714 (1967)
    """

    @pytest.fixture(scope="class")
    def sw_result(self):
        config = Config(
            system={"temperature": 2.0, "density": [0.3]},
            grid={"dr": 0.01, "r_max": 20.0},
            potential={
                "expression": SW_POTENTIAL,
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        return solve(config)

    @pytest.fixture(scope="class")
    def hs_result(self):
        """PY hard-sphere at the same density for comparison."""
        config = Config(
            system={"temperature": 1.0, "density": [0.3]},
            grid={"dr": 0.01, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        return solve(config)

    @pytest.fixture(scope="class")
    def sw_high_t_result(self):
        """Square-well at very high T (should approach hard-sphere)."""
        config = Config(
            system={"temperature": 100.0, "density": [0.3]},
            grid={"dr": 0.01, "r_max": 20.0},
            potential={
                "expression": SW_POTENTIAL,
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        return solve(config)

    def test_core_exclusion(self, sw_result):
        """g(r) ~ 0 inside the hard core (r < 0.9 sigma)."""
        r, rdf = sw_result.r, sw_result.rdf[:, 0, 0]
        core = rdf[(r > 0.1) & (r < 0.9)]
        np.testing.assert_array_less(
            np.abs(core), 0.01, err_msg="g(r) should be ~0 inside core"
        )

    def test_contact_peak(self, sw_result):
        """Contact peak should be > 1.5 (attractive well enhances contact)."""
        r, rdf = sw_result.r, sw_result.rdf[:, 0, 0]
        mask = (r > 0.9) & (r < 1.5)
        peak = np.max(rdf[mask])
        assert peak > 1.5, f"Contact peak {peak:.3f} should be > 1.5"

    def test_well_boundary_discontinuity(self, sw_result):
        """g(r) shows a clear jump at the well boundary r = 1.5 sigma.

        Inside the well g(r) is enhanced by exp(epsilon/T) relative to
        outside, so g just below 1.5 should exceed g just above 1.5.
        """
        r, rdf = sw_result.r, sw_result.rdf[:, 0, 0]
        dr = 0.01
        idx_below = np.argmin(np.abs(r - (1.5 - 2 * dr)))
        idx_above = np.argmin(np.abs(r - (1.5 + 2 * dr)))
        g_below = rdf[idx_below]
        g_above = rdf[idx_above]
        assert g_below > g_above, (
            f"g({r[idx_below]:.3f})={g_below:.4f} should exceed "
            f"g({r[idx_above]:.3f})={g_above:.4f} at well boundary"
        )
        jump = g_below - g_above
        assert jump > 0.3, f"Well boundary jump {jump:.3f} should be > 0.3"

    def test_sw_contact_exceeds_hs(self, sw_result, hs_result):
        """Attractive well enhances the contact value relative to pure HS."""
        r_sw, rdf_sw = sw_result.r, sw_result.rdf[:, 0, 0]
        r_hs, rdf_hs = hs_result.r, hs_result.rdf[:, 0, 0]
        mask_sw = (r_sw > 0.9) & (r_sw < 1.5)
        mask_hs = (r_hs > 0.9) & (r_hs < 1.5)
        peak_sw = np.max(rdf_sw[mask_sw])
        peak_hs = np.max(rdf_hs[mask_hs])
        assert peak_sw > peak_hs, (
            f"SW peak {peak_sw:.3f} should exceed HS peak {peak_hs:.3f}"
        )

    def test_high_temperature_limit(self, sw_high_t_result, hs_result):
        """At T >> epsilon, the square-well g(r) approaches the HS result.

        Contact values should agree within 1%.
        """
        r_ht, rdf_ht = sw_high_t_result.r, sw_high_t_result.rdf[:, 0, 0]
        _, rdf_hs = hs_result.r, hs_result.rdf[:, 0, 0]
        mask = (r_ht > 0.9) & (r_ht < 1.5)
        peak_ht = np.max(rdf_ht[mask])
        peak_hs = np.max(rdf_hs[mask])
        np.testing.assert_allclose(peak_ht, peak_hs, rtol=0.01)

    def test_rdf_approaches_unity(self, sw_result):
        """g(r) -> 1 at large r."""
        r, rdf = sw_result.r, sw_result.rdf[:, 0, 0]
        mean_dev = np.mean(np.abs(rdf[r > 8.0] - 1.0))
        assert mean_dev < 0.01, f"|g(r)-1| mean = {mean_dev:.4f}, expected < 0.01"


# ---------------------------------------------------------------------------
# Testbed 4B – MCT hard-sphere glass transition
# ---------------------------------------------------------------------------


class TestMCTHardSphere:
    """Validate MCT non-ergodicity parameter for the hard-sphere glass transition.

    The MCT critical packing fraction with PY input is eta_c ~ 0.516.
    Below: F(q) -> 0 (ergodic liquid). Above: F(q) > 0 (ideal glass).

    Uses a coarse 200-point grid (dr=0.05, r_max=10) to keep the Numba
    MCT kernel tractable (O(n_pts^3) per Picard step).

    References
    ----------
    - Bengtzelius, Gotze, Sjolander, J. Phys. C 17, 5915 (1984)
    - Gotze, Complex Dynamics of Glass-Forming Liquids (Oxford, 2009)
    - Franosch et al., Phys. Rev. E 55, 7153 (1997)
    """

    @staticmethod
    def _solve_and_mct(eta: float, n_iterations: int = 15) -> tuple:
        rho = 6 * eta / math.pi
        config = Config(
            system={"temperature": 1.0, "density": [rho]},
            grid={"dr": 0.05, "r_max": 10.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        result = solve(config)
        f = run_mct(result, config=config, method="picard", n_iterations=n_iterations)
        return result, f

    @pytest.fixture(scope="class")
    def ergodic(self):
        return self._solve_and_mct(0.40)

    @pytest.fixture(scope="class")
    def glass(self):
        return self._solve_and_mct(0.53)

    def test_ergodic_f_vanishes(self, ergodic):
        """Below eta_c, F(q) should decay to zero (ergodic liquid)."""
        _, f = ergodic
        max_f = np.max(np.abs(f[:, 0, 0]))
        assert max_f < 0.01, f"Ergodic max |F(q)| = {max_f:.4e}, expected ~0"

    def test_glass_f_nonzero(self, glass):
        """Above eta_c, F(q) should be robustly non-zero (glass)."""
        _, f = glass
        max_f = np.max(f[:, 0, 0])
        assert max_f > 0.5, f"Glass max F(q) = {max_f:.4f}, expected > 0.5"

    def test_glass_f_bounded_by_s(self, glass):
        """F(q) <= S(q) at all q (physical constraint)."""
        result, f = glass
        sk = result.s_k[:, 0, 0]
        fq = f[:, 0, 0]
        violations = fq > sk + 0.01
        assert not np.any(violations), (
            f"F(q) exceeds S(q)+0.01 at {np.sum(violations)} q-points; "
            f"max overshoot = {np.max(fq - sk):.4e}"
        )

    def test_glass_f_peaks_near_s_peak(self, glass):
        """F(q) peak should be near the S(k) first peak (k ~ 6-8)."""
        result, f = glass
        k = result.k
        dk = k[1] - k[0]
        mask = k > 2.0
        f_peak_k = k[mask][np.argmax(f[mask, 0, 0])]
        s_peak_k = k[mask][np.argmax(result.s_k[mask, 0, 0])]
        assert abs(f_peak_k - s_peak_k) < 1.0, (
            f"F peak at k={f_peak_k:.2f}, S peak at k={s_peak_k:.2f}, dk={dk:.3f}"
        )


# ---------------------------------------------------------------------------
# Testbed 5 – OZ self-consistency relations
# ---------------------------------------------------------------------------


class TestOZConsistency:
    """Internal self-consistency checks on the OZ solver output.

    These verify algebraic identities that must hold for any converged
    OZ solution, independent of external reference data.  They catch bugs
    in the post-processing pipeline (Fourier transforms, k-division,
    real-space decomposition, structure-factor formula).
    """

    @pytest.fixture(scope="class")
    def single(self):
        """PY hard-sphere, single component, eta = 0.3."""
        rho = 6 * 0.3 / math.pi
        config = Config(
            system={"temperature": 1.0, "density": [rho]},
            grid={"dr": 0.005, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-10},
        )
        return solve(config), rho

    @pytest.fixture(scope="class")
    def binary(self):
        """PY binary hard-sphere from Testbed 3 state point."""
        rho_total = 0.3 * 12 / (math.pi * (1.0 + 0.6**3))
        rho_1 = rho_2 = rho_total / 2
        config = Config(
            system={"temperature": 1.0, "density": [rho_1, rho_2]},
            grid={"dr": 0.005, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0, 1.0, 1.0, 1.0],
                "sigma": [1.0, 0.8, 0.8, 0.6],
            },
            solver={"closure": "PY", "tolerance": 1e-10},
        )
        return solve(config), np.array([rho_1, rho_2])

    def test_real_space_decomposition(self, single):
        """g(r) = c(r) + gamma(r) + 1  at all r."""
        result, _ = single
        lhs = result.rdf[:, 0, 0]
        rhs = result.c_r[:, 0, 0] + result.gamma_r[:, 0, 0] + 1.0
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_fourier_space_decomposition(self, single):
        """h(k) = c(k) + gamma(k)  at all k."""
        result, _ = single
        lhs = result.h_k[:, 0, 0]
        rhs = result.c_k[:, 0, 0] + result.gamma_k[:, 0, 0]
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_compressibility_route_all_k(self, single):
        """S(k) = 1/(1 - rho*c(k)) at all k > 0 (single-component OZ)."""
        result, rho = single
        sk = result.s_k[1:, 0, 0]
        ck = result.c_k[1:, 0, 0]
        sk_oz = 1.0 / (1.0 - rho * ck)
        np.testing.assert_allclose(sk, sk_oz, rtol=1e-12)

    def test_binary_oz_relation(self, binary):
        """H(k) = inv(I - C*rho) * C  for the binary mixture at all k > 0.

        This tests the matrix OZ algebra and the 2x2 matrix inverse in
        linalg.invv.
        """
        from liquidie.linalg import dotve, dotvbs, invv

        result, rho_vec = binary
        rho_diag = np.diag(rho_vec)
        n_pts = len(result.k)
        eye_stack = np.tile(np.eye(2), (n_pts - 1, 1, 1))

        ck = result.c_k[1:]
        hk = result.h_k[1:]
        hk_oz = dotve(invv(eye_stack - dotvbs(ck, rho_diag)), ck)
        np.testing.assert_allclose(hk, hk_oz, atol=1e-13)

    def test_binary_sk_formula(self, binary):
        """S_ij(k) = x_i*delta_ij + x_i*x_j*rho_total*h_ij(k) at all k."""
        result, rho_vec = binary
        rho_total = rho_vec.sum()
        x = rho_vec / rho_total
        for i in range(2):
            for j in range(2):
                expected = (
                    x[i] * (1.0 if i == j else 0.0)
                    + x[i] * x[j] * rho_total * result.h_k[:, i, j]
                )
                np.testing.assert_allclose(
                    result.s_k[:, i, j],
                    expected,
                    atol=1e-14,
                    err_msg=f"S_{i}{j}(k) mismatch",
                )


# ---------------------------------------------------------------------------
# Testbed 6 – MCT for binary hard spheres
# ---------------------------------------------------------------------------


class TestMCTBinaryHS:
    """Test the MCT Numba kernel with n_species = 2.

    The get_m kernel has 6 nested species loops that are invisible
    in the single-component case (all reduce to scalars).  This testbed
    exercises the full multi-species indexing at an ergodic density
    where F(q) must converge to zero.
    """

    @pytest.fixture(scope="class")
    def binary_mct(self):
        rho_total = 0.3 * 12 / (math.pi * (1.0 + 0.6**3))
        rho_1 = rho_2 = rho_total / 2
        config = Config(
            system={"temperature": 1.0, "density": [rho_1, rho_2]},
            grid={"dr": 0.05, "r_max": 10.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0, 1.0, 1.0, 1.0],
                "sigma": [1.0, 0.8, 0.8, 0.6],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        result = solve(config)
        f = run_mct(result, config=config, method="picard", n_iterations=15)
        return f

    @pytest.mark.slow
    def test_ergodic_all_components_vanish(self, binary_mct):
        """At eta=0.3 (ergodic), F_ij(q) -> 0 for all i, j."""
        f = binary_mct
        for i in range(2):
            for j in range(2):
                max_f = np.max(np.abs(f[:, i, j]))
                assert max_f < 0.01, (
                    f"F_{i}{j} max = {max_f:.4e}, expected ~0 in ergodic phase"
                )

    @pytest.mark.slow
    def test_symmetry_f12_f21(self, binary_mct):
        """F_12(q) = F_21(q) (MCT symmetry)."""
        f = binary_mct
        np.testing.assert_allclose(
            f[:, 0, 1],
            f[:, 1, 0],
            atol=1e-12,
            err_msg="F_12 and F_21 should be symmetric",
        )

    @pytest.mark.slow
    def test_diagonal_non_negative(self, binary_mct):
        """F_ii(q) >= 0 (physical constraint on diagonal)."""
        f = binary_mct
        for i in range(2):
            assert np.all(f[:, i, i] >= -1e-12), (
                f"F_{i}{i} has negative values: min = {np.min(f[:, i, i]):.4e}"
            )


# ---------------------------------------------------------------------------
# Testbed 7 – Yukawa fluid (custom potential expression)
# ---------------------------------------------------------------------------

YUKAWA_POTENTIAL = "epsilon * sigma * exp(-r / sigma) / r"


class TestYukawaFluid:
    """Validate the solver with a custom Yukawa potential.

    Tests the SymPy expression parser on a physically meaningful
    long-range, continuous, soft-core potential.  epsilon=10 at T=1,
    rho=0.5 produces a well-structured liquid.

    References
    ----------
    - Hamaguchi, Farouki & Dubin, Phys. Rev. E 56, 4671 (1997)
    """

    @pytest.fixture(scope="class")
    def py_yukawa(self):
        config = Config(
            system={"temperature": 1.0, "density": [0.5]},
            grid={"dr": 0.01, "r_max": 20.0},
            potential={
                "expression": YUKAWA_POTENTIAL,
                "epsilon": [10.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        return solve(config)

    @pytest.fixture(scope="class")
    def hnc_yukawa(self):
        config = Config(
            system={"temperature": 1.0, "density": [0.5]},
            grid={"dr": 0.01, "r_max": 20.0},
            potential={
                "expression": YUKAWA_POTENTIAL,
                "epsilon": [10.0],
                "sigma": [1.0],
            },
            solver={"closure": "HNC", "tolerance": 1e-8},
        )
        return solve(config)

    def test_soft_core_exclusion(self, py_yukawa):
        """g(r) ~ 0 for r < 0.5 (soft but strongly repulsive core)."""
        r, rdf = py_yukawa.r, py_yukawa.rdf[:, 0, 0]
        core = rdf[(r > 0.01) & (r < 0.5)]
        assert np.max(np.abs(core)) < 0.01, (
            f"max |g(r)| for r<0.5: {np.max(np.abs(core)):.4f}"
        )

    def test_first_peak(self, py_yukawa):
        """g(r) has a first peak between r = 0.8 and 2.0, height > 1.2."""
        r, rdf = py_yukawa.r, py_yukawa.rdf[:, 0, 0]
        mask = (r > 0.8) & (r < 2.0)
        peak = np.max(rdf[mask])
        peak_r = r[mask][np.argmax(rdf[mask])]
        assert peak > 1.2, f"Peak height {peak:.3f}, expected > 1.2"
        assert 0.8 < peak_r < 2.0, f"Peak at r={peak_r:.3f}"

    def test_structure_factor_peaked(self, py_yukawa):
        """S(k) has a clear first peak for k > 1."""
        k, sk = py_yukawa.k, py_yukawa.s_k[:, 0, 0]
        mask = k > 1.0
        sk_peak = np.max(sk[mask])
        assert sk_peak > 1.3, f"S(k) peak = {sk_peak:.3f}, expected > 1.3"

    def test_structure_factor_positive(self, py_yukawa):
        """S(k) > 0 for all k > 0 (thermodynamic stability)."""
        sk = py_yukawa.s_k[1:, 0, 0]
        np.testing.assert_array_less(-0.01, sk, err_msg="S(k) should be non-negative")

    def test_rdf_approaches_unity(self, py_yukawa):
        """g(r) -> 1 at large r."""
        r, rdf = py_yukawa.r, py_yukawa.rdf[:, 0, 0]
        mean_dev = np.mean(np.abs(rdf[r > 8.0] - 1.0))
        assert mean_dev < 0.02, f"|g(r)-1| mean = {mean_dev:.4f}"

    def test_py_hnc_comparison(self, py_yukawa, hnc_yukawa):
        """PY and HNC both produce structured liquid; peaks differ < 30%."""
        for res, label in [(py_yukawa, "PY"), (hnc_yukawa, "HNC")]:
            r, rdf = res.r, res.rdf[:, 0, 0]
            mask = (r > 0.8) & (r < 2.0)
            peak = np.max(rdf[mask])
            assert peak > 1.0, f"{label} Yukawa peak {peak:.3f} too low"

        r_p, rdf_p = py_yukawa.r, py_yukawa.rdf[:, 0, 0]
        r_h, rdf_h = hnc_yukawa.r, hnc_yukawa.rdf[:, 0, 0]
        mask_p = (r_p > 0.8) & (r_p < 2.0)
        mask_h = (r_h > 0.8) & (r_h < 2.0)
        peak_py = np.max(rdf_p[mask_p])
        peak_hnc = np.max(rdf_h[mask_h])
        np.testing.assert_allclose(peak_py, peak_hnc, rtol=0.30)


# ---------------------------------------------------------------------------
# Testbed 8 – Invariance and expression parsing
# ---------------------------------------------------------------------------


class TestInvarianceAndParsing:
    """Verify expression parser correctness and physics invariances.

    Catches bugs in registry lookup, SymPy symbol ordering, temperature
    leakage into hard-sphere results, and multi-species algebra.
    """

    @pytest.fixture(scope="class")
    def _base_params(self):
        eta = 0.2
        rho = 6 * eta / math.pi
        return {
            "rho": rho,
            "grid": {"dr": 0.02, "r_max": 20.0},
            "tolerance": 1e-8,
        }

    @pytest.fixture(scope="class")
    def shortcut_result(self, _base_params):
        """PY hard-sphere via shortcut names."""
        p = _base_params
        config = Config(
            system={"temperature": 1.0, "density": [p["rho"]]},
            grid=p["grid"],
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": p["tolerance"]},
        )
        return solve(config)

    @pytest.fixture(scope="class")
    def raw_result(self, _base_params):
        """PY hard-sphere via raw SymPy expression strings."""
        p = _base_params
        config = Config(
            system={"temperature": 1.0, "density": [p["rho"]]},
            grid=p["grid"],
            potential={
                "expression": "Piecewise((10000, r < sigma), (0, True))",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={
                "closure": "(r + gamma_r) * (exp(-inv_t * phi) - 1)",
                "tolerance": p["tolerance"],
            },
        )
        return solve(config)

    def test_expression_parsing_closure(self, shortcut_result, raw_result):
        """Raw PY expression string gives bitwise identical g(r)."""
        np.testing.assert_array_equal(
            shortcut_result.rdf,
            raw_result.rdf,
            err_msg="Raw closure expression should match 'PY' shortcut exactly",
        )

    def test_expression_parsing_potential(self, shortcut_result, raw_result):
        """Raw hard_sphere Piecewise gives bitwise identical c(r)."""
        np.testing.assert_array_equal(
            shortcut_result.c_r,
            raw_result.c_r,
            err_msg="Raw potential expression should match 'hard_sphere' shortcut",
        )

    def test_temperature_invariance(self, _base_params):
        """Hard-sphere g(r) is independent of temperature.

        The Boltzmann factor exp(-V_core/T) is effectively 0 for any realistic T.
        """
        p = _base_params
        results = {}
        for temp in [0.5, 1.0, 5.0]:
            config = Config(
                system={"temperature": temp, "density": [p["rho"]]},
                grid=p["grid"],
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0],
                    "sigma": [1.0],
                },
                solver={"closure": "PY", "tolerance": p["tolerance"]},
            )
            results[temp] = solve(config)

        ref_rdf = results[1.0].rdf[:, 0, 0]
        for temp in [0.5, 5.0]:
            np.testing.assert_allclose(
                results[temp].rdf[:, 0, 0],
                ref_rdf,
                atol=1e-6,
                err_msg=f"HS g(r) should be T-independent; T={temp} differs",
            )

    def test_identical_species_reduction(self, shortcut_result, _base_params):
        """Binary mixture with identical species reduces to single species.

        sigma=[1,1,1,1], rho=[rho/2, rho/2] must give
        g_11 = g_12 = g_22 = g_single(rho).
        """
        p = _base_params
        rho_half = p["rho"] / 2
        config_bin = Config(
            system={"temperature": 1.0, "density": [rho_half, rho_half]},
            grid=p["grid"],
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0, 1.0, 1.0, 1.0],
                "sigma": [1.0, 1.0, 1.0, 1.0],
            },
            solver={"closure": "PY", "tolerance": p["tolerance"]},
        )
        result_bin = solve(config_bin)

        g11 = result_bin.rdf[:, 0, 0]
        g12 = result_bin.rdf[:, 0, 1]
        g22 = result_bin.rdf[:, 1, 1]
        g_single = shortcut_result.rdf[:, 0, 0]

        np.testing.assert_allclose(
            g11, g12, atol=1e-9, err_msg="g_11 != g_12 for identical species"
        )
        np.testing.assert_allclose(
            g11, g22, atol=1e-9, err_msg="g_11 != g_22 for identical species"
        )
        np.testing.assert_allclose(
            g11,
            g_single,
            atol=1e-6,
            err_msg="Binary identical-species != single-species",
        )


# ---------------------------------------------------------------------------
# Testbed 9 – Write/read and restart round-trip
# ---------------------------------------------------------------------------


class TestWriteReadRestart:
    """Verify that write_results produces correct files and that restarting
    from the written gamma.dat reproduces the original solution.

    Tests the full I/O pipeline: write_results (solver.py:199-228) and
    the restart path (solver.py:80-84), including the transpose/reshape
    logic for multi-component gamma.
    """

    @pytest.fixture(scope="class")
    def single_result(self):
        rho = 6 * 0.2 / math.pi
        config = Config(
            system={"temperature": 1.0, "density": [rho]},
            grid={"dr": 0.02, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        return solve(config), config

    @pytest.fixture(scope="class")
    def binary_result(self):
        rho_total = 0.3 * 12 / (math.pi * (1.0 + 0.6**3))
        rho_1 = rho_2 = rho_total / 2
        config = Config(
            system={"temperature": 1.0, "density": [rho_1, rho_2]},
            grid={"dr": 0.02, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0, 1.0, 1.0, 1.0],
                "sigma": [1.0, 0.8, 0.8, 0.6],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        return solve(config), config

    def test_single_file_contents(self, single_result, tmp_path):
        """Written output files reproduce SolverResult arrays exactly."""
        result, _ = single_result
        write_results(result, tmp_path)

        rdf_data = np.loadtxt(tmp_path / "rdf00.dat")
        np.testing.assert_array_equal(rdf_data[:, 0], result.r)
        np.testing.assert_array_equal(rdf_data[:, 1], result.rdf[:, 0, 0])

        s_data = np.loadtxt(tmp_path / "s00.dat")
        np.testing.assert_array_equal(s_data[:, 0], result.k)
        np.testing.assert_array_equal(s_data[:, 1], result.s_k[:, 0, 0])

        c_data = np.loadtxt(tmp_path / "c00.dat")
        np.testing.assert_array_equal(c_data[:, 1], result.c_k[:, 0, 0])

        h_data = np.loadtxt(tmp_path / "h00.dat")
        np.testing.assert_array_equal(h_data[:, 1], result.h_k[:, 0, 0])

    def test_single_restart(self, single_result, tmp_path):
        """Restarting from converged gamma reproduces the solution."""
        result, config = single_result
        write_results(result, tmp_path)

        restart_cfg = Config(
            system=config.system.model_dump(),
            grid=config.grid.model_dump(),
            potential=config.potential.model_dump(),
            solver=config.solver.model_dump(),
            restart={"enabled": True, "file": str(tmp_path / "gamma.dat")},
        )
        result2 = solve(restart_cfg)
        np.testing.assert_allclose(
            result2.rdf[:, 0, 0],
            result.rdf[:, 0, 0],
            atol=1e-6,
            err_msg="Restarted g(r) should match original",
        )

    def test_binary_file_contents(self, binary_result, tmp_path):
        """Binary output files contain correct partial correlation data."""
        result, _ = binary_result
        write_results(result, tmp_path)

        for i, j in [(0, 0), (0, 1), (1, 1)]:
            rdf_data = np.loadtxt(tmp_path / f"rdf{i}{j}.dat")
            np.testing.assert_array_equal(
                rdf_data[:, 1],
                result.rdf[:, i, j],
                err_msg=f"rdf{i}{j}.dat mismatch",
            )
            s_data = np.loadtxt(tmp_path / f"s{i}{j}.dat")
            np.testing.assert_array_equal(
                s_data[:, 1],
                result.s_k[:, i, j],
                err_msg=f"s{i}{j}.dat mismatch",
            )

    def test_binary_restart(self, binary_result, tmp_path):
        """Binary restart preserves off-diagonal gamma (transpose check)."""
        result, config = binary_result
        write_results(result, tmp_path)

        restart_cfg = Config(
            system=config.system.model_dump(),
            grid=config.grid.model_dump(),
            potential=config.potential.model_dump(),
            solver=config.solver.model_dump(),
            restart={"enabled": True, "file": str(tmp_path / "gamma.dat")},
        )
        result2 = solve(restart_cfg)
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(
                    result2.rdf[:, i, j],
                    result.rdf[:, i, j],
                    atol=1e-6,
                    err_msg=f"Binary restart g_{i}{j}(r) mismatch",
                )

    def test_gamma_file_shape(self, binary_result, tmp_path):
        """gamma.dat has (n_pts, 1 + n_species^2) columns."""
        result, _ = binary_result
        write_results(result, tmp_path)
        gamma_data = np.loadtxt(tmp_path / "gamma.dat")
        n_pts = len(result.r)
        assert gamma_data.shape == (n_pts, 5), (
            f"gamma.dat shape {gamma_data.shape}, expected ({n_pts}, 5)"
        )


# ---------------------------------------------------------------------------
# Testbed 10 – Newton-Krylov MCT solver
# ---------------------------------------------------------------------------


class TestMCTNewtonKrylov:
    """Exercise the run_mct Newton-Krylov code path (mct.py:117-130).

    The residual function, flatten/reshape logic, and scipy Newton-Krylov
    wrapper for MCT are tested on the ergodic phase where convergence is
    fast.  The glass phase is skipped because NK from the default F=1
    initial guess is impractically slow without warm-start support.
    """

    @pytest.fixture(scope="class")
    def ergodic_nk(self):
        rho = 6 * 0.40 / math.pi
        config = Config(
            system={"temperature": 1.0, "density": [rho]},
            grid={"dr": 0.05, "r_max": 10.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        result = solve(config)
        f = run_mct(result, config=config, method="newton_krylov", tolerance=1e-4)
        return f

    def test_ergodic_f_vanishes(self, ergodic_nk):
        """NK MCT at eta=0.40 (ergodic): F(q) -> 0."""
        max_f = np.max(np.abs(ergodic_nk[:, 0, 0]))
        assert max_f < 0.01, f"NK ergodic max |F(q)| = {max_f:.4e}, expected ~0"

    def test_ergodic_f_shape(self, ergodic_nk):
        """F has correct shape (n_pts, 1, 1) for single component."""
        assert ergodic_nk.shape[1:] == (1, 1)

    def test_nk_matches_picard_ergodic(self, ergodic_nk):
        """NK and Picard agree in the ergodic phase (both -> 0)."""
        rho = 6 * 0.40 / math.pi
        config = Config(
            system={"temperature": 1.0, "density": [rho]},
            grid={"dr": 0.05, "r_max": 10.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        result = solve(config)
        f_picard = run_mct(result, config=config, method="picard", n_iterations=15)
        np.testing.assert_allclose(
            ergodic_nk[:, 0, 0],
            f_picard[:, 0, 0],
            atol=0.01,
            err_msg="NK and Picard should agree in ergodic phase",
        )


# ---------------------------------------------------------------------------
# Testbed 10b – MCT mass invariance
# ---------------------------------------------------------------------------


class TestMCTMasses:
    """Verify that the MCT non-ergodicity parameter F(q) is mass-independent.

    In MCT the memory kernel M_{ij}(q) acquires a factor 1/m_i (mct_kernel.py)
    while the fluctuating-force matrix N_{ij} = m_i/(T x_i) M_{ij} (mct.py)
    multiplies it back in.  The masses cancel exactly, so F(q) is purely
    structural.  Any indexing mismatch between get_m and compute_f would
    break this cancellation.
    """

    @staticmethod
    def _solve_hs(eta, dr=0.05, r_max=10.0):
        rho = 6 * eta / math.pi
        config = Config(
            system={"temperature": 1.0, "density": [rho]},
            grid={"dr": dr, "r_max": r_max},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        return solve(config), config

    @pytest.fixture(scope="class")
    def ergodic_pair(self):
        """F(q) at eta=0.40 with unit and non-unit masses."""
        result, config = self._solve_hs(0.40)
        f_unit = run_mct(result, config=config, method="picard", n_iterations=15)
        f_heavy = run_mct(
            result,
            config=config,
            method="picard",
            n_iterations=15,
            masses=np.array([5.0]),
        )
        return f_unit, f_heavy

    @pytest.fixture(scope="class")
    def glass_pair(self):
        """F(q) at eta=0.53 with unit and non-unit masses."""
        result, config = self._solve_hs(0.53)
        f_unit = run_mct(result, config=config, method="picard", n_iterations=15)
        f_heavy = run_mct(
            result,
            config=config,
            method="picard",
            n_iterations=15,
            masses=np.array([3.0]),
        )
        return f_unit, f_heavy

    @pytest.fixture(scope="class")
    def binary_pair(self):
        """Binary HS F(q) with unit and asymmetric masses."""
        rho_total = 0.3 * 12 / (math.pi * (1.0 + 0.6**3))
        rho_1 = rho_2 = rho_total / 2
        config = Config(
            system={"temperature": 1.0, "density": [rho_1, rho_2]},
            grid={"dr": 0.05, "r_max": 10.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0, 1.0, 1.0, 1.0],
                "sigma": [1.0, 0.8, 0.8, 0.6],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        result = solve(config)
        f_unit = run_mct(result, config=config, method="picard", n_iterations=15)
        f_asym = run_mct(
            result,
            config=config,
            method="picard",
            n_iterations=15,
            masses=np.array([2.0, 0.5]),
        )
        return f_unit, f_asym

    def test_ergodic_mass_invariance(self, ergodic_pair):
        """Ergodic F(q) is identical regardless of mass."""
        f_unit, f_heavy = ergodic_pair
        np.testing.assert_allclose(
            f_heavy,
            f_unit,
            atol=1e-12,
            err_msg="Ergodic F(q) should be mass-independent",
        )

    def test_glass_mass_invariance(self, glass_pair):
        """Glass F(q) is identical regardless of mass."""
        f_unit, f_heavy = glass_pair
        np.testing.assert_allclose(
            f_heavy,
            f_unit,
            atol=1e-12,
            err_msg="Glass F(q) should be mass-independent",
        )

    @pytest.mark.slow
    def test_binary_mass_invariance(self, binary_pair):
        """Binary F(q) is identical with asymmetric masses."""
        f_unit, f_asym = binary_pair
        np.testing.assert_allclose(
            f_asym,
            f_unit,
            atol=1e-12,
            err_msg="Binary F(q) should be mass-independent",
        )


# ---------------------------------------------------------------------------
# Testbed 11 – Real-space thermodynamic sum rule
# ---------------------------------------------------------------------------


class TestRealSpaceSumRule:
    """Cross-check S(0) from real-space integration against Fourier-space.

    S(0) = 1 + rho * 4*pi * integral(r^2 * h(r), r) must agree with
    the Fourier-space s_k[0].  This catches truncation errors at r_max
    and any mismatch between the SFT and numerical quadrature.
    """

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_single_component(self, eta, cached_solve):
        """Real-space S(0) matches Fourier-space S(0) and analytics."""
        result = cached_solve(eta)
        rho = 6 * eta / math.pi
        r = result.r
        h_r = result.rdf[:, 0, 0] - 1.0

        s0_real = 1.0 + rho * 4.0 * np.pi * trapezoid(r**2 * h_r, r)
        s0_fourier = result.s_k[0, 0, 0]
        analytical = (1 - eta) ** 4 / (1 + 2 * eta) ** 2

        np.testing.assert_allclose(
            s0_real,
            s0_fourier,
            rtol=1e-4,
            err_msg=f"Real vs Fourier S(0) at eta={eta}",
        )
        np.testing.assert_allclose(
            s0_real,
            analytical,
            rtol=0.03,
            err_msg=f"Real-space S(0) vs analytical at eta={eta}",
        )

    def test_binary_component(self):
        """Real-space S_ij(0) matches Fourier-space for binary HS."""
        rho_total = 0.3 * 12 / (math.pi * (1.0 + 0.6**3))
        rho_1 = rho_2 = rho_total / 2
        config = Config(
            system={"temperature": 1.0, "density": [rho_1, rho_2]},
            grid={"dr": 0.005, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0, 1.0, 1.0, 1.0],
                "sigma": [1.0, 0.8, 0.8, 0.6],
            },
            solver={"closure": "PY", "tolerance": 1e-10},
        )
        result = solve(config)
        r = result.r
        x = np.array([rho_1, rho_2]) / rho_total

        for i in range(2):
            for j in range(2):
                h_ij = result.rdf[:, i, j] - 1.0
                s0_real = x[i] * (1.0 if i == j else 0.0) + x[i] * x[
                    j
                ] * rho_total * 4.0 * np.pi * trapezoid(r**2 * h_ij, r)
                np.testing.assert_allclose(
                    s0_real,
                    result.s_k[0, i, j],
                    rtol=1e-4,
                    err_msg=f"Binary S_{i}{j}(0) real vs Fourier",
                )


# ---------------------------------------------------------------------------
# Testbed 12 – CLI end-to-end
# ---------------------------------------------------------------------------

_CLI_TOML = """\
[system]
temperature = 1.0
density = [0.3]

[grid]
dr = 0.05
r_max = 10.0

[potential]
expression = "hard_sphere"
epsilon = [1.0]
sigma = [1.0]

[solver]
closure = "PY"
tolerance = 1e-8
"""


class TestCLI:
    """End-to-end test of the Typer CLI commands.

    Exercises load_config, write_results, and the MCT CLI's file-based
    SolverResult reconstruction (cli.py:110-156).
    """

    @pytest.fixture(scope="class")
    def cli_output(self, tmp_path_factory):
        from typer.testing import CliRunner
        from liquidie.cli import app

        runner = CliRunner()
        tmp = tmp_path_factory.mktemp("cli")
        toml_path = tmp / "test.toml"
        toml_path.write_text(_CLI_TOML)

        solve_result = runner.invoke(
            app, ["solve", "-c", str(toml_path), "-o", str(tmp)]
        )
        mct_result = runner.invoke(
            app,
            ["mct", "-c", str(toml_path), "-i", str(tmp), "-o", str(tmp), "-n", "3"],
        )
        return tmp, solve_result, mct_result

    def test_solve_exit_code(self, cli_output):
        _, solve_result, _ = cli_output
        assert solve_result.exit_code == 0, f"solve failed: {solve_result.output}"

    def test_solve_stdout(self, cli_output):
        _, solve_result, _ = cli_output
        assert "Results written" in solve_result.output

    def test_solve_output_files(self, cli_output):
        tmp, _, _ = cli_output
        for name in ["gamma.dat", "rdf00.dat", "c00.dat", "h00.dat", "s00.dat"]:
            path = tmp / name
            assert path.exists(), f"Missing {name}"
            data = np.loadtxt(path)
            assert data.ndim == 2 and data.shape[1] == 2, (
                f"{name} shape {data.shape}, expected (n, 2)"
            )

    def test_solve_n_pts(self, cli_output):
        """Output row count matches grid: r_max/dr = 10/0.05 = 200."""
        tmp, _, _ = cli_output
        data = np.loadtxt(tmp / "rdf00.dat")
        assert data.shape[0] == 200

    def test_mct_exit_code(self, cli_output):
        _, _, mct_result = cli_output
        assert mct_result.exit_code == 0, f"mct failed: {mct_result.output}"

    def test_mct_stdout(self, cli_output):
        _, _, mct_result = cli_output
        assert "MCT results written" in mct_result.output

    def test_mct_output_file(self, cli_output):
        tmp, _, _ = cli_output
        path = tmp / "f00.dat"
        assert path.exists(), "Missing f00.dat"
        data = np.loadtxt(path)
        assert data.ndim == 2 and data.shape[1] == 2
        assert data.shape[0] == 200
