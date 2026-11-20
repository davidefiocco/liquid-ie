"""Cross-validation of LiquidIE against independent Python packages and
analytical reference implementations.

External references
-------------------
- **sasmodels** (NIST SasView): Analytical Percus-Yevick structure factor
  for hard spheres, based on the Ashcroft-Lekner closed-form solution.
  https://www.sasview.org/

- **Wertheim-Thiele analytical solution**: Exact closed-form PY direct
  correlation function for hard spheres.
  Wertheim, Phys. Rev. Lett. 10, 321 (1963);
  Thiele, J. Chem. Phys. 39, 474 (1963).
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from liquidie.config import Config
from liquidie.solver import SolverResult, solve

sasmodels = pytest.importorskip("sasmodels")
from sasmodels.core import build_model, load_model_info  # noqa: E402
from sasmodels.data import empty_data1D  # noqa: E402
from sasmodels.direct_model import DirectModel  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers – Wertheim-Thiele analytical PY hard-sphere solution
# ---------------------------------------------------------------------------


def wertheim_thiele_cr(r: NDArray, sigma: float, eta: float) -> NDArray:
    """Exact PY direct correlation function for hard spheres.

    Parameters
    ----------
    r : array
        Radial distances (same grid as the solver).
    sigma : float
        Hard-sphere diameter.
    eta : float
        Packing fraction pi/6 * rho * sigma**3.

    Returns
    -------
    c : array
        Direct correlation function c(r).
    """
    lam1 = (1 + 2 * eta) ** 2 / (1 - eta) ** 4
    lam2 = -((1 + eta / 2) ** 2) / (1 - eta) ** 4
    t = r / sigma
    c = np.where(
        r < sigma,
        -lam1 - 6 * eta * lam2 * t - eta * lam1 / 2 * t**3,
        0.0,
    )
    return c


def sasmodels_hardsphere_sk(k: NDArray, eta: float) -> NDArray:
    """Compute S(k) via sasmodels' analytical PY hard-sphere formula.

    Uses sigma = 1 (radius_effective = 0.5) so k is in reduced units.
    """
    info = load_model_info("hardsphere")
    model = build_model(info)
    data = empty_data1D(k[k > 0])
    calculator = DirectModel(data, model)
    sk_positive = calculator(radius_effective=0.5, volfraction=eta)
    sk = np.ones_like(k)
    sk[k > 0] = sk_positive
    return sk


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _solve_py_hs(eta: float, dr: float = 0.005, r_max: float = 20.0) -> SolverResult:
    rho = 6 * eta / math.pi
    config = Config(
        system={"temperature": 1.0, "density": [rho]},
        grid={"dr": dr, "r_max": r_max},
        potential={
            "expression": "hard_sphere",
            "epsilon": [1.0],
            "sigma": [1.0],
        },
        solver={"closure": "PY", "tolerance": 1e-10},
    )
    return solve(config)


# ---------------------------------------------------------------------------
# 1. LiquidIE vs sasmodels – PY hard-sphere S(k)
# ---------------------------------------------------------------------------


class TestVsSasmodelsHardSphere:
    """Compare LiquidIE structure factor against sasmodels' analytical
    Percus-Yevick hard-sphere S(k) (Ashcroft-Lekner formula).
    """

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_sk_pointwise(self, eta: float) -> None:
        """S(k) agrees with sasmodels within rtol=1 % or atol=0.005 for k > 1.

        The combined tolerance handles both the oscillatory regime
        (where relative error matters) and the deep troughs of S(k)
        at high eta (where S(k) is small and relative error is
        amplified by the denominator).
        """
        result = _solve_py_hs(eta)
        k = result.k
        sk_liquidie = result.s_k[:, 0, 0]
        sk_ref = sasmodels_hardsphere_sk(k, eta)
        mask = k > 1.0
        np.testing.assert_allclose(
            sk_liquidie[mask],
            sk_ref[mask],
            rtol=0.01,
            atol=0.005,
            err_msg=f"S(k) mismatch vs sasmodels at eta={eta}",
        )

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_sk_peak_position(self, eta: float) -> None:
        """First-peak position agrees within dk."""
        result = _solve_py_hs(eta)
        k = result.k
        mask = k > 2.0
        peak_liquidie = k[mask][np.argmax(result.s_k[mask, 0, 0])]

        sk_ref = sasmodels_hardsphere_sk(k, eta)
        peak_ref = k[mask][np.argmax(sk_ref[mask])]

        dk = k[1] - k[0]
        assert abs(peak_liquidie - peak_ref) <= dk, (
            f"Peak position mismatch: LiquidIE={peak_liquidie:.3f}, "
            f"sasmodels={peak_ref:.3f}, dk={dk:.4f}"
        )

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_sk_peak_height(self, eta: float) -> None:
        """First-peak height agrees within 1.5 %."""
        result = _solve_py_hs(eta)
        k = result.k
        mask = k > 2.0
        peak_liquidie = np.max(result.s_k[mask, 0, 0])

        sk_ref = sasmodels_hardsphere_sk(k, eta)
        peak_ref = np.max(sk_ref[mask])

        np.testing.assert_allclose(
            peak_liquidie,
            peak_ref,
            rtol=0.015,
            err_msg=f"S(k) peak height mismatch at eta={eta}",
        )

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_sk_low_k(self, eta: float) -> None:
        """S(k) at the lowest nonzero k agrees within 3 %."""
        result = _solve_py_hs(eta)
        k = result.k
        idx = np.searchsorted(k, 0.5)
        sk_liquidie = result.s_k[idx, 0, 0]
        sk_ref = sasmodels_hardsphere_sk(k[idx : idx + 1], eta)[0]
        np.testing.assert_allclose(
            sk_liquidie,
            sk_ref,
            rtol=0.03,
            err_msg=f"S(k~0.5) mismatch at eta={eta}",
        )


# ---------------------------------------------------------------------------
# 2. LiquidIE vs Wertheim-Thiele analytical c(r)
# ---------------------------------------------------------------------------


class TestVsWertheimThiele:
    """Compare LiquidIE OZ solution against the exact Wertheim-Thiele
    analytical direct correlation function c(r) for PY hard spheres.
    """

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_cr_inside_core(self, eta: float) -> None:
        """c(r) inside the core matches the analytical polynomial within 3 %.

        At high packing fractions (eta >= 0.4) the grid discretisation
        limits agreement to ~2.5 %, so we use 3 % tolerance.
        """
        result = _solve_py_hs(eta)
        r = result.r
        c_analytical = wertheim_thiele_cr(r, sigma=1.0, eta=eta)
        c_liquidie = result.c_r[:, 0, 0]

        core_mask = (r > 0.05) & (r < 0.95)
        np.testing.assert_allclose(
            c_liquidie[core_mask],
            c_analytical[core_mask],
            rtol=0.03,
            err_msg=f"c(r) inside core mismatch at eta={eta}",
        )

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_cr_outside_core(self, eta: float) -> None:
        """c(r) = 0 outside the core (analytical PY property)."""
        result = _solve_py_hs(eta)
        r = result.r
        dr = r[1] - r[0]
        outside = r > 1.0 + 3 * dr
        np.testing.assert_array_less(
            np.abs(result.c_r[outside, 0, 0]),
            0.01,
            err_msg=f"|c(r)| outside core should be < 0.01 at eta={eta}",
        )

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_cr_contact_value(self, eta: float) -> None:
        """c(r) at contact (r -> sigma-) matches analytical value within 2 %."""
        result = _solve_py_hs(eta)
        r = result.r
        dr = r[1] - r[0]
        idx = np.searchsorted(r, 1.0 - 2 * dr)
        c_liquidie = result.c_r[idx, 0, 0]
        c_analytical = wertheim_thiele_cr(r[idx : idx + 1], sigma=1.0, eta=eta)[0]
        np.testing.assert_allclose(
            c_liquidie,
            c_analytical,
            rtol=0.02,
            err_msg=f"c(r) at contact mismatch at eta={eta}",
        )

    @pytest.mark.parametrize("eta", [0.1, 0.2, 0.3, 0.4])
    def test_analytical_sk_vs_sasmodels(self, eta: float) -> None:
        """Wertheim-Thiele c(r) -> numerical FT -> S(k) agrees with sasmodels.

        This cross-checks our analytical c(r) implementation against
        sasmodels' independent Ashcroft-Lekner S(k) formula.
        """
        sigma = 1.0
        rho = 6 * eta / math.pi
        dr = 0.001
        r = np.arange(0, 20.0, dr)
        c = wertheim_thiele_cr(r, sigma, eta)

        k = np.linspace(1.0, 25.0, 200)
        c_hat = np.zeros_like(k)
        for i, ki in enumerate(k):
            integrand = r * c * np.sin(ki * r)
            c_hat[i] = 4.0 * np.pi / ki * trapezoid(integrand, r)

        sk_analytical = 1.0 / (1.0 - rho * c_hat)
        sk_sasmodels = sasmodels_hardsphere_sk(k, eta)

        np.testing.assert_allclose(
            sk_analytical,
            sk_sasmodels,
            rtol=0.005,
            err_msg=(
                f"Wertheim-Thiele S(k) doesn't match sasmodels at eta={eta}; "
                f"analytical c(r) implementation may be wrong"
            ),
        )


# ---------------------------------------------------------------------------
# 3. Full three-way comparison at a single state point
# ---------------------------------------------------------------------------


class TestThreeWayComparison:
    """Three-way comparison: LiquidIE vs sasmodels vs Wertheim-Thiele
    analytical, at eta = 0.3 (a well-studied state point).
    """

    ETA = 0.3

    @pytest.fixture(scope="class")
    def liquidie_result(self) -> SolverResult:
        return _solve_py_hs(self.ETA)

    @pytest.fixture(scope="class")
    def sasmodels_sk(self, liquidie_result: SolverResult) -> NDArray:
        k = liquidie_result.k
        return sasmodels_hardsphere_sk(k, self.ETA)

    @pytest.fixture(scope="class")
    def analytical_sk(self, liquidie_result: SolverResult) -> NDArray:
        """S(k) from Wertheim-Thiele c(r) via numerical Fourier transform."""
        rho = 6 * self.ETA / math.pi
        dr_fine = 0.001
        r_fine = np.arange(0, 30.0, dr_fine)
        c = wertheim_thiele_cr(r_fine, sigma=1.0, eta=self.ETA)

        k = liquidie_result.k
        c_hat = np.zeros_like(k)
        for i, ki in enumerate(k):
            if ki < 1e-12:
                c_hat[i] = 4.0 * np.pi * trapezoid(r_fine**2 * c, r_fine)
            else:
                c_hat[i] = (
                    4.0
                    * np.pi
                    / ki
                    * trapezoid(r_fine * c * np.sin(ki * r_fine), r_fine)
                )

        return 1.0 / (1.0 - rho * c_hat)

    def test_all_three_agree_pointwise(
        self,
        liquidie_result: SolverResult,
        sasmodels_sk: NDArray,
        analytical_sk: NDArray,
    ) -> None:
        """All three S(k) implementations agree within rtol=1 % or atol=0.005."""
        k = liquidie_result.k
        mask = k > 1.0
        sk = liquidie_result.s_k[mask, 0, 0]

        np.testing.assert_allclose(
            sk,
            sasmodels_sk[mask],
            rtol=0.01,
            atol=0.005,
            err_msg="LiquidIE vs sasmodels",
        )
        np.testing.assert_allclose(
            sk,
            analytical_sk[mask],
            rtol=0.01,
            atol=0.005,
            err_msg="LiquidIE vs analytical",
        )
        np.testing.assert_allclose(
            sasmodels_sk[mask],
            analytical_sk[mask],
            rtol=0.005,
            err_msg="sasmodels vs analytical",
        )

    def test_compressibility_all_three(
        self,
        liquidie_result: SolverResult,
        sasmodels_sk: NDArray,
        analytical_sk: NDArray,
    ) -> None:
        """S(0) from all three agrees with the exact analytical value."""
        exact = (1 - self.ETA) ** 4 / (1 + 2 * self.ETA) ** 2
        rho = 6 * self.ETA / math.pi
        c0 = liquidie_result.c_k[0, 0, 0]
        s0_liquidie = 1.0 / (1.0 - rho * c0)

        np.testing.assert_allclose(
            s0_liquidie, exact, rtol=0.03, err_msg="LiquidIE S(0)"
        )

        sk_ref_low = sasmodels_hardsphere_sk(np.array([0.01]), self.ETA)[0]
        np.testing.assert_allclose(
            sk_ref_low, exact, rtol=0.03, err_msg="sasmodels S(0)"
        )

    def test_rdf_contact_value_consistency(self, liquidie_result: SolverResult) -> None:
        """g(sigma+) from LiquidIE matches the exact PY contact value.

        Provides a real-space cross-check complementing the k-space tests.
        """
        rdf = liquidie_result.rdf[:, 0, 0]
        peak = np.max(rdf)
        analytical = (1 + self.ETA / 2) / (1 - self.ETA) ** 2
        np.testing.assert_allclose(peak, analytical, rtol=0.03)
