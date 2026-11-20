"""Tests for the SymPy-based expression parsing (closures and potentials)."""

import numpy as np
import pytest

from liquidie.expressions import (
    CLOSURE_SYMBOLS,
    HARD_CORE_HEIGHT,
    KNOWN_CLOSURES,
    apply_closure_vec,
    build_closure,
    build_expression,
    build_potential,
    find_potential_jumps,
    generate_potential_grid,
)


class TestBuildExpression:
    def test_known_name_expanded(self):
        """PY preset evaluates to the correct formula."""
        fn = build_expression("PY", KNOWN_CLOSURES, CLOSURE_SYMBOLS)
        # fn args: (gamma_r, inv_t, phi, r) — sorted alphabetical
        result = fn(0.0, 1.0, np.array([0.5]), np.array([1.0]))
        expected = (1.0 + 0.0) * (np.exp(-1.0 * 0.5) - 1)
        np.testing.assert_allclose(result, expected)

    def test_raw_expression(self):
        """Raw SymPy expression evaluates correctly."""
        fn = build_expression("r + phi", KNOWN_CLOSURES, CLOSURE_SYMBOLS)
        # fn args: (gamma_r, inv_t, phi, r) — sorted alphabetical
        result = fn(0.0, 0.0, np.array([3.0, 4.0]), np.array([1.0, 2.0]))
        np.testing.assert_allclose(result, np.array([4.0, 6.0]))

    def test_unknown_symbol_raises(self):
        with pytest.raises(ValueError, match="unknown symbol"):
            build_expression("r + unknown_var", KNOWN_CLOSURES, CLOSURE_SYMBOLS)

    def test_malformed_expression_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            build_expression("r @@@ phi", KNOWN_CLOSURES, CLOSURE_SYMBOLS)


class TestClosures:
    def test_py_closure_values(self):
        fn = build_closure("PY")
        r = np.array([1.0, 2.0])
        gamma_r = np.array([0.1, 0.2])
        phi = np.array([0.5, 0.3])
        inv_t = 1.0
        # (r + gamma_r) * (exp(-inv_t * phi) - 1)
        expected = (r + gamma_r) * (np.exp(-inv_t * phi) - 1)
        # fn args: (gamma_r, inv_t, phi, r) — sorted alphabetical
        result = fn(gamma_r, inv_t, phi, r)
        np.testing.assert_allclose(result, expected)

    def test_hnc_closure_values(self):
        fn = build_closure("HNC")
        r = np.array([1.0, 2.0])
        gamma_r = np.array([0.1, 0.2])
        phi = np.array([0.5, 0.3])
        inv_t = 1.0
        expected = r * np.exp(-inv_t * phi + gamma_r / r) - gamma_r - r
        result = fn(gamma_r, inv_t, phi, r)
        np.testing.assert_allclose(result, expected)

    def test_custom_closure(self):
        fn = build_closure("r * phi")
        r = np.array([2.0, 3.0])
        result = fn(0.0, 0.0, np.array([4.0, 5.0]), r)
        np.testing.assert_allclose(result, r * np.array([4.0, 5.0]))

    def test_ms_closure_values(self):
        fn = build_closure("MS")
        r = np.array([1.0, 2.0])
        gamma_r = np.array([0.1, 0.2])
        phi = np.array([0.5, 0.3])
        inv_t = 1.0
        expected = (
            r * np.exp(-inv_t * phi + np.sqrt(1 + 2 * gamma_r / r) - 1) - gamma_r - r
        )
        result = fn(gamma_r, inv_t, phi, r)
        np.testing.assert_allclose(result, expected)

    def test_bpgg_s1_matches_hnc(self):
        """BPGG with s=1 is algebraically identical to HNC."""
        fn_hnc = build_closure("HNC")
        fn_bpgg = build_closure("BPGG", params={"s": 1.0})
        r = np.array([1.0, 2.0, 3.0])
        gamma_r = np.array([0.1, 0.2, 0.3])
        phi = np.array([0.5, 0.3, 0.1])
        inv_t = 1.0
        result_hnc = fn_hnc(gamma_r, inv_t, phi, r)
        result_bpgg = fn_bpgg(gamma_r, inv_t, phi, r)
        np.testing.assert_allclose(result_bpgg, result_hnc)

    def test_bpgg_s2_matches_ms(self):
        """BPGG with s=2 is algebraically identical to MS."""
        fn_ms = build_closure("MS")
        fn_bpgg = build_closure("BPGG", params={"s": 2.0})
        r = np.array([1.0, 2.0, 3.0])
        gamma_r = np.array([0.1, 0.2, 0.3])
        phi = np.array([0.5, 0.3, 0.1])
        inv_t = 1.0
        result_ms = fn_ms(gamma_r, inv_t, phi, r)
        result_bpgg = fn_bpgg(gamma_r, inv_t, phi, r)
        np.testing.assert_allclose(result_bpgg, result_ms)

    def test_bpgg_without_param_raises(self):
        """BPGG without setting s raises a clear error."""
        with pytest.raises(ValueError, match="closure_params"):
            build_closure("BPGG")

    def test_closure_case_typo_suggests_preset(self):
        with pytest.raises(ValueError, match="did you mean preset 'PY'"):
            build_closure("py")

    def test_apply_closure_vec(self):
        """apply_closure_vec produces correct shape and values."""
        fn = build_closure("PY")
        r = np.array([0.0, 1.0, 2.0])
        gamma_r = np.zeros((3, 1, 1))
        phi = np.ones((3, 1, 1)) * 0.5
        inv_t = 1.0
        cr = apply_closure_vec(fn, r, gamma_r, phi, inv_t)
        assert cr.shape == (3, 1, 1)
        # PY: (r + gamma_r) * (exp(-inv_t * phi) - 1)
        # At r=1, gamma_r=0, phi=0.5: (1+0)*(exp(-0.5)-1)
        expected_r1 = 1.0 * (np.exp(-0.5) - 1.0)
        np.testing.assert_allclose(cr[1, 0, 0], expected_r1)
        # At r=2, gamma_r=0, phi=0.5: (2+0)*(exp(-0.5)-1)
        expected_r2 = 2.0 * (np.exp(-0.5) - 1.0)
        np.testing.assert_allclose(cr[2, 0, 0], expected_r2)


class TestPotentials:
    def test_hard_sphere(self):
        fn = build_potential("hard_sphere")
        r = np.array([0.3, 0.5, 0.8, 1.0, 1.5])
        # epsilon=1, r=..., sigma=1.0
        result = fn(1.0, r, 1.0)
        assert result[0] == HARD_CORE_HEIGHT  # r < sigma
        assert result[-1] == 0.0  # r >= sigma

    def test_lennard_jones(self):
        fn = build_potential("lennard_jones")
        r = np.array([1.0, 2.0])
        sigma = 1.0
        epsilon = 1.0
        expected = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
        result = fn(epsilon, r, sigma)
        np.testing.assert_allclose(result, expected)

    def test_custom_potential(self):
        fn = build_potential("epsilon * (sigma / r)**12")
        r = np.array([1.0, 2.0])
        result = fn(1.0, r, 1.0)
        np.testing.assert_allclose(result, 1.0 / r**12)

    def test_inverse_power_law(self):
        fn = build_potential("inverse_power_law", params={"n": 6.0})
        r = np.array([1.0, 2.0])
        np.testing.assert_allclose(fn(2.0, r, 1.5), 2.0 * (1.5 / r) ** 6)

    def test_inverse_power_law_without_param_raises(self):
        with pytest.raises(ValueError, match="potential_params"):
            build_potential("inverse_power_law")

    def test_gaussian_core(self):
        fn = build_potential("gaussian_core")
        r = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(fn(2.0, r, 1.5), 2.0 * np.exp(-((r / 1.5) ** 2)))

    def test_off_grid_jump_logs_warning(self, caplog):
        fn = build_potential("hard_sphere")
        r = np.arange(0.0, 5.0, 0.05)
        with caplog.at_level("WARNING", logger="liquidie"):
            jumps = find_potential_jumps(
                "hard_sphere", fn, r, [1.02], [1.0], 1, inv_t=1.0
            )
        assert jumps.is_empty
        assert "falls between grid nodes" in caplog.text

    def test_generate_potential_grid(self):
        fn = build_potential("hard_sphere")
        r = np.linspace(0.1, 3.0, 50)
        phi = generate_potential_grid(fn, r, [1.0], [1.0], n_species=1)
        assert phi.shape == (50, 1, 1)
        assert phi[0, 0, 0] == HARD_CORE_HEIGHT  # r=0.1 < sigma=1.0
        assert phi[-1, 0, 0] == 0.0  # r=3.0 >= sigma=1.0

    def test_find_potential_jumps_hard_sphere(self):
        fn = build_potential("hard_sphere")
        r = np.arange(0.0, 5.0, 0.05)
        jumps = find_potential_jumps("hard_sphere", fn, r, [1.0], [1.0], 1, inv_t=1.0)
        assert list(jumps.idx) == [20]
        assert jumps.phi_inner[0] == HARD_CORE_HEIGHT
        assert jumps.phi_outer[0] == 0.0

    def test_find_potential_jumps_square_well(self):
        spec = "Piecewise((1e30, r < sigma), (-epsilon, r < 1.5 * sigma), (0, True))"
        fn = build_potential(spec)
        r = np.arange(0.0, 5.0, 0.05)
        jumps = find_potential_jumps(spec, fn, r, [1.0], [1.0], 1, inv_t=1.0)
        assert sorted(jumps.idx.tolist()) == [20, 30]

    def test_find_potential_jumps_smooth_is_empty(self):
        fn = build_potential("lennard_jones")
        r = np.arange(0.0, 5.0, 0.05)
        jumps = find_potential_jumps(
            "lennard_jones", fn, r, [1.0], [1.0], 1, inv_t=100.0
        )
        assert jumps.is_empty
