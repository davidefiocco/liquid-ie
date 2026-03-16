"""Tests for the DST-I and Spherical Fourier Transform."""

import numpy as np

from liquidie.transforms import dst_i, sft


class TestDstI:
    def test_known_values(self):
        """DST-I of [1, 1, 1, 1] matches analytical sum.

        y[k] = sum_{n=1}^{4} sin(pi*k*n/5) for k=1..4.
        """
        g = np.array([1.0, 1.0, 1.0, 1.0])
        result = dst_i(g)
        expected = np.array(
            [sum(np.sin(np.pi * k * n / 5) for n in range(1, 5)) for k in range(1, 5)]
        )
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_single_element(self):
        """DST-I of [1.0]: y[1] = sin(pi/2) = 1.0."""
        g = np.array([1.0])
        result = dst_i(g)
        np.testing.assert_allclose(result, np.array([1.0]), atol=1e-14)

    def test_linearity(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        alpha = 2.5
        np.testing.assert_allclose(
            dst_i(alpha * g1 + g2),
            alpha * dst_i(g1) + dst_i(g2),
            atol=1e-12,
        )

    def test_self_inverse(self):
        """DST-I is its own inverse up to a factor of (N+1)/2."""
        g = np.array([1.0, 3.0, 2.0, 5.0, 0.7])
        n = len(g)
        roundtrip = dst_i(dst_i(g))
        np.testing.assert_allclose(roundtrip, g * (n + 1) / 2, atol=1e-12)


class TestSft:
    def test_output_shape(self):
        n_pts = 64
        n_species = 2
        r = np.arange(n_pts) * 0.1
        g = np.zeros((n_pts, n_species, n_species))
        k, f = sft(r, g)
        assert k.shape == (n_pts,)
        assert f.shape == (n_pts, n_species, n_species)

    def test_zero_input_gives_zero_output(self):
        n_pts = 32
        r = np.arange(n_pts) * 0.1
        g = np.zeros((n_pts, 1, 1))
        k, f = sft(r, g)
        np.testing.assert_allclose(f, 0.0, atol=1e-15)

    def test_k_grid_spacing(self):
        n_pts = 100
        dr = 0.05
        r = np.arange(n_pts) * dr
        g = np.zeros((n_pts, 1, 1))
        k, _ = sft(r, g)
        expected_dk = np.pi / (n_pts * dr)
        np.testing.assert_allclose(k[1] - k[0], expected_dk)

    def test_roundtrip_recovers_signal(self):
        """sft(sft(g)) = g * pi/2 (self-inverse up to known scaling).

        The DST-I is self-inverse with factor (N+1)/2 for N-1 interior
        points, and dr*dk = pi/n_pts, giving an overall round-trip
        scaling of pi/2.
        """
        n_pts = 64
        dr = 0.1
        r = np.arange(n_pts) * dr
        g = np.zeros((n_pts, 1, 1))
        g[1:, 0, 0] = np.exp(-(r[1:] ** 2) / 2.0) * r[1:]

        k, f = sft(r, g)
        r2, g2 = sft(k, f)

        np.testing.assert_allclose(r2, r, atol=1e-14)
        np.testing.assert_allclose(g2[1:], g[1:] * np.pi / 2, rtol=1e-7)

    def test_roundtrip_multispecies(self):
        """SFT round-trip works independently for each species pair."""
        n_pts = 32
        dr = 0.1
        r = np.arange(n_pts) * dr
        rng = np.random.default_rng(123)
        g = np.zeros((n_pts, 2, 2))
        for i in range(2):
            for j in range(2):
                g[1:, i, j] = rng.standard_normal(n_pts - 1) * 0.1

        k, f = sft(r, g)
        _, g2 = sft(k, f)

        np.testing.assert_allclose(g2[1:], g[1:] * np.pi / 2, rtol=1e-9)
