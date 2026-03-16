"""Tests for vectorised linear algebra operations."""

import numpy as np
import numpy.testing as npt

from liquidie.linalg import dotve, dotvbs, invv


class TestInvv:
    def test_1x1(self):
        a = np.array([[[2.0]], [[4.0]]])
        result = invv(a)
        npt.assert_allclose(result, np.array([[[0.5]], [[0.25]]]))

    def test_2x2(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((10, 2, 2))
        a += 3.0 * np.eye(2)  # ensure non-singular
        result = invv(a)
        for i in range(10):
            npt.assert_allclose(result[i], np.linalg.inv(a[i]), atol=1e-10)

    def test_3x3(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((5, 3, 3))
        a += 5.0 * np.eye(3)
        result = invv(a)
        for i in range(5):
            npt.assert_allclose(result[i], np.linalg.inv(a[i]), atol=1e-10)


class TestDotve:
    def test_identity(self):
        n_pts = 5
        n = 2
        a = np.tile(np.eye(n), (n_pts, 1, 1))
        rng = np.random.default_rng(99)
        b = rng.standard_normal((n_pts, n, n))
        result = dotve(a, b)
        npt.assert_allclose(result, b, atol=1e-12)

    def test_matches_matmul(self):
        rng = np.random.default_rng(7)
        a = rng.standard_normal((8, 3, 3))
        b = rng.standard_normal((8, 3, 3))
        result = dotve(a, b)
        for i in range(8):
            npt.assert_allclose(result[i], a[i] @ b[i], atol=1e-12)


class TestDotvbs:
    def test_diagonal_scaling(self):
        """Multiplying by a diagonal matrix scales columns."""
        b = np.ones((3, 2, 2))
        s = np.array([[2.0, 0.0], [0.0, 3.0]])
        result = dotvbs(b, s)
        for q in range(3):
            npt.assert_allclose(result[q], b[q] @ s, atol=1e-15)

    def test_matches_manual(self):
        """dotvbs(b, s) matches b[q] @ s for random inputs."""
        rng = np.random.default_rng(77)
        b = rng.standard_normal((10, 3, 3))
        s = rng.standard_normal((3, 3))
        result = dotvbs(b, s)
        for q in range(10):
            npt.assert_allclose(result[q], b[q] @ s, atol=1e-12)

    def test_identity(self):
        """Multiplying by identity leaves b unchanged."""
        rng = np.random.default_rng(55)
        b = rng.standard_normal((5, 2, 2))
        s = np.eye(2)
        result = dotvbs(b, s)
        npt.assert_allclose(result, b, atol=1e-15)
