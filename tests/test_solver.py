"""Integration test for the OZ solver."""

import numpy as np
import pytest

from liquidie.config import Config
from liquidie.solver import solve


@pytest.fixture
def hs_1species_config():
    """Hard-sphere 1-species config at low-moderate density."""
    return Config(
        system={"temperature": 1.0, "density": [0.2]},
        grid={"dr": 0.02, "r_max": 20.0},
        potential={
            "expression": "hard_sphere",
            "epsilon": [1.0],
            "sigma": [1.0],
        },
        solver={"closure": "PY", "tolerance": 1e-8},
    )


class TestSolver:
    def test_solve_returns_result(self, hs_1species_config):
        result = solve(hs_1species_config)
        assert result.r.shape[0] > 0
        assert result.k.shape[0] > 0
        assert result.rdf.shape == (len(result.r), 1, 1)
        assert result.s_k.shape == (len(result.k), 1, 1)

    def test_rdf_zero_inside_core(self, hs_1species_config):
        """For hard spheres, g(r) should be ~0 for r < sigma (skip r=0)."""
        result = solve(hs_1species_config)
        core_mask = (result.r > 0.1) & (result.r < 0.9)
        core_rdf = result.rdf[core_mask, 0, 0]
        np.testing.assert_array_less(
            np.abs(core_rdf),
            0.05,
            err_msg=f"RDF inside core should be < 0.05, got max={np.max(np.abs(core_rdf))}",
        )

    def test_rdf_peak_near_contact(self, hs_1species_config):
        """For hard spheres the first peak of g(r) is near r=sigma."""
        result = solve(hs_1species_config)
        rdf_1d = result.rdf[:, 0, 0]
        peak_idx = np.argmax(rdf_1d)
        peak_r = result.r[peak_idx]
        assert 0.9 < peak_r < 1.5, f"RDF peak at r={peak_r}, expected near sigma=1.0"

    def test_structure_factor_positive(self, hs_1species_config):
        """S(k) should be positive for a stable fluid."""
        result = solve(hs_1species_config)
        s_1d = result.s_k[1:, 0, 0]  # skip k=0
        assert np.all(s_1d > -0.01), f"S(k) has negative values: min={s_1d.min()}"
