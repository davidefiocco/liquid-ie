"""Tests for the library-first public API surface."""

from pathlib import Path

import numpy as np
import pytest

import liquidie
from liquidie import (
    Config,
    KNOWN_CLOSURES,
    KNOWN_POTENTIALS,
    SolverResult,
    list_closures,
    list_potentials,
    load_config,
    run_mct,
    solve,
    write_results,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_all_symbols_importable(self):
        for name in liquidie.__all__:
            assert hasattr(liquidie, name), f"{name} not importable from liquidie"

    def test_list_potentials_returns_known_keys(self):
        result = list_potentials()
        assert set(result) == set(KNOWN_POTENTIALS)

    def test_list_closures_returns_known_keys(self):
        result = list_closures()
        assert set(result) == set(KNOWN_CLOSURES)

    def test_list_potentials_auto_updates(self):
        """Adding to KNOWN_POTENTIALS is immediately reflected."""
        sentinel = "test_sentinel_potential"
        KNOWN_POTENTIALS[sentinel] = "0"
        try:
            assert sentinel in list_potentials()
        finally:
            del KNOWN_POTENTIALS[sentinel]

    def test_list_closures_auto_updates(self):
        """Adding to KNOWN_CLOSURES is immediately reflected."""
        sentinel = "test_sentinel_closure"
        KNOWN_CLOSURES[sentinel] = "0"
        try:
            assert sentinel in list_closures()
        finally:
            del KNOWN_CLOSURES[sentinel]


# ---------------------------------------------------------------------------
# Config.from_toml
# ---------------------------------------------------------------------------


class TestFromToml:
    def test_from_toml_1species(self):
        cfg = Config.from_toml(EXAMPLES_DIR / "hard_sphere_1species.toml")
        assert cfg.system.n_species == 1
        assert isinstance(cfg, Config)

    def test_from_toml_binary(self):
        cfg = Config.from_toml(EXAMPLES_DIR / "hard_sphere_binary.toml")
        assert cfg.system.n_species == 2
        assert len(cfg.potential.sigma) == 4

    def test_from_toml_matches_load_config(self):
        path = EXAMPLES_DIR / "hard_sphere_1species.toml"
        cfg_a = Config.from_toml(path)
        cfg_b = load_config(path)
        assert cfg_a == cfg_b

    def test_from_toml_accepts_string(self):
        cfg = Config.from_toml(str(EXAMPLES_DIR / "hard_sphere_1species.toml"))
        assert cfg.system.n_species == 1


# ---------------------------------------------------------------------------
# SolverResult.squeeze
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def single_result():
    config = Config(
        system={"temperature": 1.0, "density": [0.3]},
        grid={"dr": 0.05, "r_max": 10.0},
        potential={
            "expression": "hard_sphere",
            "epsilon": [1.0],
            "sigma": [1.0],
        },
        solver={"closure": "PY", "tolerance": 1e-8},
    )
    return solve(config)


@pytest.fixture(scope="module")
def binary_result():
    config = Config(
        system={"temperature": 1.0, "density": [0.3, 0.3]},
        grid={"dr": 0.05, "r_max": 10.0},
        potential={
            "expression": "hard_sphere",
            "epsilon": [1.0, 1.0, 1.0, 1.0],
            "sigma": [1.0, 1.0, 1.0, 1.0],
        },
        solver={"closure": "PY", "tolerance": 1e-8},
    )
    return solve(config)


class TestSqueeze:
    def test_squeeze_produces_1d(self, single_result):
        sq = single_result.squeeze()
        assert sq.rdf.ndim == 1
        assert sq.s_k.ndim == 1
        assert sq.c_r.ndim == 1
        assert sq.c_k.ndim == 1
        assert sq.gamma_r.ndim == 1
        assert sq.gamma_k.ndim == 1
        assert sq.h_k.ndim == 1

    def test_squeeze_values_match(self, single_result):
        sq = single_result.squeeze()
        np.testing.assert_array_equal(sq.rdf, single_result.rdf[:, 0, 0])
        np.testing.assert_array_equal(sq.s_k, single_result.s_k[:, 0, 0])
        np.testing.assert_array_equal(sq.c_k, single_result.c_k[:, 0, 0])

    def test_squeeze_preserves_grids(self, single_result):
        sq = single_result.squeeze()
        np.testing.assert_array_equal(sq.r, single_result.r)
        np.testing.assert_array_equal(sq.k, single_result.k)

    def test_squeeze_preserves_metadata(self, single_result):
        sq = single_result.squeeze()
        np.testing.assert_array_equal(sq.density, single_result.density)
        assert sq.temperature == single_result.temperature

    def test_squeeze_multi_species_raises(self, binary_result):
        with pytest.raises(ValueError, match="n_species == 1"):
            binary_result.squeeze()


# ---------------------------------------------------------------------------
# SolverResult metadata (density / temperature)
# ---------------------------------------------------------------------------


class TestSolverResultMetadata:
    def test_solve_populates_density(self, single_result):
        assert single_result.density is not None
        np.testing.assert_allclose(single_result.density, [0.3])

    def test_solve_populates_temperature(self, single_result):
        assert single_result.temperature == 1.0

    def test_binary_density(self, binary_result):
        np.testing.assert_allclose(binary_result.density, [0.3, 0.3])


# ---------------------------------------------------------------------------
# SolverResult.from_directory recomputes gamma_k and c_r
# ---------------------------------------------------------------------------


class TestFromDirectory:
    def test_c_r_reconstructed(self, single_result, tmp_path):
        write_results(single_result, tmp_path)
        loaded = SolverResult.from_directory(tmp_path, 1)
        np.testing.assert_allclose(
            loaded.c_r[:, 0, 0],
            single_result.c_r[:, 0, 0],
            atol=1e-10,
        )

    def test_gamma_k_reconstructed(self, single_result, tmp_path):
        write_results(single_result, tmp_path)
        loaded = SolverResult.from_directory(tmp_path, 1)
        np.testing.assert_allclose(
            loaded.gamma_k[:, 0, 0],
            single_result.gamma_k[:, 0, 0],
            atol=1e-10,
        )

    def test_binary_c_r_reconstructed(self, binary_result, tmp_path):
        write_results(binary_result, tmp_path)
        loaded = SolverResult.from_directory(tmp_path, 2)
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(
                    loaded.c_r[:, i, j],
                    binary_result.c_r[:, i, j],
                    atol=1e-10,
                    err_msg=f"c_r[{i},{j}] mismatch",
                )


# ---------------------------------------------------------------------------
# run_mct with SolverResult-only call
# ---------------------------------------------------------------------------


class TestRunMctFromResult:
    def test_mct_from_result_only(self, single_result):
        """run_mct(result) works when result carries metadata."""
        f = run_mct(single_result, method="picard", n_iterations=3)
        assert f.shape == (len(single_result.k), 1, 1)

    def test_mct_with_config_override(self, single_result):
        """run_mct(result, config=...) uses config for density/temperature."""
        config = Config(
            system={"temperature": 1.0, "density": [0.3]},
            grid={"dr": 0.05, "r_max": 10.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        f = run_mct(single_result, config=config, method="picard", n_iterations=3)
        assert f.shape == (len(single_result.k), 1, 1)

    def test_mct_result_without_metadata_raises(self):
        """run_mct(result) raises if density/temperature are missing."""
        bare = SolverResult(
            r=np.zeros(10),
            k=np.zeros(10),
            gamma_r=np.zeros((10, 1, 1)),
            gamma_k=np.zeros((10, 1, 1)),
            c_r=np.zeros((10, 1, 1)),
            c_k=np.zeros((10, 1, 1)),
            rdf=np.zeros((10, 1, 1)),
            h_k=np.zeros((10, 1, 1)),
            s_k=np.zeros((10, 1, 1)),
            n_species=1,
        )
        with pytest.raises(ValueError, match="density and temperature"):
            run_mct(bare, method="picard", n_iterations=1)

    def test_config_override_agrees_with_metadata(self, single_result):
        """config= override gives same F(q) as result metadata."""
        config = Config(
            system={"temperature": 1.0, "density": [0.3]},
            grid={"dr": 0.05, "r_max": 10.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        f_config = run_mct(
            single_result, config=config, method="picard", n_iterations=5
        )
        f_meta = run_mct(single_result, method="picard", n_iterations=5)
        np.testing.assert_array_equal(f_config, f_meta)
