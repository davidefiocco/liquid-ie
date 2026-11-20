"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest

from liquidie.config import Config, load_config


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


class TestLoadConfig:
    def test_load_1species(self):
        cfg = load_config(EXAMPLES_DIR / "hard_sphere_1species.toml")
        assert cfg.system.n_species == 1
        assert cfg.system.temperature == 1.0
        assert len(cfg.potential.sigma) == 1

    def test_load_binary(self):
        cfg = load_config(EXAMPLES_DIR / "hard_sphere_binary.toml")
        assert cfg.system.n_species == 2
        assert len(cfg.potential.epsilon) == 4
        assert cfg.potential.sigma == [1.0, 0.8, 0.8, 0.6]

    def test_load_lennard_jones(self):
        cfg = load_config(EXAMPLES_DIR / "lennard_jones_hnc.toml")
        assert cfg.system.n_species == 1
        assert cfg.solver.closure == "HNC"
        assert cfg.system.temperature == 2.74

    def test_n_species_inferred_from_density(self):
        cfg = Config(
            system={"temperature": 1.0, "density": [0.1, 0.2, 0.3]},
            grid={"dr": 0.01, "r_max": 10.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0] * 9,
                "sigma": [1.0] * 9,
            },
        )
        assert cfg.system.n_species == 3

    def test_n_species_mismatch_raises(self):
        with pytest.raises(ValueError, match="n_species=2.*len\\(density\\)=1"):
            Config(
                system={"temperature": 1.0, "density": [0.1], "n_species": 2},
                grid={"dr": 0.01, "r_max": 10.0},
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0],
                    "sigma": [1.0],
                },
            )

    def test_negative_dr_raises(self):
        with pytest.raises(ValueError, match="dr must be positive"):
            Config(
                system={"temperature": 1.0, "density": [0.1]},
                grid={"dr": -0.01, "r_max": 10.0},
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0],
                    "sigma": [1.0],
                },
            )

    def test_wrong_potential_dimension_raises(self):
        with pytest.raises(ValueError, match="potential.epsilon has 1 entries"):
            Config(
                system={"temperature": 1.0, "density": [0.1, 0.2]},
                grid={"dr": 0.01, "r_max": 10.0},
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0],
                    "sigma": [1.0, 1.0, 1.0, 1.0],
                },
            )

    def test_asymmetric_sigma_raises(self):
        with pytest.raises(ValueError, match="potential.sigma must be symmetric"):
            Config(
                system={"temperature": 1.0, "density": [0.1, 0.2]},
                grid={"dr": 0.01, "r_max": 10.0},
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0, 1.0, 1.0, 1.0],
                    "sigma": [1.0, 0.8, 0.9, 0.6],
                },
            )

    def test_asymmetric_epsilon_raises(self):
        with pytest.raises(ValueError, match="potential.epsilon must be symmetric"):
            Config(
                system={"temperature": 1.0, "density": [0.1, 0.2]},
                grid={"dr": 0.01, "r_max": 10.0},
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0, 0.5, 1.0, 1.0],
                    "sigma": [1.0, 0.8, 0.8, 0.6],
                },
            )

    def test_unknown_field_raises(self):
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            Config(
                system={"temperature": 1.0, "density": [0.1]},
                grid={"dr": 0.01, "r_max": 10.0},
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0],
                    "sigma": [1.0],
                },
                solver={"closure": "PY", "tolerence": 1e-6},
            )

    def test_unknown_section_raises(self):
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            Config(
                system={"temperature": 1.0, "density": [0.1]},
                grid={"dr": 0.01, "r_max": 10.0},
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0],
                    "sigma": [1.0],
                },
                bogus={"x": 1},
            )

    @pytest.mark.parametrize("temperature", [0.0, -1.0])
    def test_non_positive_temperature_raises(self, temperature):
        with pytest.raises(ValueError, match="temperature must be positive"):
            Config(
                system={"temperature": temperature, "density": [0.1]},
                grid={"dr": 0.01, "r_max": 10.0},
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0],
                    "sigma": [1.0],
                },
            )

    @pytest.mark.parametrize("density", [[], [0.0], [0.1, -0.2]])
    def test_invalid_density_raises(self, density):
        with pytest.raises(ValueError):
            Config(
                system={"temperature": 1.0, "density": density},
                grid={"dr": 0.01, "r_max": 10.0},
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0] * max(1, len(density) ** 2),
                    "sigma": [1.0] * max(1, len(density) ** 2),
                },
            )

    def test_r_max_not_greater_than_dr_raises(self):
        with pytest.raises(ValueError, match="r_max .* must be greater than dr"):
            Config(
                system={"temperature": 1.0, "density": [0.1]},
                grid={"dr": 1.0, "r_max": 1.0},
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0],
                    "sigma": [1.0],
                },
            )

    def test_non_positive_tolerance_and_maxiter_raise(self):
        with pytest.raises(ValueError, match="tolerance must be positive"):
            Config(
                system={"temperature": 1.0, "density": [0.1]},
                grid={"dr": 0.01, "r_max": 10.0},
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0],
                    "sigma": [1.0],
                },
                solver={"tolerance": 0.0},
            )
        with pytest.raises(ValueError, match="maxiter must be positive"):
            Config(
                system={"temperature": 1.0, "density": [0.1]},
                grid={"dr": 0.01, "r_max": 10.0},
                potential={
                    "expression": "hard_sphere",
                    "epsilon": [1.0],
                    "sigma": [1.0],
                },
                solver={"maxiter": 0},
            )
