"""Configuration loading and validation via Pydantic + TOML."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class SystemConfig(BaseModel):
    """Physical system parameters."""

    temperature: float
    density: list[float]
    n_species: Optional[int] = None

    @model_validator(mode="after")
    def _infer_n_species(self) -> "SystemConfig":
        n = len(self.density)
        if self.n_species is None:
            self.n_species = n
        elif self.n_species != n:
            raise ValueError(
                f"n_species={self.n_species} does not match len(density)={n}"
            )
        return self


class GridConfig(BaseModel):
    """Spatial / reciprocal-space grid."""

    dr: float
    r_max: float

    @field_validator("dr", "r_max")
    @classmethod
    def _positive(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v


class PotentialConfig(BaseModel):
    """Pair-potential specification."""

    expression: str
    epsilon: list[float]
    sigma: list[float]


class SolverConfig(BaseModel):
    """OZ solver settings."""

    closure: str = "PY"
    closure_params: dict[str, float] = {}
    tolerance: float = 1e-8


class RestartConfig(BaseModel):
    """Restart / warm-start options."""

    enabled: bool = False
    file: str = "gamma.dat"


class OutputConfig(BaseModel):
    """Output control."""

    directory: str = "./results"


class Config(BaseModel):
    """Root configuration model."""

    system: SystemConfig
    grid: GridConfig
    potential: PotentialConfig
    solver: SolverConfig = SolverConfig()
    restart: RestartConfig = RestartConfig()
    output: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def _validate_potential_dimensions(self) -> "Config":
        """Ensure epsilon/sigma arrays match n_species^2."""
        n = self.system.n_species
        if n is None:
            raise ValueError("n_species must be set (provide at least one density)")
        expected = n * n
        if len(self.potential.epsilon) != expected:
            raise ValueError(
                f"potential.epsilon has {len(self.potential.epsilon)} entries, "
                f"expected {expected} (n_species^2)"
            )
        if len(self.potential.sigma) != expected:
            raise ValueError(
                f"potential.sigma has {len(self.potential.sigma)} entries, "
                f"expected {expected} (n_species^2)"
            )
        return self

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        """Load and validate a TOML configuration file.

        Parameters
        ----------
        path
            Filesystem path to a ``.toml`` file.
        """
        with open(Path(path), "rb") as f:
            raw = tomllib.load(f)
        return cls(**raw)


def load_config(path: Path) -> Config:
    """Read a TOML file and return a validated Config."""
    return Config.from_toml(path)
