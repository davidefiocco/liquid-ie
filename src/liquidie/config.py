"""Configuration loading and validation via Pydantic + TOML."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SystemConfig(_StrictModel):
    """Physical system parameters."""

    temperature: float
    density: list[float]
    n_species: Optional[int] = None

    @field_validator("temperature")
    @classmethod
    def _positive_temperature(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"temperature must be positive, got {v}")
        return v

    @field_validator("density")
    @classmethod
    def _positive_density(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("density must contain at least one species")
        for i, rho_i in enumerate(v):
            if rho_i <= 0:
                raise ValueError(f"density[{i}] must be positive, got {rho_i}")
        return v

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


class GridConfig(_StrictModel):
    """Spatial / reciprocal-space grid."""

    dr: float
    r_max: float

    @field_validator("dr", "r_max")
    @classmethod
    def _positive(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _r_max_exceeds_dr(self) -> "GridConfig":
        if self.r_max <= self.dr:
            raise ValueError(
                f"r_max ({self.r_max}) must be greater than dr ({self.dr})"
            )
        return self


class PotentialConfig(_StrictModel):
    """Pair-potential specification."""

    expression: str
    epsilon: list[float]
    sigma: list[float]
    potential_params: dict[str, float] = Field(default_factory=dict)


class SolverConfig(_StrictModel):
    """OZ solver settings."""

    closure: str = "PY"
    closure_params: dict[str, float] = Field(default_factory=dict)
    tolerance: float = 1e-8
    maxiter: int | None = 200

    @field_validator("tolerance")
    @classmethod
    def _positive_tolerance(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"tolerance must be positive, got {v}")
        return v

    @field_validator("maxiter")
    @classmethod
    def _positive_maxiter(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError(f"maxiter must be positive, got {v}")
        return v


class RestartConfig(_StrictModel):
    """Restart / warm-start options."""

    enabled: bool = False
    file: str = "gamma.dat"


class OutputConfig(_StrictModel):
    """Output control."""

    directory: str = "./results"


class Config(_StrictModel):
    """Root configuration model."""

    system: SystemConfig
    grid: GridConfig
    potential: PotentialConfig
    solver: SolverConfig = Field(default_factory=SolverConfig)
    restart: RestartConfig = Field(default_factory=RestartConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _validate_potential_dimensions(self) -> "Config":
        """Ensure epsilon/sigma arrays match n_species^2 and are symmetric."""
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
        for name, values in (
            ("epsilon", self.potential.epsilon),
            ("sigma", self.potential.sigma),
        ):
            for i in range(n):
                for j in range(i + 1, n):
                    if values[i * n + j] != values[j * n + i]:
                        raise ValueError(
                            f"potential.{name} must be symmetric: "
                            f"entry ({i},{j})={values[i * n + j]} != "
                            f"({j},{i})={values[j * n + i]}"
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
