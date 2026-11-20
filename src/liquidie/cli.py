"""Typer CLI for LiquidIE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="liquidie",
    help="Ornstein-Zernike solver and mode-coupling theory for multi-component liquids.",
)


@app.command()
def solve(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to the TOML config file."),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            "-o",
            help="Override output directory from config.",
        ),
    ] = None,
    maxiter: Annotated[
        Optional[int],
        typer.Option(
            "--maxiter",
            help="Override maximum Newton-Krylov iterations from config.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable detailed progress logging."),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict", help="Raise on non-finite values instead of replacing with 0."
        ),
    ] = False,
) -> None:
    """Solve the Ornstein-Zernike equation."""
    from liquidie.config import load_config
    from liquidie.solver import solve as run_solve
    from liquidie.solver import write_results

    if verbose:
        logging.basicConfig(format="%(name)s: %(message)s")
        logging.getLogger("liquidie").setLevel(logging.INFO)

    cfg = load_config(config)
    if maxiter is not None:
        if maxiter <= 0:
            raise typer.BadParameter(f"--maxiter must be positive, got {maxiter}")
        cfg.solver.maxiter = maxiter
    out = Path(output_dir or cfg.output.directory)

    typer.echo(f"Loaded config from {config}")
    typer.echo(
        f"  {cfg.system.n_species} species, "
        f"T={cfg.system.temperature}, "
        f"closure={cfg.solver.closure!r}"
    )
    typer.echo(f"  grid: dr={cfg.grid.dr}, r_max={cfg.grid.r_max}")

    result = run_solve(cfg, strict=strict)

    write_results(result, out)
    typer.echo(f"Results written to {out}/")


@app.command()
def mct(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to the TOML config file."),
    ],
    input_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--input-dir",
            "-i",
            help="Directory with OZ solver output files. Defaults to output.directory.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for MCT output files. Defaults to input directory.",
        ),
    ] = None,
    n_iterations: Annotated[
        int,
        typer.Option(
            "--iterations",
            "-n",
            help="Maximum MCT Picard iterations; stops early once converged.",
        ),
    ] = 1000,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            "-m",
            help="MCT solver method: 'picard' or 'newton_krylov'.",
        ),
    ] = "picard",
    tolerance: Annotated[
        float,
        typer.Option(
            "--tolerance",
            "-t",
            help="Convergence tolerance for the MCT solver.",
        ),
    ] = 1e-8,
    k_max: Annotated[
        Optional[float],
        typer.Option(
            "--k-max",
            help="Upper wavevector cutoff for the MCT kernel (slices k <= k_max).",
        ),
    ] = None,
    masses: Annotated[
        Optional[str],
        typer.Option(
            "--masses",
            help="Comma-separated per-species masses (e.g. '1.0,2.0'). Defaults to all ones.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable detailed progress logging."),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict", help="Raise on non-finite values instead of replacing with 0."
        ),
    ] = False,
) -> None:
    """Run mode-coupling theory on OZ solver output."""
    import numpy as np

    from liquidie.config import load_config
    from liquidie.mct import run_mct, write_mct_results
    from liquidie.solver import SolverResult

    if verbose:
        logging.basicConfig(format="%(name)s: %(message)s")
        logging.getLogger("liquidie").setLevel(logging.INFO)

    cfg = load_config(config)
    if cfg.system.n_species is None:
        raise ValueError("n_species must be set (provide at least one density)")
    n_species = cfg.system.n_species
    in_dir = Path(input_dir or cfg.output.directory)
    out = Path(output_dir or in_dir)

    masses_arr = None
    if masses is not None:
        try:
            parsed = [float(val.strip()) for val in masses.split(",") if val.strip()]
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid --masses value: {masses!r}") from exc
        if len(parsed) != n_species or any(m <= 0 for m in parsed):
            raise typer.BadParameter(
                f"--masses must provide {n_species} positive number(s), got {masses!r}"
            )
        masses_arr = np.asarray(parsed, dtype=float)

    typer.echo(f"Loading solver results from {in_dir}/")

    result = SolverResult.from_directory(in_dir, n_species)

    f = run_mct(
        result,
        config=cfg,
        masses=masses_arr,
        method=method,  # type: ignore[arg-type]
        n_iterations=n_iterations,
        tolerance=tolerance,
        k_max=k_max,
        strict=strict,
    )
    write_mct_results(f, result.k[: len(f)], n_species, out)
    typer.echo(f"MCT results written to {out}/")


if __name__ == "__main__":
    app()
