# liquid-ie

Ornstein-Zernike integral equation solver and mode-coupling theory (MCT) for
multi-component liquids.

## Installation

```bash
uv sync --extra dev --extra plot
```

## Quick start

### As a Python library

```python
from liquidie import Config, solve, run_mct, list_potentials, list_closures

# Discover available presets
list_potentials()  # ['hard_sphere', 'lennard_jones', 'inverse_power_law', 'gaussian_core']
list_closures()    # ['PY', 'HNC', 'MS', 'BPGG']

# Load config from TOML or build from dicts
config = Config.from_toml("examples/hard_sphere_1species.toml")

# Solve the OZ equation
result = solve(config)

# For single-component systems, squeeze to 1-D arrays
sq = result.squeeze()
r, g = sq.r, sq.rdf    # 1-D numpy arrays

# Run MCT directly from the solver result (no config needed)
# k_max decouples the O(N^3) MCT kernel cost from a fine real-space OZ grid
F = run_mct(result, method="picard", k_max=40.0)
f = F[:, 0, 0] / result.s_k[:len(F), 0, 0]   # normalised non-ergodicity parameter
```

### From the command line

```bash
# Solve the OZ equation for a hard-sphere system
liquidie solve --config examples/hard_sphere_1species.toml --output-dir results/

# Run mode-coupling theory on the solver output (writes f00.dat into results/)
liquidie mct --config examples/hard_sphere_1species.toml --input-dir results/ --k-max 40
```

## Configuration

All parameters are specified in a TOML config file. See the `examples/` directory
for ready-to-use configurations.

### Closures

Closures are specified as SymPy expressions over the variables `r`, `gamma_r`,
`phi`, and `inv_t`. Built-in shortcuts are recognised automatically:

Writing *g* = `gamma_r / r` for the indirect correlation function, each
closure is `c(r) = exp(-beta*phi + g + B(g)) - 1 - g` for the bridge
function *B* below. Run `list_closures()` for the full set.

| Shortcut | Bridge function *B(g)* | Parameters |
|----------|------------------------|------------|
| `PY`     | `ln(1 + g) - g` | — |
| `HNC`    | `0` | — |
| `MS`     | `sqrt(1 + 2g) - 1 - g` | — |
| `BPGG`   | `(1 + s*g)**(1/s) - 1 - g` | `s` |

`PY`, `HNC`, and `MS` need no parameters; `BPGG` requires `s` via
`closure_params`.

`BPGG` is a one-parameter family (Ballone-Pastore-Galli-Gazzillo) that
interpolates between HNC (*s* = 1) and MS (*s* = 2). The parameter *s* is
set via `closure_params`:

```toml
[solver]
closure = "BPGG"
closure_params = { s = 1.5 }
```

You can also write any valid SymPy expression directly:

```toml
[solver]
closure = "(r + gamma_r) * (exp(-inv_t * phi) - 1)"
```

### Potentials

Potentials are SymPy expressions over `r`, `sigma`, and `epsilon`. Built-in
shortcuts:

| Shortcut            | Expression | Parameters |
|---------------------|-----------|------------|
| `hard_sphere`       | `Piecewise((1e+30, r < sigma), (0, True))` | — |
| `lennard_jones`     | `4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)` | — |
| `inverse_power_law` | `epsilon * (sigma / r)**n` | `n` |
| `gaussian_core`     | `epsilon * exp(-(r / sigma)**2)` | — |

Parameters are supplied through `potential_params`, mirroring
`closure_params`:

```toml
[potential]
expression = "inverse_power_law"
potential_params = { n = 5 }
```

Custom potentials work the same way. Use SymPy `Piecewise` for discontinuous
potentials:

```toml
[potential]
expression = "Piecewise((-epsilon, r < 1.5 * sigma), (0, True))"
```

More examples:

- **Yukawa**: `epsilon * exp(-r / sigma) / r`
- **Soft sphere**: `epsilon * (sigma / r)**12`
- **Square well**: `Piecewise((1e30, r < sigma), (-epsilon, r < 1.5 * sigma), (0, True))`

## Output files

The `solve` command writes two-column text files (space-separated, loadable with
`np.loadtxt`) to the output directory. For an *n*-component system, indices
*i*, *j* run over the upper triangle (0 ≤ *i* ≤ *j* < *n*):

| File | Columns | Description |
|------|---------|-------------|
| `gamma.dat` | *r*, γ₀₀, γ₀₁, … | Indirect correlation function (restart file) |
| `metadata.json` | JSON | `n_species`, `density`, `temperature` (reloaded by `SolverResult.from_directory`) |
| `rdf{i}{j}.dat` | *r*, g(*r*) | Radial distribution function |
| `c{i}{j}.dat` | *k*, c(*k*) | Direct correlation function |
| `h{i}{j}.dat` | *k*, h(*k*) | Total correlation function |
| `s{i}{j}.dat` | *k*, S(*k*) | Structure factor |

The `mct` command adds:

| File | Columns | Description |
|------|---------|-------------|
| `f{i}{j}.dat` | *k*, F(*k*) | MCT non-ergodicity parameter, **unnormalised** |

`run_mct` and `f{i}{j}.dat` give F(*k*) = S(*k*) f(*k*). Divide by the
structure factor to recover the f(*k*) that is conventionally plotted and
tabulated in the literature.

`gamma.dat` is also used as the warm-start file when `restart.enabled = true`.

## Physics background

This code solves the Ornstein-Zernike (OZ) equation for pair correlation
functions in multi-component liquid systems, using Newton-Krylov iteration.
The OZ equation relates the total correlation function h(r) to the direct
correlation function c(r) via an integral convolution, which is closed by
an approximate closure relation (e.g. Percus-Yevick or Hypernetted Chain).

**Units and conventions.** Boltzmann's constant is set to *k*_B = 1, so
`temperature` is in units of ε / *k*_B (with inverse temperature
`inv_t = 1 / temperature`), lengths in units of σ, and partial number
densities `density = [rho_0, ...]` in σ⁻³. The partial structure
factor uses the Nauroth–Kob / Götze–Voigtmann convention
S_ij(*k*) = *x*_i δ_ij + ρ *x*_i *x*_j h_ij(*k*) with
ρ = Σ_i ρ_i and *x*_i = ρ_i / ρ, and c_ij(*k*) is
the dimensionful 3-D Fourier transform of the OZ direct correlation function.

The mode-coupling theory module computes the long-time non-ergodicity
parameter F_ij(*q*) following Nauroth & Kob, Phys. Rev. E 55, 657 (1997).
Thermodynamic observables (virial/compressibility pressure, excess energy) and
time-dependent F(*q*, *t*) are not yet included.

## Numerical notes

**Grid resolution.** When the Boltzmann factor jumps exactly on a grid node
(a hard core with `sigma` a multiple of `dr`, or a square-well edge), the
transforms use the mean of the two one-sided closure values. This is the
trapezoidal rule for a step, so the structure is second order in `dr`. For
PY hard spheres at η = 0.4, S(0) is within 0.3% of the exact value at
`dr` = 0.02 and within 0.07% at `dr` = 0.01. At such a node the reported
c(*r*) and g(*r*) hold the outer limit, e.g. the contact value g(σ⁺). A jump
that falls between grid nodes stays first order and logs a warning, so choose
`dr` to divide every `sigma`.

**Spurious roots.** At high density or strong attraction, Newton-Krylov from
γ = 0 can converge to an unphysical root with negative S(*k*)
eigenvalues. `solve` warns when it detects one (and raises under
`strict=True`); the remedy is approaching the state point by density or
temperature ramping via `gamma_initial` (and checking `dr` / `r_max`).

**MCT cost and bias.** The MCT grid inherits Δ*k* = π/`r_max` from the OZ
grid, and the kernel scales as O(*n*³) per iteration. Passing `k_max` (or
`--k-max` on the CLI) truncates the MCT wavevector grid to *k* ≤ `k_max`
(e.g. `k_max=40` in units of σ⁻¹), decoupling MCT runtime from a
fine real-space `dr`. Near the transition the fixed-point iteration slows
critically and needs hundreds of iterations.

**MCT accuracy.** The kernel integrals use the trapezoidal rule. With the
exact PY S(*k*), η_c = 0.5155 at Δ*k* = 0.4 and 0.5157 at Δ*k* = 0.2,
against 0.5159 in the literature (Franosch et al. 1997). The full OZ → MCT
pipeline on the grid of `examples/hard_sphere_mct_glass.toml` gives
η_c ≈ 0.517. F(0) is the analytical *q* → 0 limit. The first few nonzero
*q* points carry a kernel error of order (Δ*k*/*q*)², because the vertex
cancels to O(*q*) there, so F(*q*) at *q* ≲ 3Δ*k* is resolution-limited.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src/ tests/
```
