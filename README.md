# liquid-ie

Ornstein-Zernike integral equation solver and mode-coupling theory (MCT) for
multi-component liquids.

## Installation

```bash
uv sync --extra dev --extra plot
```

## Quick start

```bash
# Solve the OZ equation for a hard-sphere system
liquidie solve --config examples/hard_sphere_1species.toml --output-dir results/

# Run mode-coupling theory on the solver output
liquidie mct --config examples/hard_sphere_1species.toml --input-dir results/
```

## Configuration

All parameters are specified in a TOML config file. See the `examples/` directory
for ready-to-use configurations.

### Closures

Closures are specified as SymPy expressions over the variables `r`, `gamma_r`,
`phi`, and `inv_t`. Built-in shortcuts are recognised automatically:

| Shortcut | Expression |
|----------|-----------|
| `PY`     | `(r + gamma_r) * (exp(-inv_t * phi) - 1)` |
| `HNC`    | `r * exp(-inv_t * phi + gamma_r / r) - gamma_r - r` |
| `MS`     | `r * exp(-inv_t * phi + sqrt(1 + 2*gamma_r/r) - 1) - gamma_r - r` |
| `BPGG`   | `r * exp(-inv_t * phi + (1 + s*gamma_r/r)**(1/s) - 1) - gamma_r - r` |

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

| Shortcut        | Expression |
|-----------------|-----------|
| `hard_sphere`   | `Piecewise((1e+30, r < sigma), (0, True))` |
| `lennard_jones` | `4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)` |

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
| `rdf{i}{j}.dat` | *r*, g(*r*) | Radial distribution function |
| `c{i}{j}.dat` | *k*, c(*k*) | Direct correlation function |
| `h{i}{j}.dat` | *k*, h(*k*) | Total correlation function |
| `s{i}{j}.dat` | *k*, S(*k*) | Structure factor |

The `mct` command adds:

| File | Columns | Description |
|------|---------|-------------|
| `f{i}{j}.dat` | *k*, f(*k*) | MCT non-ergodicity parameter |

`gamma.dat` is also used as the warm-start file when `restart.enabled = true`.

## Physics background

This code solves the Ornstein-Zernike (OZ) equation for pair correlation
functions in multi-component liquid systems, using Newton-Krylov iteration.
The OZ equation relates the total correlation function h(r) to the direct
correlation function c(r) via an integral convolution, which is closed by
an approximate closure relation (e.g. Percus-Yevick or Hypernetted Chain).

The mode-coupling theory module computes the MCT memory kernel following
Phys. Rev. E 55, 657 (1997), enabling the study of glass transition dynamics.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src/ tests/
```
