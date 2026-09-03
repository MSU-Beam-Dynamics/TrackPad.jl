# TrackPad.jl

TrackPad is a Julia accelerator tracking library for canonical 6D particle
tracking and linear optics. It provides optional GPU batch tracking, Enzyme
derivatives, and TPSA maps through PolySeries while keeping the traditional
scalar CPU tracking.

## Requirements

- Julia: 1.10 or newer
- Optional: CUDA.jl, Metal.jl, Enzyme.jl, PolySeries.jl

## Features
- TPSA tracking through PolySeries.jl
- Automatic differentiation through Enzyme.jl
- GPU backends: Apple Metal (`Float32`) and NVIDIA CUDA (`Float32`/`Float64`), Note: Not all elements are supported on GPU yet, Unsupported GPU element settings are rejected when a `GPULattice` is built.
- Matched 4D/6D Gaussian macroparticle distributions with exact finite-sample moments
- File interchange: documented PALS and MAD-X subsets



## Installation

TrackPad is currently used as a development package:

```julia
using Pkg
Pkg.develop(path = "/path/to/TrackPad")
```

For work inside this repository:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

```julia
using StaticArrays, TrackPad

# Beam energy is kinetic energy in eV.
beam = Beam(3.0e9)

ring = Lattice(AbstractElement[
    Drift(1.0; name=:D1),
    Quadrupole(0.3, 0.7; name=:QF),
    Drift(1.0; name=:D2),
    Quadrupole(0.3, -0.7; name=:QD),
]; name=:FODO, periodic=true)

r0 = @SVector [1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]
r1 = linepass(ring, r0, beam)

qx, qy = gettune(ring, beam)
twiss = periodic_twiss(
    ring, beam; sample_integrator_steps=true, max_step=0.1,
)
orbit = find_closed_orbit_4d(ring, beam)
```

The phase-space coordinates are
`($x$, $p_x$, $y$, $p_y$, $z=s/\beta_0-ct$, $\delta_E=(E-E_0)/(P_0c)$)`.
TrackPad does not store the relative momentum deviation
`$\delta_P=(P-P_0)/P_0$`; only to first order is
`$\delta_E=\beta_0\delta_P$`. Read `docs/src/conventions.md` before comparing
results with another code.

## Choose an API

| Goal | API |
|------|-----|
| Track one particle through a line | `linepass` |
| Track one particle for many turns | `ringpass` |
| Track an `N x 6` CPU matrix with loss flags | `linepass!`, `ringpass!` |
| Generate a matched Gaussian ensemble | `matched_gaussian`, `gaussian_distribution` |
| Inspect covariance and emittances | `beam_covariance`, `projected_emittances`, `eigenemittances` |
| Track a packed CPU/GPU batch | `GPULattice`, `batch_linepass!`, `batch_ringpass!` |
| Scan lattice parameters lazily | `ParamSweepLattice`, `param_sweep_linepass!` |
| Propagate entrance Twiss through a line | `transport_twiss` |
| Compute periodic ring Twiss | `periodic_twiss` |
| Compute ring tunes/chromaticity | `gettune`, `getchrom` |
| Compute a line or ring map | `transfer_map` |
| Compute one-turn maps and closed orbits | `one_turn_map`, `find_closed_orbit_4d` |
| Read/write lattice files | `read_pals`, `write_pals`, `read_madx` |
| Compute batched derivatives | `batch_jacobian!`, `batch_hessian_vector_product!` |
| Compute a TPSA map | `tpsa_map` after `using PolySeries` |

## Optional Features

```julia
# Apple GPU
using TrackPad, Metal
gl = gpu_adapt(ring, beam, MetalBackend())

# NVIDIA GPU
using TrackPad, CUDA
gl = gpu_adapt(ring, beam, CUDABackend(); dtype=Float64)

# Batched derivatives
using TrackPad, Enzyme

# TPSA maps
using TrackPad, PolySeries
```

See `docs/src/gpu.md` for supported GPU elements and settings. Loading a weak
dependency activates its TrackPad extension automatically.

## Documentation

- `examples/README.md`: numbered, workflow-oriented notebook series
- `docs/src/guide.md`: human-oriented workflow
- `docs/src/conventions.md`: normative coordinates, units, and normalization
- `docs/src/elements.md`: element catalog and constructors
- `docs/src/io.md`: PALS and MAD-X lattice file I/O
- `docs/src/gpu.md`: GPU, parameter sweeps, and batched derivatives
- `docs/src/agent-guide.md`: compact contract for package-using AI agents
- `AGENTS.md`: repository architecture and change rules for coding agents
- `llms.txt`: short machine-readable package index

## Testing

Run the package test target, not `test/runtests.jl` directly, because the test
target activates JuTrack, PolySeries, and Enzyme test dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Build documentation with:

```bash
julia --project=docs docs/make.jl
```

CUDA hardware tests use the separate environment in `test/cuda`.

## Design Boundary

TrackPad consumes one ordered `Lattice` and one `Beam`. A lattice has only an
open-line (`periodic=false`) or closed-ring (`periodic=true`) boundary.
Consumer-specific adapters own machine topology, route selection, source
metadata, serialization, and result conversion.
