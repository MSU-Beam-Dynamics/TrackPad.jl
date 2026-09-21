```@meta
CurrentModule = TrackPad
```

# TrackPad.jl

**TrackPad.jl** is a Julia library for symplectic particle tracking and linear
optics of accelerator lattices. It supports symplectic tracking, CPU and GPU batches for particle tracking, automatic differentiation and truncated-power-series tracking.

Start with [Getting Started](@ref user_guide). To understand the results and compare with
another code, read the [Physics and Data Conventions](@ref conventions).
AI agents and integration tools should also read the [Agent Guide](@ref agent_guide).

## Features

- Symplectic 6-D tracking in canonical coordinates ``(x,p_x,y,p_y,z,\delta_E)``
  with the exact drift Hamiltonian; 
- Supports multipoles magnets, RF and crab cavities, solenoids, wigglers, correctors,
  beam–beam effect, longitudinal wakes and space charge effects
- Linear optics of coupled and uncoupled lattices, in 4-D and 6-D (in progress)
- Multi-particle tracking on CPU threads or on NVIDIA/Apple GPUs from one
  packed lattice; lazily generated parameter sweeps
- Support time-dependent parameters of the element for ramping study
- Support the Particle Accelerator Lattice Standard (PALS) and a documented MAD-X subset for lattice interchange

## Installation

TrackPad is not yet in the General registry; install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/MSU-Beam-Dynamics/TrackPad.jl")
```

TrackPad depends on [PolySeries.jl](https://github.com/MSU-Beam-Dynamics/PolySeries.jl)
(the truncated-power-series backend of the default optics method), which is
also unregistered for now: add it first with
`Pkg.add(url = "https://github.com/MSU-Beam-Dynamics/PolySeries.jl")`.

Optional capabilities are activated by loading their companion package
alongside TrackPad — nothing else to configure:

| capability | load | notes |
|---|---|---|
| batched derivatives | `using Enzyme` | registered |
| NVIDIA GPU | `using CUDA` | `Float32`/`Float64` |
| Apple GPU | `using Metal` | `Float32` only |
| lattice plots | `using CairoMakie` (or another Makie backend) | `plot_lattice!` |

Julia 1.12 or newer is required.

## Quick Start

```julia
using TrackPad
using StaticArrays

# 1 GeV electron beam
beam = Beam(1.0e9)

# Simple FODO lattice
d  = Drift(1.0)
qf = Quadrupole(0.3,  0.7)
qd = Quadrupole(0.3, -0.7)
lat = Lattice(AbstractElement[d, qf, d, qd]; periodic=true)

# Track a single particle from the origin
r0 = SVector(1e-3, 0.0, 0.0, 0.0, 0.0, 0.0)
r1 = linepass(lat, r0, beam)

# Linear optics
tunes = gettune(lat, beam)
println("Qx = ", tunes[1], "  Qy = ", tunes[2])
```

## Contents

```@contents
Pages = [
    "guide.md",
    "conventions.md",
    "elements.md",
    "io.md",
    "gpu.md",
    "performance.md",
    "examples.md",
    "agent-guide.md",
    "api.md",
]
Depth = 2
```
