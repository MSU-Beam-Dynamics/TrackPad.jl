```@meta
CurrentModule = TrackPad
```

# TrackPad.jl

**TrackPad.jl** is a general Julia engine for fast local accelerator models. It
provides single- and multi-particle tracking, linear optics, time-dependent
elements, PALS/MAD-X interchange, and optional GPU, Enzyme, and PolySeries
extensions.

Start with [Getting Started](@ref user_guide). Before comparing results with
another code, read the normative [Physics and Data Conventions](@ref conventions).
AI agents and integration tools should also read the [Agent Guide](@ref agent_guide).

## Features

- 6D symplectic tracking with exact Hamiltonian by default
- Canonical elements including bends, multipoles, RF cavities, solenoids,
  beam-beam, space-charge, wake, and wiggler models
- **GPU-accelerated tracking** via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl):
  multi-particle batch tracking and parameter sweeps on Metal or CUDA
- Batched Jacobians and second-order derivatives through Enzyme
- Optional PolySeries TPSA transfer maps
- Linear optics: Twiss parameters, tunes, chromaticity, closed orbit
- Time-dependent (turn-by-turn or real-time) element parameters
- PALS branch compilation and a documented subset of MAD-X

## Installation

TrackPad is currently used as a local/development package:

```julia
using Pkg
Pkg.develop(PackageSpec(path = "/path/to/TrackPad"))
```

Or, if you are working inside the repository:

```julia
using Pkg
Pkg.activate(".")
```

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
    "agent-guide.md",
    "api.md",
]
Depth = 2
```
