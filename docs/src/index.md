# TrackPad.jl

**TrackPad.jl** is a Julia package for single- and multi-particle tracking in
particle accelerators.  It is designed for high performance and automatic
differentiation (AD): all particle coordinates are stored as
[`StaticArrays.SVector{6,T}`](https://github.com/JuliaArrays/StaticArrays.jl),
making the hot loop allocation-free and fully compatible with
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl),
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl), and
[PolySeries.jl](https://github.com/PolySeries.jl/PolySeries.jl) (TPSA maps).

## Features

- 6D symplectic tracking with exact Hamiltonian by default
- 16+ element types: dipoles, quadrupoles, sextupoles, RF cavities, solenoids,
  beam–beam kicks, space-charge elements, wigglers, and more
- **GPU-accelerated tracking** via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl):
  multi-particle batch tracking and parameter sweeps on Metal, CUDA, or ROCm
- Automatic-differentiation–friendly (Enzyme, ForwardDiff, PolySeries TPSA)
- Linear optics: Twiss parameters, tunes, chromaticity, closed orbit
- Time-dependent (turn-by-turn or real-time) element parameters

## Installation

TrackPad.jl is a local/development package.  Add it with Pkg:

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
d  = Drift(2.5)
qf = Quadrupole(0.5;  k1 =  1.2)
qd = Quadrupole(0.5;  k1 = -1.2)
lat = Lattice([d, qf, d, qd])

# Track a single particle from the origin
r0 = SVector(1e-3, 0.0, 0.0, 0.0, 0.0, 0.0)
r1 = linepass(lat, r0, beam)

# Linear optics
tunes = gettune(lat, beam)
println("Qx = ", tunes[1], "  Qy = ", tunes[2])
```

## Contents

```@contents
Pages = ["guide.md", "elements.md", "api.md"]
Depth = 2
```
