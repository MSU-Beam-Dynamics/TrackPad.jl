```@meta
CurrentModule = TrackPad
```

# Getting Started

This guide walks through the typical workflow: building a beam and lattice,
tracking particles, computing linear optics, and extracting TPSA transfer maps.

## Coordinate Convention

TrackPad uses the standard 6D phase-space vector

```math
\mathbf{r} = (x,\ p_x,\ y,\ p_y,\ z,\ \delta)
```

| Index | Symbol | Description |
|-------|--------|-------------|
| 1 | ``x`` | horizontal position [m] |
| 2 | ``p_x`` | normalised horizontal momentum ``p_x / p_0`` |
| 3 | ``y`` | vertical position [m] |
| 4 | ``p_y`` | normalised vertical momentum ``p_y / p_0`` |
| 5 | ``z`` | longitudinal position (``-c \Delta t``) [m] |
| 6 | ``\delta`` | fractional momentum deviation ``(p - p_0)/p_0`` |

Coordinates are stored as `StaticArrays.SVector{6, T}`.  The element type `T`
may be `Float64` for normal tracking or a `PolySeries.CTPS` type for TPSA maps.

## Beam

The [`Beam`](@ref) struct carries the reference momentum and relativistic
factors for all element pass functions.

```julia
using TrackPad

# 3 GeV electron beam (default particle)
beam = Beam(3.0e9)

# 6.5 TeV proton beam
p_beam = Beam(6.5e12; mass = M_PROTON, charge = 1.0)
```

[`beti(beam)`](@ref) returns ``1/\beta`` and is the single beam parameter
passed to element kernels.

## Lattice Construction

Build a [`Lattice`](@ref) from any `Vector` of elements:

```julia
using TrackPad

d  = Drift(1.5)
qf = Quadrupole(0.3;  k1 =  1.8, name = :QF)
qd = Quadrupole(0.3;  k1 = -1.8, name = :QD)

fodo = Lattice([qf, d, qd, d])
```

Standard collection operations work directly:

```julia
length(fodo)      # 4
fodo[2]           # Drift element
```

## Single-Particle Tracking

[`linepass`](@ref) is the allocation-free inner loop — it returns a new
`SVector{6}` without heap allocation:

```julia
using TrackPad, StaticArrays

beam = Beam(3.0e9)
r0   = SVector(1e-3, 0.0, 0.5e-3, 0.0, 0.0, 0.0)

# One pass through the lattice
r1 = linepass(fodo, r0, beam)

# 100 turns around a ring
r100 = ringpass(fodo, r0, beam, 100)
```

### Lost Particles

[`check_lost`](@ref) returns `true` when any coordinate is `NaN`.  Both
`linepass` and `ringpass` propagate the NaN state automatically — no exception
is thrown:

```julia
r_lost = ringpass(fodo, r0, beam, 10_000)
check_lost(r_lost)   # true if the particle was lost
```

## Multi-Particle Tracking

For large ensembles use the in-place variants [`linepass!`](@ref) and
[`ringpass!`](@ref), which operate on a pre-allocated matrix:

```julia
using TrackPad

N     = 1000
beam  = Beam(3.0e9)

# Rows = particles, columns = 6 phase-space coordinates
coords     = zeros(N, 6)
lost_flags = zeros(Int, N)   # 0 = alive, 1 = lost

for i in 1:N
    coords[i, 1] = 1e-4 * randn()  # x
    coords[i, 3] = 1e-4 * randn()  # y
end

# Track 200 turns in-place
ringpass!(coords, fodo, beam, lost_flags, 200)

println(count(==(0), lost_flags), " / $N particles survived")
```

### Element-Level Tracking

The primitive `pass!(elem, r, β_inv)` is the lowest-level interface:

```julia
β_inv = beti(beam)
r_out = pass!(qf, r0, β_inv)
```

## Linear Optics

All linear-optics routines live in `src/optics.jl` and use finite-difference
Jacobians.

### Tunes

```julia
qx, qy = gettune(fodo, beam)
```

Returns fractional tunes in ``[0, 1)`` from the 4×4 transverse block of the
one-turn map.

### Chromaticity

```julia
ξx, ξy = getchrom(fodo, beam)
```

Natural chromaticity via finite-difference tune variation with momentum.

### Twiss Along a Line

[`twissline`](@ref) propagates Courant–Snyder parameters through each element
and returns a [`TwissLineResult`](@ref):

```julia
tw = twissline(fodo, beam)

println("Qx = ", tw.tunex)
println("max βx = ", maximum(tw.betax), " m")
```

| Field | Description |
|-------|-------------|
| `s` | longitudinal positions [m] |
| `betax`, `betay` | β-functions [m] |
| `alphax`, `alphay` | α-functions |
| `mux`, `muy` | accumulated betatron phase [rad] |
| `tunex`, `tuney` | total phase advance / 2π |

### Periodic Solution

[`twissring`](@ref) finds the self-consistent Courant–Snyder solution:

```julia
twiss = twissring(fodo, beam)
```

### Closed Orbit

```julia
co = find_closed_orbit_6d(fodo, beam)   # full 6D Newton search
co = find_closed_orbit_4d(fodo, beam; dp = 1e-3)  # 4D at fixed δ
```

### One-Turn Map

```julia
M = one_turn_map(fodo, beam)   # 6×6 finite-difference Jacobian
```

## TPSA Transfer Maps

TrackPad supports Truncated Power Series Algebra via the
[PolySeries.jl](https://github.com/MSU-Beam-Dynamics/PolySeries.jl) extension.
Load `PolySeries` after `TrackPad` to activate it:

```julia
using TrackPad, PolySeries

ring = Lattice([
    Quadrupole(0.3;  k1 =  1.8),
    Drift(1.5),
    Quadrupole(0.3;  k1 = -1.8),
    Drift(1.5),
])
beam = Beam(3.0e9)

# Second-order one-turn map
M = tpsa_map(ring, beam; order = 2)

# First-order terms (transfer matrix R)
R = [M[i].c[j + 1] for i in 1:6, j in 1:6]
```

Each `M[i]` is a `CTPS` (complex truncated power series) object.
`M[i].c[1]` is the constant term, `M[i].c[2..7]` are the first-order
coefficients ``R_{ij}``, and higher entries are the second-order monomials.

!!! note "Hamiltonian convention"
    `tpsa_map` uses the exact relativistic drift Hamiltonian
    (controlled by `TrackPad.USE_EXACT_HAMILTONIAN`).  When comparing with
    other codes, make sure they use the same convention.

## Time-Dependent Parameters

Elements can be made time-varying with [`timed`](@ref):

```julia
using TrackPad

# Quadrupole whose k1 ramps linearly with turn number
k1_ramp(ctx) = 1.2 * (1 + 0.01 * ctx.turn)
qf_tv = timed(qf; k1 = TimeFunction(k1_ramp))

# Build a lattice with the time-varying element
ring_tv = Lattice([qf_tv, d, qd, d])

# Snapshot at turn 50 → plain Quadrupole with k1 = 1.2 * 1.50
lat50 = materialize_lattice(ring_tv; turn = 50)
```

See [API Reference — Time Dependence](@ref time_dependence) for the full API.

## Lattice Utilities

```julia
total_length(fodo)            # total arc length [m]
spos(fodo)                    # s-positions at each element boundary
findelem(fodo, :QF)           # indices of elements named :QF
```
