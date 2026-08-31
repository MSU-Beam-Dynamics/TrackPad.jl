```@meta
CurrentModule = TrackPad
```

# [Getting Started](@id user_guide)

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
| 2 | ``p_x`` | normalized canonical horizontal momentum ``P_x/P_0`` |
| 3 | ``y`` | vertical position [m] |
| 4 | ``p_y`` | normalized canonical vertical momentum ``P_y/P_0`` |
| 5 | ``z`` | canonical longitudinal coordinate ``s/\beta_0-ct`` [m] |
| 6 | ``\delta_E`` | relative energy deviation ``(E-E_0)/(P_0c)`` |

Coordinates are stored as `StaticArrays.SVector{6,T}`. The element type `T`
may be `Float64` for normal tracking or a `PolySeries.CTPS` type for TPSA maps.
The complete normative definition is in
[Physics and Data Conventions](@ref conventions).

The longitudinal canonical pair is
``(z,\delta_E)=(-c(t-t_0),(E-E_0)/(P_0c))``. A particle arriving early has
positive ``z``. TrackPad does not use ``\delta_P=(P-P_0)/P_0``; see the
normative page for the exact conversion.

## Beam

The [`Beam`](@ref TrackPad.Beam) struct carries the reference kinetic energy,
rest-mass energy, signed charge, and relativistic factors. Energies are in eV.

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
qf = Quadrupole(0.3,  1.8; name=:QF)
qd = Quadrupole(0.3, -1.8; name=:QD)

fodo = Lattice(
    AbstractElement[qf, d, qd, d];
    name=:FODO,
    periodic=true,
)
```

`Lattice(elements)` is an open line by default. Set `periodic=true` only when
the end point is physically the same reference point as the start. Multi-turn
tracking, tune, chromaticity, periodic Twiss, and closed-orbit APIs reject open
lines.

Standard collection operations work directly:

```julia
length(fodo)      # 4
fodo[2]           # Drift element
```

## Single-Particle Tracking

[`linepass`](@ref) is the scalar tracking interface and returns a new
`SVector{6}`:

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

[`check_lost`](@ref) checks NaN state and the current global transverse
coordinate/momentum safety limits. Both `linepass` and `ringpass` stop at the
first detected loss; no exception is thrown:

```julia
r_lost = ringpass(fodo, r0, beam, 10_000)
check_lost(r_lost)   # true if the particle was lost
```

## Multi-Particle Tracking

For large ensembles use the in-place variants [`linepass!`](@ref) and
[`ringpass!`](@ref), which operate on a pre-allocated matrix:

```julia
using Random, TrackPad

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

### Matched Gaussian Distributions

TrackPad keeps the reference [`Beam`](@ref TrackPad.Beam) separate from macroparticle
coordinates. Generate an `N x 6` ensemble and pass it directly to the batch
tracking APIs:

```julia
using Random

entrance = optics4DUC(12.0, -1.2, 7.0, 0.4)
coords = matched_gaussian(
    MersenneTwister(1234), 10_000, entrance;
    emitx=20e-9,
    emity=2e-9,
    emitz=1e-6,
    betaz=0.2,
    dispersion=[0.35, -0.04, 0.0, 0.0],
    crab_dispersion=[0.02, 0.0, 0.0, 0.0],
)

sigma = beam_covariance(coords)
projected_emittances(sigma)
eigenemittances(sigma)
```

The response vectors are ordered `(x, px, y, py)`. `dispersion` multiplies
TrackPad's ``\delta_E`` coordinate, while `crab_dispersion` multiplies ``z``.
The `eta` and `etap` fields of an [`optics4DUC`](@ref) value supply ordinary
dispersion when the explicit keyword is omitted. MAD-X momentum dispersion must
be divided by the reference ``\beta_0`` before use.

For a fully coupled beam, construct the target covariance directly:

```julia
coords4 = gaussian_distribution(MersenneTwister(1), 1000, sigma4)
coords6 = gaussian_distribution(MersenneTwister(2), 1000, sigma6;
                                centroid=closed_orbit)
```

`sigma4` is in `(x, px, y, py)` and `sigma6` follows TrackPad's full canonical
order. By default, [`gaussian_distribution`](@ref) and
[`matched_gaussian`](@ref) whiten finite-sample random correlations and impose
the requested centroid and covariance to roundoff. Set `exact_moments=false`
for statistically independent samples. Exact matching requires more particles
than coordinates and produces a constrained finite ensemble.

[`projected_emittances`](@ref) reports the determinant of each diagonal
canonical plane. For coupled or dispersive covariance matrices these are not
the normal-mode invariants; use [`eigenemittances`](@ref) for the sorted
symplectic eigen-emittances.

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

Natural chromaticity via finite-difference tune variation with TrackPad's
sixth coordinate ``\delta_E``.

### Periodic Ring Twiss

[`periodic_twiss`](@ref) solves the periodic entrance condition, propagates it
through each element, and returns a [`TwissLineResult`](@ref):

```julia
tw = periodic_twiss(fodo, beam)

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

The older `twissline(fodo, beam)` spelling remains compatible. The `twissring`
overloads are JuTrack-style interfaces accepting a sixth-coordinate energy
offset and map order.

### Open-Line Twiss

An open line has no periodic entrance solution. Supply entrance Twiss values:

```julia
line = Lattice(AbstractElement[qf, d, qd])
entrance = optics4DUC(12.0, -0.4, 8.0, 0.2)
tw = transport_twiss(line, beam, entrance)
```

The result contains boundary positions, beta, alpha, and accumulated phase,
but no tune.

### Closed Orbit

```julia
co = find_closed_orbit_6d(fodo, beam)   # full 6D Newton search
co = find_closed_orbit_4d(fodo, beam; dp = 1e-3)  # 4D at fixed δ_E
```

### One-Turn Map

```julia
M = one_turn_map(fodo, beam)   # 6×6 finite-difference Jacobian
```

For an open line, use the same finite-difference machinery without a closure
assumption:

```julia
line = Lattice(AbstractElement[qf, d, qd])
Mline = transfer_map(line, beam)
```

## TPSA Transfer Maps

TrackPad supports Truncated Power Series Algebra via the
[PolySeries.jl](https://github.com/MSU-Beam-Dynamics/PolySeries.jl) extension.
Load `PolySeries` after `TrackPad` to activate it:

```julia
using TrackPad, PolySeries

ring = Lattice([
    Quadrupole(0.3, 1.8),
    Drift(1.5),
    Quadrupole(0.3, -1.8),
    Drift(1.5),
]; periodic=true)
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
qf_tv = timed(qf; k1=k1_ramp)

# Build a lattice with the time-varying element
ring_tv = Lattice([qf_tv, d, qd, d]; periodic=true)

# Snapshot at turn 50 → plain Quadrupole with k1 = 1.2 * 1.50
lat50 = materialize_lattice(ring_tv; turn = 50)
```

See [API Reference — Time Dependence](@ref time_dependence) for the full API.

!!! warning "GPU materialization"
    `TimeVaryingElement` stores host closures. Call `materialize_lattice` for a
    specific time/turn before constructing a `GPULattice`.

## Lattice Utilities

```julia
total_length(fodo)            # total arc length [m]
spos(fodo)                    # s-positions at each element boundary
findelem(fodo, :QF)           # indices of elements named :QF
```

## Common Failure Modes

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `MethodError` constructing a multipole | Strength supplied as a keyword | Use `Quadrupole(L, k1)`, `Sextupole(L, k2)`, or `Octupole(L, k3)` |
| RF cavity produces no kick | Direct cavity has `energy=0` | Set `energy=beam.energy` and `charge=beam.charge` |
| GPU packing throws `ArgumentError` | Element or setting is unsupported | Keep that model on CPU or implement/test exact GPU support; do not remove physics silently |
| `JuTrack`/`PolySeries` missing in tests | `test/runtests.jl` was run directly | Use `julia --project=. -e 'using Pkg; Pkg.test()'` |
| Cross-code chromaticity differs | Convention/step/reference mismatch | Record `h`, `dpp`, centered mode, closed-orbit mode, RF state, and bend geometry |
| Metal type error | Metal does not support `Float64` kernels | Use `Float32` on Metal or `Float64` CPU/CUDA |

## Next Steps

- [Elements](@ref elements_guide): constructors and model categories
- [Lattice File I/O](@ref io_guide): PALS and MAD-X
- [GPU Acceleration](@ref gpu_guide): batch tracking and derivatives
- [Agent Guide](@ref agent_guide): explicit API and array-shape contract
