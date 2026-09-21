```@meta
CurrentModule = TrackPad
```

# [Getting Started](@id user_guide)

This guide walks through the typical workflow: building a beam and lattice,
tracking particles, computing linear optics, and extracting TPSA transfer maps.

## Coordinate Convention

TrackPad uses the standard 6D phase-space vector

```math
\mathbf{r} = (x,\ p_x,\ y,\ p_y,\ z,\ \delta_E)
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
positive ``z``. TrackPad will translate `dE` to `dp` when needed, for example in
[`getchrom`](@ref) and [`periodic_twiss`](@ref). The default is to use
``\delta_P=(P-P_0)/P_0`` for those routines, but `wrt=:deltae` switches to the
stored coordinate. 

## Beam

The [`Beam`](@ref TrackPad.Beam) struct carries the reference total energy,
rest-mass energy, signed charge, and the derived ``\gamma_0`` and ``\beta_0``.
Energies are in eV, and the positional energy is the **total** energy, as in
MAD-X:

```julia
using TrackPad

beam   = Beam(3.0e9)                                  # 3 GeV electron (total energy)
p_beam = Beam(6.5e12; mass = M_PROTON, charge = 1.0)  # 6.5 TeV proton

# Low-energy machines usually quote kinetic energy or momentum instead:
ion    = Beam(kinetic = 200.0e6, mass = M_PROTON, charge = 1.0)   # 200 MeV kinetic proton
inj    = Beam(pc = 1.2e9, mass = M_PROTON, charge = 1.0)          # P0c = 1.2 GeV

kinetic_energy(beam), p0c(beam)                       # the other two energy-like numbers
```

A positional energy below the rest mass is an error, which catches a kinetic
energy passed by mistake for a heavy particle.

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

Standard collection operations in Julia work directly:

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
r1 = track(fodo, r0, beam)

# 100 turns around a ring
r100 = track(fodo, r0, beam; nturns=100)
```

[`track`](@ref) is the one entry point for tracking: one particle or a bunch,
one pass or many turns, serial, threaded or on a GPU. `nturns > 1` requires a
periodic lattice. `linepass(fodo, r0, beam)` is the single-pass
JuTrack-compatible spelling of the first call.

### Lost Particles

[`check_lost`](@ref) checks NaN state and the current global transverse
coordinate/momentum safety limits. Tracking stops at the first detected loss
and returns `NaN` coordinates; no exception is thrown:

```julia
r_lost = track(fodo, r0, beam; nturns=10_000)
check_lost(r_lost)   # true if the particle was lost
```

An aperture loss is reported the same way (`check_apertures = false` tracks
through apertures as if they were absent). `linepass` keeps JuTrack's
convention instead and returns the coordinates the particle held when it
exceeded the limits.

## Multi-Particle Tracking

For large ensembles [`track!`](@ref) operates in place on a pre-allocated
matrix of 6D macroparticle coordinates. The first dimension is the particle
index, the second the canonical phase-space coordinates:

```julia
using Random, TrackPad

N     = 1000
beam  = Beam(3.0e9)

# Rows = particles, columns = 6 phase-space coordinates
coords = zeros(N, 6)
lost   = zeros(Int, N)   # optional: 0 = alive, nonzero = lost

for i in 1:N
    coords[i, 1] = 1e-4 * randn()  # x
    coords[i, 3] = 1e-4 * randn()  # y
end

# Track 200 turns in-place, on all Julia threads
track!(coords, fodo, beam; nturns=200, lost=lost, threaded=true)

println(count(==(0), lost), " / $N particles survived")
```

Lost particles are written back as `NaN` and skipped by the remaining elements
and turns; the `lost` vector is optional. `threaded=true` splits the bunch into
one contiguous chunk per Julia thread and gives bit-identical results, but
collective elements (longitudinal wakes) need the whole bunch and are serial
only. `track(coords, ...)` without the `!` tracks a copy and leaves the input
alone.

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

The response vectors are ordered `(x, px, y, py)`. The covariance is built in
the stored coordinates, so `dispersion` multiplies ``\delta_E`` and
`crab_dispersion` multiplies ``z``. Take it from the `dx`, `dpx`, `dy`, `dpy`
of `periodic_twiss(ring, beam; wrt=:deltae)`, or divide a ``\delta_P``
dispersion (MAD-X, or `periodic_twiss`'s default) by ``\beta_0``. The
`eta` and `etap` fields of an [`optics4DUC`](@ref) value supply ordinary
dispersion when the explicit keyword is omitted.

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
ξx, ξy = getchrom(ring, beam)                 # dQ/dδ_P about the off-momentum closed orbit
ξx, ξy = getchrom(ring, beam; wrt=:deltae)    # same quantity, differentiated with respect to δ_E
ξx, ξy = getchrom(ring, beam; method=:fd)     # finite differences of tracking instead of the Taylor map
```

`getchrom` returns ``\xi=\mathrm{d}Q/\mathrm{d}\delta_P`` with
``\delta_P=(P-P_0)/P_0``; its `dp`/`dpp` are offsets and steps in the same
variable. Pass `wrt=:deltae` to differentiate with respect to the stored
coordinate instead (see [Conventions](@ref conventions) for which quantities
use which).

The chromaticity is always measured about the **off-momentum closed orbit**:
for every momentum step the 4-D closed orbit is re-solved, so each sextupole is
sampled at ``x = D\,\delta_P``, where its feed-down gradient ``k_2 D\,\delta_P``
is exactly the sextupole chromaticity correction. 

By default (`method=:tpsa`) the order-2 one-turn Taylor map about the closed
orbit is differentiated analytically, including the closed-orbit term
``\sum_k D_k\,\partial Q/\partial x_k``; there is no step to choose. With
`method=:fd` the tunes are differentiated by finite differences of tracking
(`dpp=1e-6` centered by default; the noise floor is about ``10^{-6}``
absolute); the two agree to the truncation error of the finite difference.

### Dispersion

```julia
tw = periodic_twiss(ring, beam)
tw.dx, tw.dpx, tw.dy, tw.dpy        # at every element boundary, along tw.s
```

`tw.dx` is ``\mathrm{d}x/\mathrm{d}\delta_P`` along the closed orbit, the
standard dispersion, which is what makes the textbook feed-down relation
``\Delta\xi_x=\tfrac{1}{4\pi}\oint\beta_x k_2 D\,\mathrm{d}s`` hold with
the ``\xi`` that `getchrom` returns. Pass `wrt=:deltae` for
``\mathrm{d}x/\mathrm{d}\delta_E``. There is no separate dispersion
function: the dispersion is part of the periodic optics and comes from the same
Jacobians as the Twiss functions.

### Twiss: rings and lines

One function, [`twiss`](@ref), serves both cases and returns the same
[`TwissResult`](@ref). Without an `entrance` it finds the closed orbit of a
periodic lattice, solves the periodic condition and propagates it through each
element — the complete linear and first-order chromatic description of the
ring; with an `entrance` it propagates the supplied optics through the lattice
(an open line, or a single pass through a ring). [`periodic_twiss`](@ref) and
[`transport_twiss`](@ref) are the same function with the case spelled out.

```julia
tw = twiss(ring, beam)             # ≡ periodic_twiss(ring, beam)

println("Qx = ", tw.tunex, "  Qy = ", tw.tuney)
println("max βx = ", maximum(tw.betax), " m")
println("ξ = (", tw.chromx, ", ", tw.chromy, ")")
println("αc = ", tw.alphac, "  γ_tr = ", transition_gamma(tw))
```

| Field | Description |
|-------|-------------|
| `periodic` | `true` for the periodic solution of a ring, `false` for propagated entrance optics |
| `method` | `:tpsa` (default) or `:fd` |
| `s` | longitudinal positions [m] |
| `betax`, `betay` | β-functions [m] |
| `alphax`, `alphay` | α-functions |
| `mux`, `muy` | accumulated betatron phase [rad] |
| `tunex`, `tuney` | fractional tunes of a ring; total phase advance / 2π of a line |
| `dx`, `dpx`, `dy`, `dpy` | dispersion ``\mathrm{d}(x,p_x,y,p_y)/\mathrm{d}\delta_P`` [m, –] |
| `length` | path length [m] (the circumference of a ring) |
| `alphac` | momentum compaction ``(1/C)\,\mathrm{d}C/\mathrm{d}\delta_P`` (ring; `nothing` for a line) |
| `slip` | slip factor ``\eta = \alpha_c - 1/\gamma_0^2`` (ring) |
| `chromx`, `chromy` | chromaticity ``\mathrm{d}Q/\mathrm{d}\delta_P`` about the closed orbit (ring) |
| `chrom2x`, `chrom2y` | ``\delta_P^2`` coefficient of ``Q(\delta_P)`` with `second_order=true` (ring) |
| `radiation` | `(I1, I2, I3, I4, I5)` synchrotron-radiation integrals with `radiation_integrals=true` |
| `detuning` | 2×2 ``\partial Q_i/\partial J_j`` [1/(m·rad)] with `detuning=true` (ring) |

Fields that were not requested, or that a line does not have, are `nothing`.

The optional fields are filled on request, because each costs extra tracking
or a higher-order map:

```julia
tw = periodic_twiss(
    ring, beam;
    second_order        = true,   # chrom2x, chrom2y = ½ d²Q/dδ_P²  (3-point stencil, dpp2=1e-4)
    radiation_integrals = true,   # I1…I5 over the bend bodies with the periodic dispersion
    detuning            = true,   # ∂Q/∂J from 1024 turns at four actions + NAFF tune
)
tw.radiation.I2                   # 2π/ρ for a ring of identical bends
tw.detuning[1, 1]                 # ∂Qx/∂Jx
```

By default every quantity comes from truncated Taylor maps (`method=:tpsa`):
the closed orbit and the transfer matrices from an order-1 map, the
chromaticities from the order-2 map, and the amplitude detuning from the
order-3 map, with no step sizes to choose. `method=:fd` obtains the same
quantities from finite differences of tracking, which is also the route for lattices containing an element without a series
map (`LBend`). See [Performance](@ref performance_guide) for timings.

#### Sampling inside elements

By default, values are returned at the element boundaries. For smooth curves
and local extrema inside thick elements, oversample without touching the
lattice:

```julia
tw = twiss(ring, beam; slices = 10)       # ≥ 10 points in every drift, magnet and bend
tw = twiss(ring, beam; max_step = 0.10)   # pieces no longer than 10 cm (uniform in s)
tw = twiss(ring, beam; sample_integrator_steps = true)   # one point per configured step
```

The three combine (the largest count per element wins). Each piece integrates
in one step, so sampling never integrates an element in fewer steps than
configured; asking for more pieces than steps refines its integration, which
moves the numbers by the integrator error of the coarse lattice (about
``10^{-4}`` for the default step counts — see [Performance](@ref
performance_guide)) towards the converged values. Bend curvature and body
multipoles are distributed over the pieces; entrance pole-face, fringe, offset
and rotation maps sit on the first piece, exit maps on the last.
[`refine_lattice`](@ref) exposes the sampled lattice itself when
element-to-sample correspondence is needed.

The `wrt=:deltae` keyword switches every momentum derivative in the result —
dispersion, `alphac`, `slip`, chromaticities — to the stored ``\delta_E``.

The older `twissline(ring, beam)` spelling remains compatible. The `twissring`
overloads are JuTrack-style interfaces taking a momentum offset ``\delta_P`` and
a map order (`wrt=:deltae` switches the offset to the stored coordinate).

### Lattice Illustration

After loading a Makie backend, draw a lattice strip on its own axis and link it
to any longitudinal plot:

```julia
using CairoMakie, TrackPad

fig = Figure()
ax_lattice = Axis(fig[1, 1])
ax_twiss = Axis(fig[2, 1], xlabel="s [m]", ylabel="beta [m]")
plot_lattice!(ax_lattice, fodo)
lines!(ax_twiss, tw.s, tw.betax)
linkxaxes!(ax_lattice, ax_twiss)
```

`lattice_plot_data(fodo)` provides the same glyph positions and element classes
without requiring Makie. This is useful for other plotting backends. Physical
element boundaries remain in `s_start` and `s_end`; zero-length BPMs, kickers,
and cavities receive a small visible `plot_start` to `plot_end` width.

### Open-Line Twiss

An open line has no periodic solution, so supply the entrance optics — β and α
of both planes, optionally the phases and the horizontal dispersion
``(\eta, \eta')`` in the same ``\delta_P`` convention as the result:

```julia
line = Lattice(AbstractElement[qf, d, qd])
entrance = optics4DUC(12.0, -0.4, 8.0, 0.2)                       # β, α only
entrance = optics4DUC(optics2D(12.0, -0.4, 0.0, 0.5, 0.02),       # + phase, η, η′
                      optics2D(8.0, 0.2, 0.0, 0.0, 0.0))
tw = transport_twiss(line, beam, entrance)                        # ≡ twiss(line, beam; entrance)
```

The result is the same [`TwissResult`](@ref) as for a ring: β, α, phases,
dispersion propagated from the entrance, `tunex`/`tuney` as the total phase
advance over 2π, and `radiation_integrals=true` works. The ring-only fields
(`alphac`, `slip`, chromaticities, `detuning`) are `nothing`, and asking for
`second_order` or `detuning` is an error. Lines chain: passing a `TwissResult`
as the entrance continues from its exit values, phases included,

```julia
t1 = twiss(arc,  beam; entrance = tw_injector)
t2 = twiss(next, beam; entrance = t1)
```

and `reference` is the launch coordinate (its sixth entry the momentum), not a
closed-orbit seed.

### Closed Orbit

```julia
co = find_closed_orbit_6d(fodo, beam)   # full 6D Newton search
co = find_closed_orbit_4d(fodo, beam; dp = 1e-3)  # 4D at fixed δ_P = 1e-3
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

TrackPad's element kernels run on the truncated power series of
[PolySeries.jl](https://github.com/MSU-Beam-Dynamics/PolySeries.jl), a
dependency, so a Taylor map of any lattice is one call away (`using PolySeries`
is only needed to read the coefficients through its API):

```julia
using TrackPad, PolySeries

ring = Lattice(AbstractElement[
    Quadrupole(0.3, 1.8),
    Drift(1.5),
    Quadrupole(0.3, -1.8),
    Drift(1.5),
]; periodic=true)
beam = Beam(3.0e9)

# Second-order one-turn map about the origin
M = tpsa_map(ring, beam; order = 2)

# Read coefficients through the PolySeries API. `e(j)` is the exponent vector
# of the j-th coordinate; `element` returns the coefficient of a monomial.
e(j) = [k == j ? 1 : 0 for k in 1:6]
R    = [element(M[i], e(j)) for i in 1:6, j in 1:6]        # transfer matrix
T116 = element(M[1], e(1) .+ e(6))                          # ∂²x/∂x∂δ
c    = [cst(M[i]) for i in 1:6]                             # constant terms
```

Each `M[i]` is a PolySeries `CTPS` — a truncated power series in the six
initial offsets. Always read coefficients with `cst` and `element`; the raw
coefficient vector is stored lazily by degree and is not safe to index
directly.

!!! note "Hamiltonian convention"
    `tpsa_map` uses the exact relativistic drift Hamiltonian, like all
    TrackPad tracking (`TrackPad.USE_EXACT_HAMILTONIAN` is a constant `true`).
    When comparing with other codes, make sure they use the same convention.

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
