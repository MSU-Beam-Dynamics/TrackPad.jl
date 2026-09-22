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
the end point is physically the same reference point as the start: multi-turn
tracking, `gettune`, `getchrom`, the periodic solution of [`twiss`](@ref) and
the closed-orbit searches all reject an open line.

Standard collection operations in Julia work directly:

```julia
length(fodo)      # 4
fodo[2]           # Drift element
```

## Single-Particle Tracking

[`track`](@ref) is the tracking entry point — one particle or a bunch, one
pass or many turns, serial, threaded or on a GPU. Given an `SVector{6}` it
returns a new one:

```julia
using TrackPad, StaticArrays

beam = Beam(3.0e9)
r0   = SVector(1e-3, 0.0, 0.5e-3, 0.0, 0.0, 0.0)

# One pass through the lattice
r1 = track(fodo, r0, beam)

# 100 turns around a ring
r100 = track(fodo, r0, beam; nturns=100)
```

`nturns > 1` requires a periodic lattice; a single pass works for a line or a
ring. `linepass(fodo, r0, beam)` is the JuTrack-compatible spelling of the
first call.

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

[`twiss`](@ref) is the one entry point. It finds the closed orbit, solves (or
propagates) the linear optics, and returns a single [`TwissResult`](@ref)
holding the Twiss functions, the dispersion, the momentum compaction and the
chromaticity — all from the same set of transfer maps, so they are guaranteed
to describe the same machine. Rings and open lines differ only by whether you
supply entrance optics.

### The ring in one call

```julia
tw = twiss(ring, beam)          # ≡ periodic_twiss(ring, beam)

tw.tunex, tw.tuney              # fractional tunes
maximum(tw.betax)               # peak β_x [m]
tw.chromx, tw.chromy            # ξ = dQ/dδ_P about the off-momentum closed orbit
tw.alphac, transition_gamma(tw) # momentum compaction and γ_tr
tw.dx[1], tw.dpx[1]             # dispersion at the start [m], [–]
```

Array fields are sampled at every element boundary and share the `s` grid, so
they plot directly against `tw.s`:

| Field | Description |
|-------|-------------|
| `s` | longitudinal positions [m] |
| `betax`, `betay` | β-functions [m] |
| `alphax`, `alphay` | α-functions |
| `mux`, `muy` | accumulated betatron phase [rad] |
| `dx`, `dpx`, `dy`, `dpy` | dispersion ``\mathrm{d}(x,p_x,y,p_y)/\mathrm{d}\delta_P`` [m, –] |

and the scalars describe the lattice as a whole:

| Field | Description |
|-------|-------------|
| `tunex`, `tuney` | fractional tunes of a ring; total phase advance / 2π of a line |
| `length` | path length [m] (the circumference of a ring) |
| `alphac` | momentum compaction ``(1/C)\,\mathrm{d}C/\mathrm{d}\delta_P`` |
| `slip` | slip factor ``\eta=\alpha_c-1/\gamma_0^2`` ([`transition_gamma`](@ref) derives γ_tr) |
| `chromx`, `chromy` | chromaticity ``\mathrm{d}Q/\mathrm{d}\delta_P`` about the closed orbit |
| `chrom2x`, `chrom2y` | ``\delta_P^2`` coefficient of ``Q(\delta_P)`` (opt-in) |
| `radiation` | `(I1, I2, I3, I4, I5)` synchrotron-radiation integrals (opt-in) |
| `detuning` | 2×2 ``\partial Q_i/\partial J_j`` [1/(m·rad)] (opt-in) |
| `periodic` | `true` for a ring's periodic solution, `false` for propagated optics |
| `method` | how the maps were obtained: `:tpsa` (default) or `:fd` |

A field that was not requested, or that an open line does not have, is
`nothing` — so `tw.alphac === nothing` is how you ask "was this a line?".

### Quantities you ask for

Three quantities cost a higher-order map or extra tracking, so they are opt-in
and `nothing` otherwise:

```julia
tw = twiss(
    ring, beam;
    second_order        = true,   # chrom2x, chrom2y = ½ d²Q/dδ_P²
    radiation_integrals = true,   # I1…I5 over the bend bodies
    detuning            = true,   # ∂Q/∂J, the amplitude-dependent tune shift
)

tw.chrom2x                        # ½ d²Qx/dδ_P²
tw.radiation.I2                   # 2π/ρ for a ring of identical bends
tw.detuning[1, 1]                 # ∂Qx/∂Jx
```

`second_order` and `detuning` need a periodic lattice; `radiation_integrals`
works for a line as well. See [Performance](@ref performance_guide) for what
each one costs.

### Open lines, and chaining sections

An open line has no periodic solution, so give `twiss` the entrance optics —
β and α of both planes, optionally the phases and the dispersion
``(\eta,\eta')`` in the same ``\delta_P`` convention as the result:

```julia
line     = Lattice(AbstractElement[qf, d, qd])
entrance = optics4DUC(12.0, -0.4, 8.0, 0.2)                  # β, α only
entrance = optics4DUC(optics2D(12.0, -0.4, 0.0, 0.5, 0.02),  # + phase, η, η′
                      optics2D( 8.0,  0.2, 0.0, 0.0, 0.0))

tw = twiss(line, beam; entrance = entrance)   # ≡ transport_twiss(line, beam, entrance)
```

The result is the same `TwissResult`, with the ring-only fields `nothing` and
`tunex`/`tuney` reporting the total phase advance over 2π. A `TwissResult`
itself works as an `entrance`, continuing from its exit values (phases and
dispersion included), so sections chain:

```julia
t_arc  = twiss(arc,  beam; entrance = t_injector)
t_next = twiss(next, beam; entrance = t_arc)
```

For a line, `reference` is the launch coordinate rather than a closed-orbit
seed; its sixth entry is the momentum at which the optics are evaluated.

### Smooth curves: sampling inside elements

By default one point is returned per element boundary. To resolve the β-beat
inside a long quadrupole or to plot a smooth dispersion, oversample — the
lattice itself is untouched:

```julia
tw = twiss(ring, beam; slices = 10)                     # ≥ 10 points per element
tw = twiss(ring, beam; max_step = 0.10)                 # pieces ≤ 10 cm, uniform in s
tw = twiss(ring, beam; sample_integrator_steps = true)  # one point per integration step
```

The three combine, and the largest count per element wins. Each piece
integrates in one step, so sampling never uses fewer steps than the element was
configured with; asking for more pieces than steps *refines* the integration,
which moves the numbers by the integrator error of the coarse lattice (about
``10^{-4}`` in the tune at the default step counts) towards their converged
values. Bend curvature and body multipoles are distributed over the pieces;
entrance pole-face, fringe, offset and rotation maps stay on the first piece and
their exit counterparts on the last. [`refine_lattice`](@ref) returns the
sampled lattice itself when you need element-to-sample correspondence.

### How the derivatives are taken

```julia
tw = twiss(ring, beam; method = :fd)      # finite differences of tracking
tw = twiss(ring, beam; wrt = :deltae)     # derivatives w.r.t. the stored δ_E
```

`method = :tpsa` (the default) reads every derivative off a truncated Taylor
map about the closed orbit: the orbit and the transfer matrices from an order-1
map, the chromaticity from order 2, the amplitude detuning from order 3. There
are no step sizes and no noise floor. `method = :fd` differentiates tracking
instead; it is the route for a lattice holding an element with no series map
(`LBend`), and it is somewhat faster for the linear quantities.

`wrt` selects the momentum variable for *every* derivative in the result —
dispersion, `alphac`, `slip` and the chromaticities. The default
``\delta_P=(P-P_0)/P_0`` is what MAD-X, elegant and AT report; `:deltae`
switches to TrackPad's stored ``\delta_E`` (see [Conventions](@ref
conventions)).

### [Tunes and chromaticity on their own](@id chromaticity)

When the full optics are not needed:

```julia
qx, qy = gettune(ring, beam)     # fractional tunes from the 4×4 one-turn block
ξx, ξy = getchrom(ring, beam)    # dQ/dδ_P, identical to tw.chromx, tw.chromy
```

Both take the same `method` and `wrt` keywords. The chromaticity is always
measured about the **off-momentum closed orbit**: the 4-D closed orbit is
re-solved at each momentum, so every sextupole is sampled at ``x=D\,\delta_P``
and its feed-down gradient ``k_2D\,\delta_P`` — the sextupole chromaticity
correction — is included by construction. This is what makes the textbook
relation ``\Delta\xi_x=\tfrac{1}{4\pi}\oint\beta_x k_2 D\,\mathrm{d}s`` hold
with the ``D`` in `tw.dx`.

### Closed orbit

```julia
co = find_closed_orbit_6d(ring, beam)              # full 6-D Newton search
co = find_closed_orbit_4d(ring, beam; dp = 1e-3)   # 4-D at fixed δ_P = 1e-3
```

### Transfer maps

```julia
M     = one_turn_map(ring, beam)                   # 6×6 Jacobian of one turn
Mline = transfer_map(Lattice(AbstractElement[qf, d, qd]), beam)   # any ordered path
```

### JuTrack-style entry points

`twissline(ring, beam)` is the historic spelling of `periodic_twiss`. The
`twissring`, `findm66` and `fastfindm66` overloads take a momentum offset
``\delta_P`` and a map order, and return JuTrack-shaped results; `wrt=:deltae`
switches the offset to the stored coordinate.

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

Any field of any element can be made a function of time and turn number, which
covers ramps, modulation, and kickers that fire on a chosen turn.
[`timed`](@ref) wraps an element and the fields that should vary:

```julia
using TrackPad

t, nturn = Time(), Turn()

qf_ramp = timed(qf; k1 = 1.8 * (1 + 0.01 * nturn))        # 1 % per turn
sx_mod  = timed(sx; k2 = 3.0 + 0.2 * sin(2π * 60 * t))    # 60 Hz ripple
```

### The evaluation context

Every tracking and optics call resolves these fields through a
[`TimeContext`](@ref), which carries both a physical time and a turn index:

```julia
TimeContext(real_time; turn = 0)
```

[`Time`](@ref) and [`Turn`](@ref) are symbolic stand-ins for its two fields.
They support ordinary arithmetic and the usual math functions (`sin`, `exp`,
`sqrt`, …), so an expression built from them *is* the parameter — no closure
needed. Where an expression is not enough — a conditional, a table lookup, or a
vector-valued field — pass a function of the context instead:

```julia
# an injection kicker that fires on turn 3 only
kicker = timed(Corrector(0.0, 0.0, 0.0);
               xkick = ctx -> ctx.turn == 3 ? 1.0e-4 : 0.0)

# a whole multipole vector at once
ramped = timed(sx; polynom_b = ctx -> SVector(0.0, 0.0, 2.0 * ctx.turn, 0.0))
```

`timed` checks the field names against the element type, so a typo throws
immediately rather than being silently ignored.

### When it is evaluated

Tracking resolves the lattice **once per turn**, at
`(time + (n-1) * dt_turn, turn + n - 1)` for the `n`-th turn, so a multi-turn
call steps both the clock and the turn counter:

```julia
r = track(ring, r0, beam; nturns = 100, time = 0.0, dt_turn = 1e-6, turn = 0)
```

`track!` takes the same three keywords, and the turn index advances even when
`dt_turn` is zero — which is what makes `Turn()` useful on its own.

Optics functions evaluate at `(0, 0)`, because the Twiss parameters of a
machine that is changing are only defined for a frozen snapshot. Take the
snapshot yourself when you want the optics at another moment:

```julia
lat50 = materialize_lattice(ring_tv; time = 0.0, turn = 50)   # plain elements
tw50  = twiss(lat50, beam)
```

[`materialize`](@ref) does the same for a single element, which is the quickest
way to check that a ramp does what you think:

```julia
materialize(qf_ramp, TimeContext(0.0; turn = 50)).k1   # 1.8 * 1.5
```

!!! warning "Time-varying elements stay on the CPU"
    A `TimeVaryingElement` holds host closures, so it cannot be packed into a
    [`GPULattice`](@ref) or expanded into a Taylor map. Call
    `materialize_lattice` for the time and turn you want first, then adapt or
    expand the resulting static lattice.

See [API Reference — Time Dependence](@ref time_dependence) for the full API.

## Lattice Utilities

### Geometry and lookup

```julia
length(ring)             # number of elements
ring[2]                  # the second element
isperiodic(ring)         # was it built with periodic=true?
total_length(ring)       # total arc length [m]
spos(ring)               # s at every element boundary [m] (length(ring) + 1 values)
get_length(ring[2])      # arc length of one element [m]
findelem(ring, :QF)      # indices of every element named :QF
```

`findelem` returns indices into the lattice, which is what the element-index
arguments of [`ParamSweepLattice`](@ref) and the `refpts` interfaces expect.

### Derived lattices

Elements are immutable, so these return a *new* lattice rather than modifying
one:

```julia
refine_lattice(ring; slices = 10)                  # split elements for sampling
materialize_lattice(ring_tv; time = 0.0, turn = 5) # freeze time-varying elements
GPULattice(ring, beam)                             # pack for the CPU/GPU kernels
```

[`refine_lattice`](@ref) takes the same `slices`, `max_step` and
`sample_integrator_steps` keywords as [`twiss`](@ref), and is what that function
uses internally.

### Drawing the lattice

After loading a Makie backend, draw a lattice strip on its own axis and link it
to any longitudinal plot:

```julia
using CairoMakie, TrackPad

fig      = Figure()
ax_lat   = Axis(fig[1, 1])
ax_twiss = Axis(fig[2, 1], xlabel = "s [m]", ylabel = "β [m]")

plot_lattice!(ax_lat, ring)
lines!(ax_twiss, tw.s, tw.betax)
lines!(ax_twiss, tw.s, tw.betay)
linkxaxes!(ax_lat, ax_twiss)
```

[`lattice_plot_data`](@ref) returns the same glyph positions and element classes
without requiring Makie, for any other plotting backend. Physical element
boundaries are in `s_start` and `s_end`; zero-length BPMs, kickers and cavities
are given a small visible `plot_start` to `plot_end` width so that they can
still be seen.

## Common Failure Modes

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `MethodError` constructing a multipole | Strength supplied as a keyword | Use `Quadrupole(L, k1)`, `Sextupole(L, k2)`, or `Octupole(L, k3)` |
| RF cavity produces no kick | Direct cavity has `energy=0` | Set `energy=beam.energy` and `charge=beam.charge` |
| GPU packing throws `ArgumentError` | Element or setting is unsupported | Keep that model on CPU or implement/test exact GPU support; do not remove physics silently |
| `Enzyme` missing in tests | `test/runtests.jl` was run directly | Use `julia --project=. -e 'using Pkg; Pkg.test()'` |
| Cross-code chromaticity differs | Convention or bend-model mismatch | Record `wrt`, `method` (and `dpp` for `:fd`), the RF state, and the bend model — see [Bend models](@ref bend_models) |
| Metal type error | Metal does not support `Float64` kernels | Use `Float32` on Metal or `Float64` CPU/CUDA |

## Next Steps

- [Elements](@ref elements_guide): constructors and model categories
- [Lattice File I/O](@ref io_guide): PALS and MAD-X
- [GPU Acceleration](@ref gpu_guide): batch tracking and derivatives
- [Agent Guide](@ref agent_guide): explicit API and array-shape contract
