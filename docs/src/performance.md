```@meta
CurrentModule = TrackPad
```

# [Performance](@id performance_guide)

TrackPad's element kernels are allocation-free and generic; how fast a job runs
is decided almost entirely by three choices made above the kernels: which
tracking entry point you call, how many integration steps thick magnets carry,
and whether you use threads or a GPU. This page gives measured numbers for each
so the choice can be made deliberately. All CPU figures are from a single
Apple-silicon-class core (Julia 1.12) on a 240-element FODO arc with bends and
sextupoles, 4000 particles, in nanoseconds per particle-element; treat them as
ratios rather than absolutes.

## Choose the entry point

| entry point | layout | parallelism | ns / particle-element |
|---|---|---|---|
| `track` / `linepass` | one `SVector{6}` | none | ≈ same kernel cost, plus one dynamic dispatch per element |
| `track!` / `linepass!` | `N×6` matrix | none | 83 |
| `track!(...; threaded=true)` | `N×6` matrix | Julia threads | 24 on 4 threads (≈ 98 per thread) |
| `track!(coords, ::GPULattice)` | packed, device resident | GPU | see the [GPU guide](@ref gpu_guide) |

The scalar form is the right tool for optics, closed orbits and TPSA —
anything that tracks one particle at a time. For ensembles use one of the
matrix paths; both amortise element dispatch over all particles (a function
barrier specialises on each element's concrete type once per element), which is
why they are allocation-free and roughly 2.3x faster than calling `pass!` on
each particle yourself. `threaded=true` splits the bunch into contiguous chunks
and needs Julia started with `-t N`; without threads it matches the serial path
within a few percent.

`linepass!` keeps the JuTrack loss semantics (an integer flag, coordinates left
as they were at the loss); `cpu_batch_linepass!` writes `NaN`. See
[Loss Convention](@ref loss_convention).

## Integration steps

Thick magnets are sliced for the 4th-order Yoshida integrator, and cost is
linear in `num_int_steps`: about 42 ns per step per particle for a quadrupole,
against 12.5 ns for a drift. Because convergence is 4th order each doubling of
the step count buys ~16x in accuracy, so a small count is usually enough. The
defaults are set per element type (quadrupoles and bends 4, sextupoles 2,
octupoles 1; see [Integration steps](@ref integration_steps)). Their effect on a
ring, against a 200-step reference:

| `num_int_steps` (quads and bends) | Δtune | Δchromaticity | `track!` | `track!` threaded (4 threads) |
|---|---|---|---|---|
| 10 | 4.5e-06 | 2.1e-05 | 247 ns | 67 ns |
| 4 (default) | 1.8e-04 | 8.2e-04 | 84 ns | 24 ns |
| 2 | 2.9e-03 | 1.3e-02 | 52 ns | 15 ns |

Raise the count for precision optics matching or resonance studies, where a
2e-4 tune offset is larger than the effect under study; 8–10 brings it below
1e-5 at 2–2.5x the cost. Every count is symplectic, so what the choice trades
is a bounded map distortion, never a growth. Importers leave `num_int_steps`
at `nothing` (per-type defaults) unless told otherwise.

## Optics functions

The ring-optics functions default to Taylor maps (`method=:tpsa`). Measured on
one thread, warm, against `method=:fd`:

| lattice | quantity | `:fd` | `:tpsa` |
|---|---|---|---|
| 24-cell ring, 456 elements, 20 steps/element | Twiss + D + αc + ξ | 0.024 s | 0.065 s |
| | + second-order chromaticity | 0.046 s | 0.17 s |
| | + radiation integrals | 0.034 s | 0.093 s |
| | + amplitude detuning | 3.0 s | 0.27 s |
| 200-cell ring, 2400 elements, default steps | Twiss + D + αc + ξ | 0.045 s | 0.067 s |
| | + amplitude detuning | 5.5 s | 0.26 s |
| | `getchrom` | 0.033 s | 0.036 s |

Each quantity uses the lowest series order that contains it: order 1 (7
coefficients per coordinate) for the closed orbit and every Jacobian, order 2
(28) for the chromaticity, order 3 (84) for the amplitude detuning; the
second-order chromaticity is a three-point difference of order-1 closed-orbit
tunes seeded from the linear dispersion. A ring without orbit errors needs no
Newton solve at all — the per-element Jacobian pass shows the reference
closes — so a default `periodic_twiss` is one order-1 turn plus one order-2
turn. The Taylor maps cost 1.5–4× more than finite differences for the linear
and chromatic quantities and 5–20× less for the detuning, which `:fd` obtains
from 8 × 1024 turns of tracking. What `:tpsa` buys is the absence of step parameters and noise floors:
the closed orbit, Jacobians, ξ and ξ₂ are exact derivatives of the truncated
map (ξ agrees with PTC to 1e-6 relative, ξ₂ to 1e-6), while `:fd` carries a
~1e-6 absolute floor on ξ and ~1e-3 on ξ₂. Both are far below a second on
production-size rings, so the default is the exact one; choose `:fd` for a
lattice with an `LBend` (no series map) or when many thousands of Twiss
evaluations of a linear quantity are the bottleneck.

Finite-difference optics run the scalar path a fixed number of times:
`gettune` and `one_turn_map` need 12 single-particle passes, `getchrom` two
tune evaluations each preceded by a 4-D closed-orbit Newton solve (a few tens
of passes). `periodic_twiss` adds a Jacobian per element boundary; its
optional `second_order` adds one more chromaticity evaluation,
`radiation_integrals` is free, and `detuning` tracks `detuning_turns` turns at
each of the `detuning_actions` in both planes (8 × 1024 turns by default),
which dominates when it is on. The default step sizes sit at their measured
optimum (about 1e-6 absolute on ξ, seven correct digits); there is nothing to
gain by tuning `h` or `dpp` on a zero reference orbit.

`method=:tpsa` replaces all of that with one Taylor map per quantity (order 1
for the linear optics, 2 for chromaticity, 3 for detuning), whose cost is set
by the order rather than by step counts — see below.

## TPSA and AD

A TPSA map of order ``n`` in six variables carries ``\binom{n+6}{6}``
coefficients per coordinate, so cost and memory grow steeply with order: an
order-3 one-turn map of 1200 elements takes about 0.06 s and 600 MB of
transient allocation. Keep `order` as low as the question allows, and compute
maps about the closed orbit rather than tracking a TPSA particle for many
turns. Enzyme derivatives (`batch_jacobian!`, `batch_hessian_vector_product!`)
run on the packed `GPULattice` path and scale with particles the same way
batch tracking does.

## Writing fast kernels

If you add an element, keep its `pass!` free of heap allocation (build every
intermediate as an `SVector`/scalar), derive constants from the element's
scalar type `T` rather than the coordinate type `S`, and hoist anything that
does not depend on the coordinates — `tan(edge_angle)`, `sin(kick_angle)/L`,
`iszero(r1)` — into the element constructor or a precomputed field where the
type allows. The [Agent Guide](@ref agent_guide) lists the generic-kernel rules
that keep a new element working for TPSA and AD coordinates as well.
