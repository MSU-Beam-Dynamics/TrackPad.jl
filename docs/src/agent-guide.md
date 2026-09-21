```@meta
CurrentModule = TrackPad
```

# [Guide for AI Agents and Integrators](@id agent_guide)

This page gives a compact, explicit contract for an AI agent that writes code
using TrackPad. It complements the human [Getting Started](@ref user_guide) and
the normative [Physics and Data Conventions](@ref conventions).

## Decision Table

| User intent | Preferred public API | Result |
|-------------|----------------------|--------|
| Track one initial condition through a line or for `nturns` turns | `track(lat, r0, beam; nturns)` | `SVector{6}`, `NaN` if lost |
| Track an `N x 6` CPU matrix, optionally on Julia threads | `track!(coords, lat, beam; nturns, lost, threaded)` | mutates matrix; lost rows become `NaN` |
| Track with JuTrack's loss convention (flags, coordinates kept) | `linepass`, `linepass!` | single pass only |
| Generate a Gaussian with a target `4 x 4`/`6 x 6` covariance | `gaussian_distribution` | `N x 4`/`N x 6` matrix |
| Generate from Twiss, emittance, and longitudinal responses | `matched_gaussian` | `N x 6` matrix |
| Measure covariance or emittances | `beam_covariance`, `projected_emittances`, `eigenemittances` | matrices/vectors |
| Track a packed batch on CPU/GPU | `track!(coords, gpulattice; nturns)` | mutates device/host matrix |
| Run correlated or Monte Carlo settings | `ParamSweepLattice(...; mode=:aligned)` | one result per trial |
| Run a systematic parameter grid | `ParamSweepLattice(...; mode=:cartesian)` | Cartesian product |
| Compute tunes or chromaticity | `gettune`, `getchrom` | two-value tuple (``\delta_P`` derivative, closed orbit) |
| Compute periodic ring optics (Twiss, dispersion, αc, ξ; optional ξ₂, I1–I5, ∂Q/∂J) from Taylor maps | `twiss` / `periodic_twiss` | `TwissResult` |
| Propagate entrance optics (β, α, phase, dispersion) through a line, or chain lines | `twiss(...; entrance)` / `transport_twiss` | `TwissResult` (`periodic=false`) |
| Same from finite differences of tracking (e.g. a lattice with `LBend`) | `twiss(...; method=:fd)` | `TwissResult` |
| Smooth optics inside thick elements for plotting | `twiss(...; slices=10)` or `max_step=0.1` | `TwissResult` on the refined lattice |
| Compute a first-order map | `one_turn_map` or `findm66` | `6 x 6` matrix |
| Compute closed orbit | `find_closed_orbit_4d/6d` | static vector |
| Read PALS or MAD-X | `read_pals`, `read_madx` | `(Lattice, Beam)` |
| Compute derivatives for many particles | Enzyme batch APIs | preallocated arrays |
| Compute a TPSA map | `tpsa_map` | six TPSA outputs |

## Minimal Correct Model

```julia
using StaticArrays, TrackPad

beam = Beam(3.0e9)  # total energy [eV]; Beam(kinetic=K, mass=m) for low-energy ions
lat = Lattice(AbstractElement[
    Drift(1.0; name=:D1),
    Quadrupole(0.3, 0.7; name=:QF),
    Drift(1.0; name=:D2),
    Quadrupole(0.3, -0.7; name=:QD),
]; periodic=true)
r0 = @SVector [1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]
r1 = linepass(lat, r0, beam)
```

Do not generate `Quadrupole(L; k1=...)`; `k1` is positional. Use an
`AbstractElement` vector for heterogeneous lattices.

Use `periodic=true` only when the lattice end closes at its start. An omitted
keyword means an open line.

## Writing Generic Kernels

Element kernels are generic in the coordinate type (`SVector{6,S}`): the same
code runs on `Float64`, `Float32`, `ForwardDiff`/`Enzyme` duals and PolySeries
`CTPS` truncated power series. Two rules keep that working.

Build scalars from the *element's* numeric type `T`, never from the coordinate
type `S`. `one(T)` and `zero(T)` are fine; `one(S)` and `zero(S)` are not,
because `CTPS` has no type-level `one`/`zero` — a polynomial needs a
descriptor, and the type does not carry one. PolySeries provides the instance
forms `zero(p)` and `one(p)`, built over `p`'s descriptor, so use those if you
need an identity shaped like a coordinate you already hold. The same applies to
default arguments: `beti::Real = one(S)` silently breaks TPSA tracking, while
`beti::Real = 1.0` does not. For the same reason `zeros(CTPS{T}, n)` and
reductions over an empty collection cannot work; reductions over a non-empty
one are fine.

Guard every branch that inspects a coordinate. Ordered comparisons, `isnan`
and aperture tests are meaningless for a series or a dual, so loss checks go
through `_check_pz2`, `_check_tiny`, `_safe_clamp` and `check_lost`, which are
no-ops for non-`Real` coordinates.

## Required Assumptions to State

When producing analysis code or numerical comparisons, state:

1. reference particle, total energy (or kinetic energy / momentum, if given that way), mass, and signed charge;
2. coordinate order and units;
3. whether RF is active and how phase/lag was converted;
4. finite-difference steps (or `method=:tpsa`) and the chromaticity
   conventions in use — centered/one-sided, ``\delta_P`` vs. ``\delta_E``
   (`wrt`, default ``\delta_P``), and that TrackPad measures about the
   off-momentum closed orbit (a code that launches on-axis measures a
   different quantity once sextupoles are present);
5. CPU/GPU backend and precision;
6. integration steps for thick nonlinear elements; and
7. whether optics assumes an uncoupled periodic lattice.

For generated distributions, also state whether population (`1/N`) or sample
(`1/(N-1)`) covariance is used and whether `exact_moments=true` removed random
finite-sample correlations.

If one is unknown, ask or expose it as a parameter instead of silently choosing
a convention from another accelerator code.

In particular, never describe `delta` as ``(P-P_0)/P_0``. TrackPad stores the
canonical pair ``(z,\delta_E)`` with
``z=s/\beta_0-ct=-c(t-t_0)`` and ``\delta_E=(E-E_0)/(P_0c)``.

## Array Shapes

| Quantity | Shape | Index order |
|----------|-------|-------------|
| Particle coordinates | `N x 6` | particle, coordinate |
| Transverse covariance | `4 x 4` | `(x, px, y, py)` |
| Full covariance | `6 x 6` | `(x, px, y, py, z, delta_E)` |
| Batched Jacobian | `N x 6 x 6` | particle, output, input |
| Batched Hessian-vector product | `N x 6 x 6` | particle, output, input |
| Batched Hessian | `N x 6 x 6 x 6` | particle, output, input1, input2 |
| Packed float parameters | `18 x M` | slot, element |
| Sweep values | `N x K` | configuration, varied parameter |

Preallocate derivative outputs on the same backend and with the same scalar type
as coordinates and `GPULattice`.

Do not attach generated coordinates to `Beam`; `Beam` is reference-particle
metadata. `matched_gaussian` returns the ordinary matrix consumed by tracking.
Ordinary dispersion is with respect to `delta_E`; divide a MAD-X
momentum-dispersion vector by `beam.beta`. Crab dispersion multiplies TrackPad's
positive-early `z` coordinate.

## File and Interchange Workflow

Prefer native Julia `Lattice`/`Beam` values inside Julia. Use files or mappings
only at process/package boundaries.

For a PALS file:

```julia
lat, beam = read_pals("model.yaml"; lattice="machine", branch="ring")
```

Files requiring expressions, includes, controllers, forks, or full PALS
bookkeeping must be reduced by an external parser or consumer adapter before
calling `read_pals`.

Do not put application engine policy, worker management, or consumer result
types into TrackPad.

## Failure and Capability Handling

- Keep `strict=true` for file compilation unless exploratory fallback is
  explicitly requested.
- Treat `ArgumentError` from `GPULattice` as a capability rejection. Do not
  remove the offending setting or replace the element silently.
- Check scalar loss with `check_lost`; check packed batches for nonfinite rows.
- Do not promise coupled Twiss, full MAD-X language support, GPU TPSA, or exact
  nested-Enzyme CUDA Hessians.
- Confirm current supported GPU elements in [Known Limitations](@ref gpu_limitations).
- Treat periodic-operation rejection as a boundary error; do not silently set
  `periodic=true` unless physical closure is known.

## Source of Truth

When documentation and generated assumptions conflict, use this order:

1. `docs/src/conventions.md` for published physics/data conventions;
2. public docstrings and implementation in `src`;
3. focused tests in `test` for verified numerical behavior;
4. `CHANGELOG.md` for what changed between versions.

Repository-changing agents must also follow `AGENTS.md` at the repository root.
