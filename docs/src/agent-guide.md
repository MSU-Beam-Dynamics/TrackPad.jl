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
| Track one initial condition through a line | `linepass(lat, r0, beam)` | `SVector{6}` |
| Track one initial condition for turns | `ringpass(lat, r0, beam, nturns)` | `SVector{6}` |
| Track an `N x 6` CPU matrix with explicit loss flags | `linepass!`, `ringpass!` | mutates matrix and flags |
| Track a packed batch on CPU/GPU | `batch_linepass!`, `batch_ringpass!` | mutates device/host matrix |
| Run correlated or Monte Carlo settings | `ParamSweepLattice(...; mode=:aligned)` | one result per trial |
| Run a systematic parameter grid | `ParamSweepLattice(...; mode=:cartesian)` | Cartesian product |
| Compute tunes or chromaticity | `gettune`, `getchrom` | two-value tuple |
| Propagate supplied entrance Twiss | `transport_twiss` | `TransportTwissResult` |
| Compute periodic uncoupled Twiss | `periodic_twiss` | `TwissLineResult` |
| Compute a first-order map | `one_turn_map` or `findm66` | `6 x 6` matrix |
| Compute closed orbit | `find_closed_orbit_4d/6d` | static vector |
| Read PALS or MAD-X | `read_pals`, `read_madx` | `(Lattice, Beam)` |
| Compute derivatives for many particles | Enzyme batch APIs | preallocated arrays |
| Compute a TPSA map | `tpsa_map` with PolySeries loaded | six TPSA outputs |

## Minimal Correct Model

```julia
using StaticArrays, TrackPad

beam = Beam(3.0e9)  # kinetic energy [eV]
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

## Required Assumptions to State

When producing analysis code or numerical comparisons, state:

1. reference particle, kinetic energy, mass, and signed charge;
2. coordinate order and units;
3. whether RF is active and how phase/lag was converted;
4. finite-difference steps and forward/centered chromaticity choice;
5. CPU/GPU backend and precision;
6. integration steps for thick nonlinear elements; and
7. whether optics assumes an uncoupled periodic lattice.

If one is unknown, ask or expose it as a parameter instead of silently choosing
a convention from another accelerator code.

In particular, never describe `delta` as ``(P-P_0)/P_0``. TrackPad stores the
canonical pair ``(z,\delta_E)`` with
``z=s/\beta_0-ct=-c(t-t_0)`` and ``\delta_E=(E-E_0)/(P_0c)``.

## Array Shapes

| Quantity | Shape | Index order |
|----------|-------|-------------|
| Particle coordinates | `N x 6` | particle, coordinate |
| Batched Jacobian | `N x 6 x 6` | particle, output, input |
| Batched Hessian-vector product | `N x 6 x 6` | particle, output, input |
| Batched Hessian | `N x 6 x 6 x 6` | particle, output, input1, input2 |
| Packed float parameters | `18 x M` | slot, element |
| Sweep values | `N x K` | configuration, varied parameter |

Preallocate derivative outputs on the same backend and with the same scalar type
as coordinates and `GPULattice`.

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
4. migration/status documents for historical context only.

Repository-changing agents must also follow `AGENTS.md` at the repository root.
