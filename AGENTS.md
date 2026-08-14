# TrackPad Repository Guide for Coding Agents

This file is the operational contract for AI coding agents modifying TrackPad.
For physics conventions, `docs/src/conventions.md` is normative. For public API
usage, start with `README.md` and `docs/src/agent-guide.md`.

## Scope

TrackPad is a general Julia accelerator tracking engine. Keep application
topology, UI/session behavior, worker lifecycle, serialization policy, and
consumer-specific wrappers outside this package.

Do not add an application-named API to `src`. Add a general typed Julia
primitive only when the capability belongs to an accelerator tracking library.

## Source Layout and Include Order

`src/TrackPad.jl` defines the include order. Dependencies flow downward:

1. `src/elements.jl` includes immutable element definitions by category.
2. `src/time_dependence.jl` defines lazy time/turn-dependent wrappers.
3. `src/tracking.jl` defines scalar element maps and symplectic helpers.
4. `src/lattice.jl` defines `Beam`, `Lattice`, and line/ring tracking.
5. `src/optics.jl` defines finite-difference maps, optics, and closed orbit.
6. `src/tpsa.jl` declares optional TPSA entry points.
7. `src/io.jl` defines PALS/MAD-X interchange and compilation.
8. `src/gpu.jl` defines packed KernelAbstractions tracking and sweeps.

Optional integrations live in `ext/` and must not become hard dependencies:

- `TrackPadCUDAExt.jl`
- `TrackPadMetalExt.jl`
- `TrackPadEnzymeExt.jl`
- `TrackPadPolySeriesExt.jl`

## Non-Negotiable Physics Invariants

- Coordinate order is `(x, px, y, py, z, delta)`.
- `z = s/beta0 - c*t = -c(t - t0)` is positive for an early particle and is
  canonically paired with `delta`.
- `delta` is `delta_E = (E - E0)/(P0*c)`, not `(P - P0)/P0`.
- `Beam.energy` is kinetic energy in eV, not total energy or momentum.
- `Beam.mass` is rest-mass energy in eV.
- Scalar CPU tracking is the behavioral reference.
- The default drift uses the exact relativistic Hamiltonian.
- JuTrack parity is only normative where JuTrack uses the same canonical
  variables. Mixed-convention maps are compared in the ultrarelativistic limit;
  finite-`beta` behavior follows `docs/src/conventions.md` and canonical tests.
- `Sextupole.k2` and `Octupole.k3` are normalized strengths; tracking applies
  the `1/2!` and `1/3!` polynomial factors internally.
- `RFCavity.lag` is a longitudinal offset in metres, not radians or cycles.
- `RFCavity.energy > 0` and the correct reference charge are required for an RF
  kick in directly constructed cavities.
- RF kicks are normalized by `P0*c`, computed from reference kinetic energy and
  `beta`; do not restore JuTrack's `K0*beta0^2` approximation.
- Unsupported GPU elements/settings must throw during `GPULattice`
  construction. Never substitute a drift or marker silently.
- PALS `Patch` is a physical reference-frame transformation. Consumer adapters
  may choose another policy, but TrackPad must not special-case one consumer.

Do not change any convention without updating `docs/src/conventions.md`, adding
focused tests, and documenting compatibility consequences.

## Constructors and Types

Element constructors use positional physical strengths:

```julia
Drift(L; name=:D)
Quadrupole(L, k1; name=:QF)
Sextupole(L, k2; name=:SF)
Octupole(L, k3; name=:OF)
SBend(L, angle, e1=0.0, e2=0.0; name=:B)
RFCavity(L, volt, freq, lag=0.0; energy, charge, name=:RF)
Corrector(L, hkick, vkick; name=:COR)
Solenoid(L, ks; name=:SOL)
```

Mixed lattices should normally use `AbstractElement[...]` so Julia does not
infer an unusable concrete vector element type.

## Compatibility Rules

- Preserve canonical JuTrack numerical behavior where parity is already tested.
- Element parity tests generally use an absolute tolerance of `1e-15`.
- Keep finite-difference step defaults stable unless a targeted analysis and
  cross-code fixture justify a change.
- Keep the built-in PALS reader limited to its documented subset. Full-language
  expansion belongs in an external parser or consumer adapter.
- Keep machine topology, route selection, and source-occurrence metadata in
  the consumer adapter. TrackPad's public I/O returns only a `Lattice` and
  `Beam`.
- Never introduce TPSA implementation into core. PolySeries owns TPSA types.

## GPU Rules

`GPULattice` is a packed representation with one kernel launch per lattice
pass, not one kernel per element. Validate every newly supported element against
scalar CPU tracking in both `Float32` and `Float64` where the backend permits.

When adding an element:

1. Add/reuse an element type code.
2. Validate unsupported fields in `_validate_gpu_element`.
3. Pack parameters in `_gpu_fparams`/`_gpu_iparams`.
4. Implement `_gpu_apply_elem` behavior.
5. Add CPU KernelAbstractions parity tests.
6. Add CUDA/Metal hardware coverage if advertised.

Do not claim AMD/ROCm support without an AMDGPU extension and hardware test.

## Test Commands

Use the package target for the complete suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Directly including `test/runtests.jl` does not activate test-only dependencies.
Focused tests can be run after activating an environment that provides their
extras, for example:

```bash
julia --project=. test/verify_io.jl
julia --project=. test/verify_optics.jl
```

CUDA acceptance:

```bash
julia --project=test/cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=test/cuda test/cuda/runtests.jl
```

Documentation:

```bash
julia --project=docs docs/make.jl
git diff --check
```

If the execution sandbox cannot write `~/.julia`, prepend a writable depot and
retain the existing depot as a read fallback:

```bash
JULIA_DEPOT_PATH=/tmp/trackpad-julia-depot:$HOME/.julia julia --project=. ...
```

## Change Checklist

Before declaring a change complete:

1. Inspect the dirty worktree and preserve unrelated edits.
2. Update public docstrings and the relevant guide page.
3. Add focused behavioral or rejection tests.
4. Run the focused test, then `Pkg.test()`.
5. Build Documenter when public API or docs changed.
6. Run `git diff --check` and search for stale symbols.
7. Report intentionally unrun hardware tests explicitly.

Do not rewrite generated files, manifests, or unrelated user changes. Do not
weaken tests to accommodate a physics regression.
