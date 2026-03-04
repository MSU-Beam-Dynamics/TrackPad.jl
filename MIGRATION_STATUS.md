# Migration Status

**Last Updated:** 2026-02-13

## Current Snapshot
- JuTrack element migration is largely complete for the currently scoped element set.
- Element definitions were refactored into category files under `src/elements/`.
- JuTrack parity testing is consolidated in `test/verify_elements.jl` with `atol = 1e-15`.
- Wiggler tracking routine was migrated and included in parity checks.
- Linear optics and closed-orbit APIs are implemented and verified.

## Scope Decision
- TPSA migration from JuTrack is intentionally deferred.
- TPSA functionality is planned to be separated into a standalone package.
- Until that package boundary is defined, TrackPad migration scope excludes JuTrack TPSA APIs and TPSA-specific tracking paths.

## Phase Status

| Phase | Status | Notes |
|------|--------|-------|
| 1. Architecture & Type System | ✅ COMPLETE | Parametric element structs, `StaticArrays`, `Adapt` methods, module structure in place. |
| 2. Core Physics Migration | ✅ COMPLETE (for current scope) | Core integrators and `pass!` implementations are in place for migrated elements. |
| 2.6 Optics/Orbit APIs | ✅ COMPLETE | `gettune`, `getchrom`, `twissline`, `find_closed_orbit_4d`, `find_closed_orbit_6d` implemented and tested. |
| 3. Advanced Features | 🔄 PARTIAL | Time-dependent lattice/materialization implemented; CUDA/Enzyme extension modules still stubs; TPSA migration intentionally deferred. |
| 4. Documentation | ❌ NOT STARTED | No dedicated user-facing migration/usage docs yet. |

## Implemented Element Organization

Element definitions are now split by category:
- `src/elements/marker.jl`
- `src/elements/drift.jl`
- `src/elements/quad.jl`
- `src/elements/multipole.jl`
- `src/elements/bend.jl`
- `src/elements/rf.jl`
- `src/elements/wake.jl`
- `src/elements/solenoid.jl`
- `src/elements/corrector.jl`
- `src/elements/wiggler.jl`
- `src/elements/beambeam.jl`
- `src/elements/SCelements.jl`

Legacy grouped files (`linear.jl`, `multipoles.jl`, `bends.jl`, `auxiliary.jl`, `advanced.jl`) were removed.

## Implemented Tracking Coverage

`pass!` methods are implemented for:
- Core: `Marker`, `Drift`, `Quadrupole`, `Sextupole`, `Octupole`, `ThinMultipole`
- Bends: `SBend`, `ExactSBend`, plus convenience constructors (`RBend`, `ERBend`)
- RF: `RFCavity`, `CrabCavity`, `AccelCavity`, `LongitudinalRFMap`
- Utilities/magnets: `Solenoid`, `Corrector`, `HKicker`, `VKicker`, `Wiggler`
- Beam-beam/transform: `LorentzBoost`, `InvLorentzBoost`, `StrongThinGaussianBeam`, `StrongGaussianBeam`
- Wake: `LongitudinalRLCWake`, `LongitudinalWake`
- SC family: `DriftSC`, `QuadrupoleSC`, `SextupoleSC`, `OctupoleSC`, `SBendSC`, `RBendSC`, `LBend`, `SpaceCharge`, `Translation`, `YRotation`

## Verification Status

### Element Parity
- `test/verify_elements.jl` (renamed from `verify_canonical_elements.jl`):
  - `Element JuTrack Parity`: **99/99 passed**
  - `Element Gaps`: **2 broken (intentional skips)**
    - `LongitudinalWake` parity skipped (JuTrack lacks Float64 `pass!` support)
    - `StrongThinGaussianBeam` parity skipped (JuTrack lacks Float64 `pass!` support)

### Full Test Entry Point
- `test/runtests.jl` includes:
  - `verify_fodo.jl`, `verify_dba.jl`, `verify_bend.jl`, `verify_optics.jl`,
  - `verify_closed_orbit.jl`, `verify_time_dependence.jl`, `verify_enzyme_compat.jl`, `verify_elements.jl`
- In the current environment, all non-Enzyme suites run, but `verify_enzyme_compat.jl` errors if `Enzyme` is not installed.

## Extensions and AD/GPU Status
- `ext/TrackPadCUDAExt.jl`: present, currently scaffold/TODO.
- `ext/TrackPadEnzymeExt.jl`: present, currently scaffold/TODO.
- Enzyme compatibility test exists (`test/verify_enzyme_compat.jl`), but full extension-level AD rule implementation is still pending.

## Next Actions
1. Implement Enzyme extension rules (`ext/TrackPadEnzymeExt.jl`) and make AD coverage deterministic.
2. Implement CUDA/KernelAbstractions kernels (`ext/TrackPadCUDAExt.jl`).
3. Remove known optics/closed-orbit rough edges and tighten tolerances where feasible.
4. Add migration/user documentation and API usage guides.
5. Define TPSA package split and interface boundary before any TPSA migration work.
