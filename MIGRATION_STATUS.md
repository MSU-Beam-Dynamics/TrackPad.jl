# Migration Status

**Last Updated:** 2026-08-05

## Current Snapshot

- JuTrack element migration is complete for the currently scoped CPU element set.
- Element definitions are organized by category under `src/elements/`.
- CPU element parity is tested against JuTrack with a target tolerance of `1e-15`.
- Wiggler tracking, optics/map APIs, closed-orbit APIs, time-dependent parameters,
  lattice I/O, and optional TPSA maps are implemented.
- TPSA support is separated from the TrackPad core through the optional
  `PolySeries` extension rather than by migrating JuTrack's internal TPSA types.
- KernelAbstractions-based batch tracking, parameter sweeps, Metal adaptation,
  and CUDA adaptation are implemented for the explicitly supported element set.
- The complete test suite passes through `Pkg.test()` in one invocation.
- GPU tracking remains scoped to a subset of elements, with unsupported physics
  rejected during lattice encoding.

## Scope Decisions

### TPSA

- JuTrack's internal TPSA implementation is not being migrated into TrackPad.
- TrackPad exposes `tpsa_map` from the core API, with its implementation provided
  by `ext/TrackPadPolySeriesExt.jl` when `PolySeries.jl` is loaded.
- First- and second-order maps are compared against JuTrack CTPS results in
  `test/verify_tpsa.jl`.
- Multi-TPSA GPU tracking is not implemented.

### Element Migration

- The current migration scope covers canonical CPU tracking elements and their
  public tracking behavior.
- Non-canonical JuTrack element variants remain intentionally excluded.
- `LongitudinalWake` and `StrongThinGaussianBeam` cannot currently be compared
  with Float64 JuTrack tracking because JuTrack lacks the required `pass!`
  methods. `LongitudinalRLCWake` tracking parity uses `Rshunt = 0` (zero kick)
  to validate plumbing, while its nonzero Green function is compared directly;
  TrackPad's wake semantics are collective
  (cloud-in-cell histogram of `r[5]` convolved with the wake Green function,
  with bin-edge-interpolated kicks) and are covered by dedicated convolution
  tests instead. Bin-edge reconstruction and per-particle interpolation follow
  JuTrack's scheme; TrackPad uses a translation-invariant padded grid and
  linear cloud-in-cell deposition instead of JuTrack's zero-centered grid and
  quadratic neighbor weights. `physical_wake_scale(beam, bunch_charge,
  nmacro)` provides the physically normalized kick scale for TrackPad's
  `delta_E` convention.

- A 2026-08 audit against JuTrack corrected five divergences: the
  `Translation` longitudinal shift is now drift-consistent (intentionally
  diverging from JuTrack's uncompensated term), rectangular/elliptical
  apertures are enforced in multi-particle CPU tracking, bend constructors
  auto-raise the multipole kick order from nonzero `polynom_b` entries, the
  Forest (13.29) entrance/exit fringe correction is applied by thick
  multipoles and bends in scalar and PolySeries tracking, and vertical wiggler
  harmonics were verified bit-exact against JuTrack on physical configurations;
  malformed blocks and zero denominator wave vectors are rejected. Known remaining
  JuTrack-only capabilities: beam-current-driven `SPACECHARGE` (TrackPad's
  single-particle `SpaceCharge` stays a no-op), radiation (`rad_on!`,
  `ElementRadiation/WigglerRadiation/ElossRadiation`), matrix-formalism
  `QUAD`, `CRABCAVITY_K2`, `easyCRABCAVITY`, `Beam_Gauss`, `dynamic_aperture`,
  RDTs, and reference-point tracking overloads. The LorentzBoost pair remains
  a documented symplectic divergence (see conventions.md).

## Phase Status

| Phase | Status | Notes |
|------|--------|-------|
| 1. Architecture and Type System | COMPLETE | Parametric element structs, `StaticArrays`, `Adapt`, optional extensions, and category-based source layout are in place. |
| 2. Core CPU Physics Migration | COMPLETE FOR CURRENT SCOPE | CPU `pass!` implementations and strict JuTrack element parity are in place for the scoped elements. |
| 2.6 Optics and Orbit APIs | IMPLEMENTED AND VERIFIED | JuTrack-compatible finite-difference separation and exact inverse-velocity conventions are covered by parity tests. |
| 3. TPSA Maps | IMPLEMENTED AS OPTIONAL EXTENSION | `PolySeries` supplies TPSA types; standalone first- and second-order parity tests pass. |
| 4. Time Dependence | IMPLEMENTED | Time-dependent parameters, element materialization, and verification are present. |
| 5. Lattice I/O | IMPLEMENTED FOR DOCUMENTED SUBSET | The exported API is reduced to `read_pals`, `read_madx`, and `write_pals`, operating on native `(Lattice, Beam)` values. Full PALS and production MAD-X language expansion require external parsers. |
| 6. Enzyme AD | PARTIAL | Batched Jacobians use Enzyme forward mode; second derivatives use finite differences of Enzyme Jacobians. Custom extension-level AD rules remain TODO. |
| 7. GPU Tracking | IMPLEMENTED FOR SUPPORTED ELEMENTS | CPU backend parity is in the package suite; CUDA Float32/Float64 tracking is verified on an NVIDIA A100. |
| 8. Documentation | COMPLETE FOR CURRENT SCOPE | User, element, conventions, I/O, TPSA, GPU, API, and coding-agent documentation is included. |

## Element Organization

Element definitions are split by category:

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

Legacy grouped files (`linear.jl`, `multipoles.jl`, `bends.jl`,
`auxiliary.jl`, and `advanced.jl`) were removed.

## CPU Tracking Coverage

`pass!` methods are implemented for:

- Core: `Marker`, `Drift`, `Quadrupole`, `Sextupole`, `Octupole`,
  `ThinMultipole`
- Bends: `SBend`, `ExactSBend`, plus `RBend` and `ERBend` constructors
- RF: `RFCavity`, `CrabCavity`, `AccelCavity`, `LongitudinalRFMap`
- Magnets/utilities: `Solenoid`, `Corrector`, `HKicker`, `VKicker`, `Wiggler`
- Beam-beam/transforms: `LorentzBoost`, `InvLorentzBoost`,
  `StrongThinGaussianBeam`, `StrongGaussianBeam`
- Wakes: `LongitudinalRLCWake`, `LongitudinalWake`
- Space charge: `DriftSC`, `QuadrupoleSC`, `SextupoleSC`, `OctupoleSC`,
  `SBendSC`, `RBendSC`, `LBend`, `SpaceCharge`, `Translation`, `YRotation`

## Verification Status

The latest verification on 2026-08-05 used Julia 1.12.6 through the package
test target on local Apple hardware.
No numerical or test-integration failures were observed.

### Verified Results

- The latest complete `Pkg.test()` invocation reported **637 passed, 0 failed,
  2 intentional broken tests**.
- The result includes CPU physics, ExactSBend, optics, closed orbit, time
  dependence, Enzyme compatibility, element parity, TPSA verification, and
  PALS/MAD-X interchange coverage.
- CPU `GPULattice` batch smoke test: maximum difference from ordinary tracking
  was `0.0`; parameter-sweep output was finite.

### Element Parity

- `test/verify_elements.jl` uses `PARITY_ATOL = 1e-15`.
- The migrated element comparisons pass at that threshold.
- Intentional gaps:
  - `LongitudinalWake`: JuTrack lacks Float64 `pass!` support; the collective
    histogram/convolution semantics are covered by dedicated TrackPad tests.
  - `StrongThinGaussianBeam`: JuTrack lacks Float64 `pass!` support.

### Optics and Map Parity

All 50 current optics checks pass. The one-turn matrix, tune, and chromaticity
tests enforce exact equality when JuTrack is configured to use the same exact
inverse-velocity convention as TrackPad. Measured maximum differences for the
verification FODO cell are:

| Quantity | Maximum difference from JuTrack | Current test tolerance |
|----------|---------------------------------|------------------------|
| One-turn matrix (`fastfindm66`) | `0.0` | Exact equality |
| Tune (`gettune`) | `0.0` | Exact equality |
| Chromaticity (`getchrom`) | `0.0` | Exact equality |
| Periodic Twiss parameters | `8.881784197001252e-15` | `1e-13` |

The former matrix discrepancy was caused by two convention mismatches:

- JuTrack's `3e-8` map scaling is the full separation between the positive and
  negative perturbations; TrackPad had treated it as the displacement on each
  side.
- JuTrack defaults `use_exact_beti` to `0`, while TrackPad consistently uses the
  exact beam inverse velocity. Parity tests set JuTrack's flag to `1` and restore
  it afterward.

An additional few-`1e-9` longitudinal map residual came from floating-point
parenthesization in the drift update. TrackPad now matches JuTrack's in-place
evaluation order, producing identical finite-difference endpoints.

### Test Environment and Isolation

- `JuTrack` and `PolySeries` are declared as test-only dependencies and as
  project-relative sources in `Project.toml`; Enzyme, Serialization, and Test
  remain declared in the test target.
- `test/runtests.jl` evaluates each verification file in a separate module, so
  test globals and constants cannot collide across files.
- `test/verify_exact_bend.jl` is included in the main physics verification
  group.
- The supported complete-suite command is
  `julia --project=. -e 'using Pkg; Pkg.test()'`.

## TPSA Status

- Core interface: `src/tpsa.jl`
- Optional implementation: `ext/TrackPadPolySeriesExt.jl`
- Verification: `test/verify_tpsa.jl`
- Verified capabilities:
  - First-order map versus finite-difference `one_turn_map`
  - Constant term at zero and nonzero expansion points
  - Second-order nonlinear coefficients
  - First- and second-order coefficient parity with JuTrack CTPS
  - Float64 tracking regression after TPSA use

TPSA is now separated through a package extension, so the earlier "TPSA
deferred" status no longer applies.

## Enzyme Status

- `test/verify_enzyme_compat.jl` passes in the dependency-complete audit
  environment.
- `ext/TrackPadEnzymeExt.jl` remains a scaffold with custom AD rules marked TODO.
- Current evidence verifies selected AD-compatible tracking paths, not complete
  differentiation coverage for every element and API.

## GPU Status

Implemented in `src/gpu.jl`:

- Flat-array `GPULattice` encoding
- `ParamSweepLattice`
- KernelAbstractions batch tracking kernels
- CPU backend execution
- Multi-particle batch tracking
- Multi-turn `batch_ringpass!` tracking
- Parameter-sweep tracking
- Metal and CUDA extension scaffolding/adaptation paths

Verified on 2026-08-05 with CUDA.jl 5.11.3, Julia 1.12.6, and one NVIDIA
A100-SXM4-40GB (`sm_80`):

- CUDA tracking suite: **33/33 passed**
- CUDA Enzyme derivative suite: **4/4 passed**
- Float32 one-turn maximum difference from the CPU backend: `7.932431e-8`
- Float32 three-turn maximum difference from the CPU backend: `2.379893e-7`
- Float64 one-turn maximum difference from the CPU backend:
  `9.387412044535234e-17`
- Float64 three-turn maximum difference from the CPU backend:
  `2.434304927773079e-16`
- Float32 and Float64 aligned and Cartesian parameter-sweep parity passed.
- Float64 Enzyme Jacobian, Hessian-vector product, and full-Hessian parity
  passed with maximum differences `2.2204669603761984e-16`,
  `2.775798172452394e-11`, and `3.3306690738754696e-11`, respectively.
- CUDA tests disable scalar indexing with `CUDA.allowscalar(false)`.
- A 1,000,000-particle, 100-element Float64 benchmark measured `0.43127 s` on
  the 64-thread KernelAbstractions CPU backend, `0.007977 s` on one A100, and
  `0.003461 s` across four host-partitioned A100s. This corresponds to `54.06x`
  one-GPU acceleration over the flattened CPU backend and `2.31x` four-GPU
  strong scaling over one GPU.

Current limitations:

- GPU element encoding supports only a subset of CPU elements.
- `ExactSBend`, `LBend`, space-charge elements, collective elements, and other
  unsupported types raise `ArgumentError` rather than using approximate bend or
  drift behavior.
- Nonzero misalignments, apertures, radiation, multipole fringe settings, and
  kick-angle corrections are rejected because the GPU kernels do not implement
  them.
- `test/verify_gpu.jl` runs backend-neutral parity and rejection checks in every
  package test invocation. `test/verify_cuda.jl` is a standalone hardware suite
  and is not run on hosts without CUDA.
- Automated Metal hardware testing is not yet part of the package suite.
- Multi-TPSA GPU tracking is not implemented.

## Lattice I/O Status

- The public model is one ordered `Lattice`, one `Beam`, and an open-line or
  periodic-ring boundary. Topology and occurrence identity are not retained.
- `read_pals`: built-in selected-branch expansion is implemented for the
  documented PALS subset and returns `(Lattice, Beam)`.
- `write_pals`: emits a canonical BeginningEle/BeamLine/Lattice/use document
  and takes periodicity from the lattice by default.
- `read_madx`: compact LINE/simple-sequence parsing plus BEAM species and
  reference momentum/energy conversion are implemented. `USE, PERIOD` marks a
  ring; other paths default to open unless overridden.
- Resolver stages remain private implementation details; the public PALS reader
  returns only `(Lattice, Beam)`.
- The EIC RCS `RCS9GeV40CellV1.madx` LINE lattice imports as 2,850 elements.
  At 80 integration steps its fractional tunes agree with the supplied MAD-X
  TWISS table within `7e-9`, and all four beta/alpha arrays within `3e-8`.
- Its centered TrackPad chromaticity converges to `(0.82009799, 0.99729708)`
  versus MAD-X `(0.78937414, 1.09294806)`. The discrepancy is not caused by
  finite-difference step, integration resolution, K2 normalization, or the
  off-momentum closed orbit; the remaining off-momentum map convention needs
  investigation.
- User documentation exists in `docs/src/io.md`.
- `test/verify_io.jl` covers canonical PALS resolution, inline branches,
  inheritance/repetition, strict errors, round trips, duplicate names,
  reference beams, MAD-X beam conversion, boundary inference, skew multipole
  mapping, and the reduced public API boundary.
- Full PALS expressions, controllers, includes, forks, and bookkeeping require
  preprocessing by an external parser or consumer adapter.
- Production MAD-X placement semantics remain delegated to an external full
  parser such as MAD-X/cpymad.

## Next Actions

The prioritized TrackPad-side integration backlog is maintained in
`EICVIBE_INTEGRATION_TODO.md`.

1. Complete transport-line optics and selected-point diagnostics in
   `EICVIBE_INTEGRATION_TODO.md`.
2. Add representative PALS subset fixtures and document external-parser
   handoff requirements.
3. Implement and verify Enzyme extension rules where generic differentiation is
   insufficient.
4. Add automated Metal hardware verification and expand GPU element coverage
   only where exact CPU-equivalent kernels are implemented.
5. Build the documentation in a clean environment and commit the completed GPU,
   Metal, scripts, and documentation work.
