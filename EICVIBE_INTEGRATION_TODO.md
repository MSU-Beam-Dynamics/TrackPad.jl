# TrackPad Integration with EICViBE

## Objective

Make TrackPad the always-available local reference engine for EICViBE.
TrackPad should provide fast basic optics and tracking without taking ownership
of EICViBE's machine topology, controls, sessions, or user interface. XSuite,
BMAD, MAD-X, and other engines remain optional higher-fidelity tools selected
when their additional capabilities are required.

The primary boundary is an occurrence-specific, PALS-compatible branch:

```text
EICViBE machine and branch model
        |
        | in-memory PALS-compatible dictionaries
        v
TrackPad.compile_branch
        |
        v
CompiledBranch(lattice, beam, source_index)
```

## P0: Minimum Local Engine

### Stable Interchange Contract

- [ ] Version the in-memory interchange schema accepted by `compile_branch`.
- [ ] Document coordinate order, units, energy semantics, charge convention,
      bend geometry, RF phase, and normalized multipole strengths.
- [ ] Return element names, source names, occurrence numbers, and TrackPad
      indices in a bridge-friendly metadata structure.
- [ ] Preserve EICViBE occurrence identity for repeated elements and diagnostics.
- [ ] Add strict validation that reports all unsupported elements and parameters
      before tracking starts.
- [ ] Add a machine-readable TrackPad capability report for engine selection.

### Optics Result

- [ ] Introduce a stable optics result containing:
  - [ ] `s`, beta, alpha, gamma, and phase arrays.
  - [ ] Closed-orbit arrays at requested reference points.
  - [ ] First-order dispersion and dispersion-prime arrays.
  - [ ] Tune, chromaticity, circumference, and reference-particle metadata.
  - [ ] Transfer matrices at requested reference points.
- [ ] Support periodic ring optics around the closed orbit.
- [ ] Support linac Twiss propagation from supplied entrance conditions.
- [ ] Define behavior for coupled lattices; reject them explicitly until a
      coupled optics implementation is available.
- [ ] Resolve and test the remaining MAD-X/TrackPad off-momentum `RBEND`
      chromaticity convention difference.

### Tracking and Diagnostics

- [ ] Add reference-point tracking for selected occurrence indices.
- [ ] Add batch reference-point tracking without retaining every element result.
- [ ] Return per-particle state and loss location, not only coordinate-limit loss.
- [ ] Implement rectangular and elliptical aperture checks on the CPU path.
- [ ] Add monitor reductions for centroid, RMS size, charge/transmission, and
      covariance at BPM/profile-monitor occurrences.
- [ ] Add chunked multi-turn tracking with a configurable diagnostic interval.
- [ ] Ensure diagnostic output uses bounded memory for long ring sessions.

### Live Parameter Updates

- [ ] Add an occurrence-aware element replacement API for immutable lattices.
- [ ] Map PALS parameter-group names to TrackPad element fields in one place.
- [ ] Support immediate updates to quadrupoles, sextupoles, correctors, bends,
      RF cavities, and solenoids.
- [ ] Support lazy per-turn parameter overlays without duplicating the lattice.
- [ ] Rebuild or invalidate cached `GPULattice` data after relevant updates.
- [ ] Return clear errors for updates that require unsupported physics.

## P1: Reliable Deployment

### Python/Julia Bridge Support

- [ ] Keep the public bridge limited to dictionaries, strings, scalars, and dense
      arrays that JuliaCall can convert without custom Python wrappers.
- [ ] Provide bridge entry points that do not expose internal parametric Julia
      element types.
- [ ] Add an API version and a startup health-check function.
- [ ] Add warm-up functions for lattice compilation, optics, CPU tracking, and
      optional GPU tracking.
- [ ] Measure cold-start, warm-start, branch compilation, Twiss, and tracking
      latency on macOS and Linux.
- [ ] Evaluate a PackageCompiler sysimage after the bridge API stabilizes.

### Time-Dependent and Ramping Operation

- [ ] Define the mapping from EICViBE beam-time and turn number to `TimeContext`.
- [ ] Support coordinated magnet, RF, and reference-energy ramp schedules.
- [ ] Define how `Beam` changes during acceleration rather than treating it as
      immutable for an entire ramp.
- [ ] Add ramp milestone and turn-index metadata to diagnostic results.
- [ ] Verify deterministic replay of a scheduled ramp.

### Reproducibility

- [ ] Add provenance output with TrackPad version, Julia version, backend,
      precision, integration settings, and enabled physics.
- [ ] Define deterministic random-seed handling for error ensembles.
- [ ] Add serialization for bridge inputs and diagnostic outputs used in
      regression reports.

## P2: Optional Accelerated Capabilities

### GPU

- [ ] Add reference-point and monitor reductions to GPU batch tracking.
- [ ] Preserve particle state and first-loss location on GPU.
- [ ] Add capability checks that distinguish CPU-only from GPU-supported
      elements and settings.
- [ ] Benchmark realistic EICViBE workloads, including many short interactive
      jobs, not only million-particle throughput.
- [ ] Add automated CUDA acceptance tests on the A100 server.

### Differentiation and Maps

- [ ] Complete Enzyme rules required by supported tracking and parameter-update
      paths.
- [ ] Expose stable batched Jacobian and Hessian-vector-product bridge functions.
- [ ] Define derivatives with respect to initial coordinates separately from
      derivatives with respect to lattice parameters.
- [ ] Keep PolySeries/TPSA optional and expose coefficients in plain dense arrays.
- [ ] Add response-matrix and sensitivity APIs useful to EICViBE correction and
      optimization workflows.

## Verification Matrix

- [ ] Small FODO ring: compile, Twiss, closed orbit, chromaticity, tracking.
- [ ] Representative linac: entrance Twiss propagation and monitor diagnostics.
- [ ] EIC RCS: MAD-X import reference, PALS/EICViBE branch compilation, optics,
      chromaticity, RF-off and RF-on behavior.
- [ ] Repeated names: occurrence mapping, monitor selection, and parameter update.
- [ ] Ramping ring: turn-dependent quadrupole, corrector, RF, and beam energy.
- [ ] CPU batch parity with scalar tracking.
- [ ] CUDA Float64 parity for every advertised GPU element.
- [ ] Unsupported physics: deterministic rejection without silent approximation.
- [ ] JuliaCall smoke test from a clean Python environment.

## Default-Engine Acceptance Gates

TrackPad is ready to be EICViBE's default local engine when:

- [ ] A clean EICViBE installation can initialize TrackPad without manual Julia
      package configuration.
- [ ] Ring and linac core workflows pass through the EICViBE engine interface.
- [ ] Required EICViBE `TwissData` fields are populated with documented units.
- [ ] BPM diagnostics and occurrence-specific live updates work correctly.
- [ ] Unsupported elements and requested physics are rejected before execution.
- [ ] Warm interactive latency is measured and acceptable on supported desktops.
- [ ] Cross-engine fixtures document expected TrackPad, MAD-X, XSuite, and BMAD
      differences rather than hiding convention mismatches.
- [ ] TrackPad's complete Julia suite and EICViBE's bridge suite pass in CI.

## TrackPad Non-Goals

- EICViBE retains ownership of multi-branch topology, forks, merging, control
  models, session lifecycle, asynchronous transport, and GUI behavior.
- TrackPad does not silently approximate unsupported collective effects,
  apertures, radiation, or element models.
- TrackPad does not need to replace XSuite or BMAD for high-fidelity studies.
- Full PALS expansion remains delegated to PALSJulia/pals-cpp; TrackPad consumes
  the selected expanded branch.
