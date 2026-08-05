# TrackPad Integration-Side TODO

## Scope

TrackPad is a tracking engine for one ordered path and one reference beam. Its
runtime model contains only:

- `Beam`;
- `Lattice.elements`;
- an open-line or periodic-ring boundary.

Machine topology, branch graphs, route selection, source-occurrence identity,
control mappings, sessions, serialization policy, and consumer result types do
not belong in TrackPad.

```text
consumer or full parser
        |
        | one selected ordered path
        v
(Lattice, Beam) -> TrackPad tracking / maps / optics
```

## Completed Simplification

- [x] Remove consumer-named APIs and special cases from TrackPad core.
- [x] Keep native Julia tracking and optics independent of YAML or dictionaries.
- [x] Reduce normal file I/O to `read_pals`, `read_madx`, and `write_pals`, each
      producing or consuming native `Lattice` and `Beam` values.
- [x] Keep the built-in PALS reader limited to a documented tracking subset;
      complete language expansion belongs outside TrackPad.
- [x] Represent the only structural boundary as `Lattice.periodic`.
- [x] Reject ring-only tracking and optics operations for open lines.
- [x] Provide `transfer_map` for either an open line or periodic ring.
- [x] Publish coordinate, energy, multipole, bend, RF, loss, and boundary
      conventions.

Resolver stages are private implementation details; consumer adapters should
use the returned `Lattice` and `Beam` directly.

## P0: Tracking and Optics

- [ ] Add entrance-conditioned transport-line Twiss results with the same field
      conventions as periodic Twiss results.
- [ ] Define coupled-lattice behavior: return coupled optics explicitly or
      reject coupled requests before calculation.
- [ ] Add transfer matrices at selected element indices without retaining every
      intermediate matrix unless requested.
- [ ] Resolve and test the MAD-X/TrackPad off-momentum `RBEND` chromaticity
      convention difference.
- [ ] Add scalar and batch coordinates at selected element indices.
- [ ] Return stable per-particle loss state and first-loss element index.
- [ ] Implement rectangular and elliptical aperture checks in CPU tracking.
- [ ] Keep selected-point output and loss behavior consistent across scalar,
      threaded CPU, KernelAbstractions CPU, CUDA, and Metal paths.

## P0: Import Reliability

- [ ] Add representative fixtures for the built-in PALS subset and external
      parser handoff.
- [ ] Add representative open-line and periodic-ring PALS fixtures.
- [ ] Automate the existing EIC RCS MAD-X/TFS optics comparison.
- [ ] Aggregate unsupported imported elements and parameters into one useful
      error report rather than failing at only the first problem.
- [ ] Keep production MAD-X placement and language expansion delegated to a
      complete external parser.

## P1: Time Dependence and Acceleration

- [ ] Define schedules in terms of `TimeContext`, physical time, and turn.
- [ ] Support coordinated magnet, RF, and reference-particle schedules.
- [ ] Define reference energy, momentum, beta, and gamma evolution during
      acceleration instead of assuming one immutable `Beam` for a full ramp.
- [ ] Verify deterministic replay of scheduled tracking.

## P1: Performance and Reliability

- [ ] Benchmark cold compilation, warm optics, scalar tracking, batch tracking,
      and short interactive workloads on supported macOS and Linux systems.
- [ ] Add automated CUDA and Metal acceptance tests for every advertised GPU
      element and setting.
- [ ] Reject unsupported physics consistently rather than approximating it
      silently in production (`strict=true`).

## P2: Differentiation and High-Order Maps

- [ ] Complete Enzyme support required by generally supported tracking paths.
- [ ] Stabilize batched Jacobian, Hessian-vector-product, and Hessian APIs.
- [ ] Distinguish derivatives with respect to initial coordinates from
      derivatives with respect to lattice parameters.
- [ ] Keep PolySeries/TPSA optional and provide dense coefficient export.
- [ ] Add response-matrix and sensitivity primitives; correction and
      optimization workflows remain consumer responsibilities.

## Consumer Responsibilities

- Multi-branch topology, forks, merging, route lifecycle, and source identity.
- Engine registration, engine fallback, process startup, caching, and transport.
- Conversion to consumer-specific optics, tracking, diagnostic, and error types.
- Control-system mappings, live parameter orchestration, sessions, and UI state.
- Consumer-specific provenance, serialization, plotting, and cross-engine
  comparisons.
