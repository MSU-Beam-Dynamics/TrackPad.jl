# Contributing to TrackPad.jl

Thank you for considering a contribution. This page covers the mechanics;
`AGENTS.md` describes the code architecture and the rules every change must
respect, and applies to people as much as to automated tools.

## Setting up

```bash
git clone https://github.com/MSU-Beam-Dynamics/TrackPad.jl
cd TrackPad.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

TrackPad requires Julia 1.12 or newer. Its one unregistered dependency is
PolySeries.jl, the TPSA backend behind the default optics method; its location
is given in `Project.toml`'s `[sources]` table and fetched by Pkg automatically
when this repository is the active project.

## Running the tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Always go through `Pkg.test()` — it activates the test dependencies that
`include("test/runtests.jl")` would not see. Individual files under `test/`
are self-contained and can be run inside that environment with
`julia --project=. test/verify_optics.jl` once the test dependencies are
available. GPU hardware tests live in `test/cuda/` with their own environment.

The suite is the specification. A physics change is not done until a test
pins the new behaviour against an independent reference — an analytic result,
a converged high-step-count map, a TPSA derivative, or a cross-code
comparison with the convention differences applied explicitly.
`test/verify_ring_optics.jl` holds a reference ring whose tunes, Twiss,
dispersion, momentum compaction, chromaticities, radiation integrals and
amplitude detuning are pinned to MAD-X, PTC, Xsuite and pyAT; the Python
scripts that produce those numbers are in `test/crosscode/`. If a change moves
one of them, re-run the external codes rather than editing the reference.

## Building the documentation

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The build is strict: every exported symbol needs a docstring and a place in a
page (`checkdocs = :exports`), and cross-reference or doctest failures are
errors. Open `docs/build/index.html` to review.

## What a good change looks like

- Kernels stay allocation-free and generic in the coordinate type: build
  scalars from the element's `T`, never from the coordinate `S`, so TPSA and
  AD coordinates keep working. The agent guide's "Writing Generic Kernels"
  section lists the rules.
- Physics conventions are documented in `docs/src/conventions.md`; if a change
  alters one, update that page in the same pull request and add an entry to
  `CHANGELOG.md`.
- Defaults are physics decisions. Changing one (integration steps, reference
  orbit, momentum convention) needs a measured accuracy-versus-cost argument
  in the pull request and a note in the changelog and docs.
- Cross-code parity tests (JuTrack) pin every convention explicitly —
  `num_int_steps`, `closed_orbit`, `wrt` — so they keep comparing the same
  model when TrackPad's defaults move. JuTrack's side of every one of them is
  frozen in `test/jutrack_reference.jl`, so the suite does not depend on
  JuTrack; adding a parity case means regenerating that file and reading the
  diff. See `test/jutrack_reference/README.md`.

## Reporting problems

Open an issue with the lattice (or a minimal one), the beam, the call, the
result you got and the result you expected, and — for cross-code comparisons —
the settings listed under "Required Assumptions to State" in the agent guide.
