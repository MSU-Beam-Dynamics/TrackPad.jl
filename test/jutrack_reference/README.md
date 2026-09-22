# Frozen JuTrack reference data

TrackPad started as a rewrite of [JuTrack.jl](https://github.com/MSU-Beam-Dynamics/JuTrack.jl),
and a large part of the test suite pins the two codes against each other:
element by element, for whole lines, and for the closed-orbit, optics and TPSA
interfaces TrackPad inherited. Those assertions are worth keeping, but running
them meant the test suite depended on JuTrack, which pulled an unregistered
package (and its own dependency tree) into every `Pkg.test("TrackPad")`.

So JuTrack's side of every comparison is recorded once, here:

- `generate.jl` — the only file in the repository that uses JuTrack. It builds
  each case, runs it, and writes the results to `test/jutrack_reference.jl`.
- `Project.toml` — the environment that script runs in.
- `../jutrack_reference.jl` — the generated fixture: one
  `JUTRACK_REFERENCE::Dict{String,Any}`, keyed by case name, included by
  `test/convention_helpers.jl` and read through `jutrack_reference(key)`.

The tests themselves still build their TrackPad side from scratch and still
assert the same tolerances; they just look JuTrack's answer up instead of
computing it.

## Regenerating

```
julia --project=test/jutrack_reference test/jutrack_reference/generate.jl
```

The environment needs JuTrack, which is unregistered, so instantiating it
fetches from GitHub (`[sources]` in its `Project.toml`). The script prints how
many entries it recorded and rewrites `test/jutrack_reference.jl` in full.

Run it when you add a parity case, or deliberately to re-baseline against a
newer JuTrack — and then **read the diff**. Every number that moves is a change
in what the two codes agree on, and re-baselining silently would turn a
regression into a new expectation. A regeneration that changes nothing you
touched should produce no diff at all: the emitter writes `repr(::Float64)`, so
values round-trip exactly.

## Keeping the two sides in sync

Each case has one key, used by both `generate.jl` and the test that consumes it:

| Key prefix | Consumer |
| --- | --- |
| `element/<name>` | `test/verify_elements.jl` |
| `wake/…`, `aperture/…` | `test/verify_elements.jl` |
| `line/…` | `verify_bend.jl`, `verify_exact_bend.jl`, `verify_fodo.jl`, `verify_dba.jl` |
| `closed_orbit/…` | `test/verify_closed_orbit.jl` |
| `optics/…` | `test/verify_optics.jl` |
| `tpsa/…` | `test/verify_tpsa.jl` |
| `enzyme/…` | `test/verify_enzyme_compat.jl` |
| `time/…` | `test/verify_time_dependence.jl` |

A case added to a test but not to the generator would otherwise fail only with
a confusing lookup error, so `jutrack_reference` names the regeneration command
in its error message, and `verify_elements.jl` asserts that the set of
`element/` keys matches the set of cases it runs exactly — an element parity
case added on one side and not the other fails loudly.

## Conventions baked into the data

`generate.jl` runs JuTrack with `use_exact_beti = 1` and
`use_exact_Hamiltonian = 1`, and all element parity cases are taken in the
ultrarelativistic limit (1e15 eV kinetic). JuTrack's finite-beta longitudinal
normalisation differs between element families — see the "Differences from
JuTrack" section of the conventions page — so these comparisons deliberately
live where the two codes are supposed to agree; TrackPad's own finite-beta
behaviour is tested directly, against analytic canonical maps, in
`verify_elements.jl`.

JuTrack's fifth coordinate has the opposite sign of TrackPad's, and its
one-turn maps come in a different variable order. `generate.jl` applies
`flip_longitudinal_coordinate` and `canonicalize_jutrack_map` (from
`test/convention_helpers.jl`) before recording, so the stored numbers are
already in TrackPad's convention and the tests compare them directly.
