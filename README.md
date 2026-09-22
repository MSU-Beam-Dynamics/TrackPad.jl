# TrackPad.jl

[![CI](https://github.com/MSU-Beam-Dynamics/TrackPad.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/MSU-Beam-Dynamics/TrackPad.jl/actions/workflows/CI.yml)
[![Docs stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MSU-Beam-Dynamics.github.io/TrackPad.jl/stable/)
[![Docs dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MSU-Beam-Dynamics.github.io/TrackPad.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Symplectic 6-D particle tracking and linear optics for accelerator lattices,
in Julia.

Each element's physics is written once and reused everywhere: the same
allocation-free `pass!` code tracks single particles or CPU/GPU batches, is
differentiated by Enzyme, and expands into truncated-power-series maps with
PolySeries. A lattice built once can be tracked, differentiated and expanded
without any of it being rewritten — and there is no separate AD or TPSA
implementation of the physics to keep in sync.

## Installation

TrackPad is not yet in the General registry. Install from GitHub (Julia ≥ 1.12):

```julia
using Pkg
Pkg.add(url = "https://github.com/MSU-Beam-Dynamics/TrackPad.jl")
```

TrackPad depends on [PolySeries.jl](https://github.com/MSU-Beam-Dynamics/PolySeries.jl)
(the TPSA backend of the default optics method), also unregistered for now —
add it first:

```julia
Pkg.add(url = "https://github.com/MSU-Beam-Dynamics/PolySeries.jl")
```

Optional capabilities switch on by loading a companion package next to
TrackPad — no configuration needed:

| capability | load | install |
|---|---|---|
| batched derivatives | `using Enzyme` | `Pkg.add("Enzyme")` |
| NVIDIA GPU (`Float32`/`Float64`) | `using CUDA` | `Pkg.add("CUDA")` |
| Apple GPU (`Float32`) | `using Metal` | `Pkg.add("Metal")` |
| lattice plots | `using CairoMakie` | `Pkg.add("CairoMakie")` |

## Quick start

```julia
using TrackPad, StaticArrays

beam = Beam(3.0e9)                     # total energy in eV; electron by default

ring = Lattice(AbstractElement[
    Quadrupole(0.5,  0.9; name=:QF), Drift(0.6),
    SBend(1.2, 2π/40; name=:B),      Drift(0.6),
    Sextupole(0.2, 3.0; name=:SF),   Drift(0.6),
    Quadrupole(0.5, -0.9; name=:QD), Drift(0.6),
    SBend(1.2, 2π/40; name=:B),      Drift(0.6),
    Sextupole(0.2, -3.0; name=:SD),  Drift(0.6),
]; periodic=true)

# one particle, one turn / many turns
r0 = @SVector [1e-3, 0.0, 0.5e-3, 0.0, 0.0, 0.0]
r1 = track(ring, r0, beam)
rN = track(ring, r0, beam; nturns=1000)

# linear optics
qx, qy  = gettune(ring, beam)
ξx, ξy  = getchrom(ring, beam)         # dQ/dδP about the off-momentum closed orbit
tw      = twiss(ring, beam)            # β, α, μ, dispersion, αc, ξ at every element boundary
tws     = twiss(ring, beam; slices=10) # the same, 10 points inside every element, for plots

# an ensemble: 10 000 particles, 500 turns, on all Julia threads
coords = matched_gaussian(10_000, optics4DUC(tw.betax[1], tw.alphax[1],
                                              tw.betay[1], tw.alphay[1]);
                          emitx=1e-9, emity=1e-10)
track!(coords, ring, beam; nturns=500, threaded=true)
```

Coordinates are `(x, px, y, py, z, δE)` with `z = s/β₀ − ct` (positive for an
early particle) and `δE = (E−E₀)/(P₀c)`. The *interface* — every `dp`
argument, chromaticity, dispersion — speaks the relative momentum
`δP = (P−P₀)/P₀` that MAD-X, elegant and AT use; only the tracked state stores
`δE`. Read the [conventions page](https://MSU-Beam-Dynamics.github.io/TrackPad.jl/stable/conventions/)
before comparing with another code.

## What is in the box

| | |
|---|---|
| **Tracking** | `track`, `track!` (one particle or an `N×6` matrix, one pass or many turns, serial or threaded, and on a packed `GPULattice`); `linepass`/`linepass!` for JuTrack-compatible single passes; `ParamSweepLattice` for lazily generated parameter scans |
| **Elements** | drifts, quadrupoles, sextupoles, octupoles, thin multipoles, sector/rectangular/exact bends with AT-style or hard-edge fringes, RF/crab/accelerating cavities, solenoids, wigglers, correctors, patches, beam–beam (Bassetti–Erskine, synchro-beam slices), longitudinal wakes, space charge; turn-by-turn or real-time parameters via `timed` |
| **Optics** | `gettune`, `getchrom`, `twiss` (rings and lines: Twiss, dispersion, αc, ξ; optional ξ₂, I1–I5, ∂Q/∂J; from TPSA maps by default or from tracking; `periodic_twiss`/`transport_twiss`), `one_turn_map`, `find_closed_orbit_4d/6d`, plus JuTrack-style `findm66`, `twissring` |
| **Beams** | `Beam`; `matched_gaussian` / `gaussian_distribution` with exact finite-sample moments; `beam_covariance`, `projected_emittances`, `eigenemittances` |
| **Maps & derivatives** | `tpsa_map`; `batch_jacobian!`, `batch_hessian_vector_product!`, `batch_hessian!` (Enzyme) |
| **I/O** | `read_pals`, `write_pals`, `read_madx` |

## Documentation

The [manual](https://MSU-Beam-Dynamics.github.io/TrackPad.jl/stable/) has a
getting-started guide, the normative physics and data conventions, the element
catalogue, file I/O, GPU, performance and API reference pages. The
[`examples/`](examples/) directory holds a numbered notebook series from a FODO
quickstart through MAD-X import, TPSA, AD, wakes, matching, orbit and optics
correction, and GPU parameter sweeps.

## Development

```bash
git clone https://github.com/MSU-Beam-Dynamics/TrackPad.jl
cd TrackPad.jl
julia --project=. -e 'using Pkg; Pkg.test()'        # full suite
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl                    # build the manual
```

Run tests through `Pkg.test()` rather than `include("test/runtests.jl")`: the
test target activates the Enzyme test dependency. CUDA hardware tests live in
the separate `test/cuda` environment. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

If TrackPad contributes to published work, please cite it; the metadata is in
[`CITATION.cff`](CITATION.cff) (GitHub's *Cite this repository* button renders
it as BibTeX or APA).

## License

MIT — see [LICENSE](LICENSE).
