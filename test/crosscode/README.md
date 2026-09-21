# Cross-code reference scripts

These scripts produce the MAD-X, PTC, Xsuite and pyAT numbers that
`test/verify_ring_optics.jl` ("Cross-code reference ring") pins TrackPad
against. They are not part of the Julia test suite; run them by hand (Python
3.11, `pip install cpymad xsuite nafflib accelerator-toolbox`) when a physics
change moves one of the pinned values, and update the test from their output
rather than editing the reference numbers.

| Script | Reference | What it gives |
|---|---|---|
| `run_madx.py` | MAD-X 5.09.03 `twiss, chrom` | Q, ξ, αc, βx, Dx, synch_1–5, Q(δ) fit |
| `run_ptc.py` | PTC (`exact=true`, model 1, method 6, 20 steps) | Q, ξ, αc, d²Q/dδ², anharmonicities |
| `run_xsuite.py` | Xsuite 0.114 (`Bend` rot-kick-rot, edges `linear` or `full`) | Q, ξ, ξ₂, αc, I1–I5, ∂Q/∂J |
| `run_at.py` | pyAT (`BndMPoleSymplectic4Pass`, `ExactSectorBendPass`) | Q, ξ, ξ₂, I1–I5 |

The bend-model correspondence (which TrackPad element matches which code) is
documented in `docs/src/conventions.md`, "Bend models and what other codes
compute". Conventions to keep in mind when reading the outputs: PTC's `DQ`
of order 2 and Xsuite's `ddqx` are d²Q/dδ² (twice TrackPad's `chrom2x`), PTC's
anharmonicities are dQ/d(2J), and for β0 < 1 MAD-X's summary `DQ1` and table
`DX` are PT derivatives (TrackPad `wrt=:deltae`) while its `deltap` scan is a
δP derivative.
