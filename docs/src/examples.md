# Examples

The repository's [`examples`](https://github.com/MSU-Beam-Dynamics/TrackPad.jl/tree/main/examples)
directory contains complete, executable calculations. Each example uses the
public TrackPad API and includes numerical checks, so it serves both as a human
tutorial and as an executable reference for an AI coding agent.

## FODO cell with dipoles

Open
[`examples/fodo_cell_with_dipoles.ipynb`](https://github.com/MSU-Beam-Dynamics/TrackPad.jl/blob/main/examples/fodo_cell_with_dipoles.ipynb)
in Jupyter for the combined notes, Julia code, result tables, and plots. The
notebook activates `examples/Project.toml`, where CairoMakie is isolated from
TrackPad's core dependencies. Its figures include the periodic ``\beta_x``,
``\beta_y``, and ``D_x`` functions through the finite cell, in addition to the
analytical phase-advance scans. The shared calculation module can also be run
from the repository root:

```bash
julia --project=. examples/fodo_cell_with_dipoles.jl
```

The example follows the
[MSU Accelerator Physics FODO-cell notes](https://msu-beam-dynamics.github.io/AP_enotes/transverse_BD/examples/FODO_cell.html)
from the transfer matrix through the periodic Twiss and dispersion solutions,
natural chromaticity, dispersion invariant ``\mathcal{H}``, radiation-integral
estimate, and optimum phase advance.

The lattice itself is not a thin-lens toy: it contains finite-length
[`Quadrupole`](@ref) and [`SBend`](@ref) elements. The analytical reference uses
the thin-lens and small-angle assumptions of the notes. Consequently, the
example tests exact matrix identities separately from explicitly bounded
finite-magnet approximation errors.

Two details of the source calculation are made explicit. Equations 9.4 and 9.9
are labeled “after QF,” although their matrix corresponds to the after-QD
cyclic boundary under the focusing signs in Equation 9.1. The example checks
that published matrix at the matching boundary and also computes the true
after-QF map. The published optimum phase advance is reproduced from its
100-point plotting scan, then compared with the continuous analytic minimum.

The sixth TrackPad coordinate is
``\delta_E = (E-E_0)/(P_0c)``, whereas the notes use
``\delta_P = (P-P_0)/P_0``. The example applies the first-order conversion
``\delta_E = \beta_0\delta_P`` before comparing dispersion and chromaticity.
