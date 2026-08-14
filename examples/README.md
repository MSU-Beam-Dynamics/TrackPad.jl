# TrackPad Examples

Each example keeps its numerical calculation in an executable Julia file using
only TrackPad and its direct dependencies. Notebook-only visualization packages
are declared in `examples/Project.toml`. Calculation modules expose a
`run_example` function, print a compact report when run directly, and are also
exercised by the package test suite.

## FODO Cell With Dipoles

[`fodo_cell_with_dipoles.ipynb`](fodo_cell_with_dipoles.ipynb) is the primary
tutorial. It combines explanatory notes, executable Julia cells, result tables,
and plots in one Jupyter notebook. It builds one periodic FODO cell from
finite-length quadrupoles and sector dipoles and reproduces the calculations in the
[MSU Accelerator Physics notes](https://msu-beam-dynamics.github.io/AP_enotes/transverse_BD/examples/FODO_cell.html):

- transfer matrices at three starting points;
- stability, phase advance, and periodic Twiss parameters;
- periodic horizontal dispersion;
- natural chromaticity;
- the dispersion invariant `H` and an endpoint radiation-integral estimate;
- the phase advance that minimizes the normalized endpoint `H` estimate.

The notes' Equations 9.4 and 9.9 are labeled “after QF,” but their matrix is the
after-QD cyclic map under the focusing signs used in Equation 9.1. The example
tests the published equation at the matching after-QD boundary and also reports
the physically after-QF map. It likewise reproduces the notes' 100-point
phase-scan minimum before calculating the continuous analytic minimum.

Start Jupyter from the TrackPad repository root or the `examples` directory and
open the notebook. Its first cell activates and instantiates the isolated
examples environment. To prepare that environment manually:

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

The calculation module can also be run directly:

```bash
julia --project=. examples/fodo_cell_with_dipoles.jl
```

The analytical equations use the thin-lens and small-angle approximations from
the notes. The TrackPad lattice deliberately uses finite magnets, so the example
reports and bounds the resulting approximation errors instead of expecting the
two models to be bit-for-bit identical.

The notebook uses CairoMakie to plot the periodic beta functions and dispersion
through the finite cell, as well as the analytical phase-advance scans.
CairoMakie remains isolated from TrackPad's core dependencies.
