# TrackPad Examples

These notebooks are for accelerator physicists who already know lattices,
Twiss functions, response matrices, wakes, and maps, but are new to TrackPad.
They introduce the Julia API directly and avoid repeating accelerator-physics
derivations.

## Start here

Run once from the repository root:

```bash
julia --project=examples -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

Start Jupyter using your existing IJulia installation and open `examples/`.
The first cell of each notebook activates the required environment.
Notebooks 02 through 09 reuse the lattice imported in notebook 01, so each
workflow operates on the same dipole-quadrupole-sextupole model.

| Notebook | Workflow | Extra requirement |
|---|---|---|
| `00_fodo_quickstart.ipynb` | Construct a ring, track, obtain Twiss functions | CairoMakie |
| `01_madx_import.ipynb` | Import MAD-X; compute and plot periodic optics | CairoMakie |
| `02_tpsa_map.ipynb` | Compute and inspect a second-order map | PolySeries |
| `03_ad_tracking.ipynb` | Batched Jacobians of tracking | Enzyme |
| `04_ad_tpsa.ipynb` | Differentiate a TPSA coefficient with respect to strength | Enzyme + PolySeries |
| `05_impedance_wake.ipynb` | Apply an RLC wake to a macroparticle bunch | CairoMakie |
| `06_lattice_matching.ipynb` | Match two quadrupole families to target tunes | none |
| `07_orbit_correction.ipynb` | Build and invert an orbit response matrix | none |
| `08_optics_correction.ipynb` | Correct beta beating with family response knobs | none |
| `09_gpu_and_parameter_sweeps.ipynb` | Batch tracking and aligned/Cartesian scans | CUDA for NVIDIA execution |

The optional notebooks activate and instantiate small environments under
`environments/` in their first cell. To prepare one manually instead:

```bash
julia --project=examples/environments/tpsa -e 'using Pkg; Pkg.instantiate()'
```

The GPU notebook also runs its batch and sweep sections on the CPU, so its data layout
can be learned without NVIDIA hardware.

All notebooks are committed without saved output. Paths are resolved relative
to this directory, so either the repository root or `examples/` is a valid
Jupyter working directory.

## Detailed derivation

`advanced/fodo_cell_derivations.ipynb` retains the original, equation-by-equation
FODO example based on the MSU Accelerator Physics notes. It is a validation
study rather than the recommended introduction to TrackPad.
