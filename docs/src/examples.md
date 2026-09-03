# Examples

The repository's [numbered notebook series](https://github.com/MSU-Beam-Dynamics/TrackPad.jl/tree/main/examples)
is a concise introduction for accelerator physicists. It assumes the physics
terminology and concentrates on TrackPad's types, coordinate conventions, and
execution APIs.

The sequence starts with direct construction and MAD-X import, then covers TPSA
maps, Enzyme derivatives, collective longitudinal wakes, matching, orbit and
optics response correction, and packed CPU/GPU parameter scans. Every notebook
is independent, has cleared output, and identifies optional package or hardware
requirements before using them.

Prepare the base plotting environment with:

```bash
julia --project=examples -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

Optional environments for Enzyme, PolySeries, and CUDA are under
`examples/environments/`. The CUDA notebook performs its first batch and
parameter sweep with KernelAbstractions' CPU backend, then shows the two lines
that move the same packed objects and arrays to an NVIDIA device.

The former long FODO derivation remains available as
`examples/advanced/fodo_cell_derivations.ipynb`; it is intended for validation
and comparison with analytical notes, not as the first tutorial.
