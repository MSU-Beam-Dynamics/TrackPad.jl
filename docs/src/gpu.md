```@meta
CurrentModule = TrackPad
```

# GPU Acceleration

TrackPad.jl supports GPU-accelerated particle tracking via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
which provides a single backend-agnostic kernel language that compiles to:

| Backend | Package | Float width |
|---------|---------|-------------|
| Apple Metal | `Metal.jl` | **Float32 only** (Apple GPU constraint) |
| NVIDIA CUDA | `CUDA.jl` | Float32 or Float64 |
| CPU (multi-core) | *(built-in)* | Float32 or Float64 |

Three use cases are supported:

1. **Multi-particle batch tracking** — N particles tracked in parallel through the same lattice.
2. **Parameter sweep** — one (or N) particle(s) tracked through N different lattice configurations simultaneously.
3. **CPU multi-threaded fallback** — no GPU required; uses `Threads.@threads` over particles.

---

## Setup

GPU support is provided by extension packages.  Load the backend-specific
package after TrackPad:

```julia
# Apple Metal
using TrackPad, Metal

# NVIDIA CUDA
using TrackPad, CUDA
```

No extra configuration is needed.  The extensions are activated automatically.

---

## Core Types

### `GPULattice`

[`GPULattice`](@ref) is a flat-array encoding of a [`Lattice`](@ref) that can
be executed by a GPU kernel.  All element parameters are packed into two
matrices:

- `fparams :: Matrix{T}` — float parameters, shape `(18, n_elems)`
- `iparams :: Matrix{Int32}` — integer parameters (e.g. integration steps), shape `(2, n_elems)`
- `elem_types :: Vector{Int32}` — element type codes

The `T` type parameter controls precision.  Use `Float32` for Metal or
`Float64` for CUDA/CPU.

### `ParamSweepLattice`

[`ParamSweepLattice`](@ref) extends `GPULattice` with a third dimension in
`fparams` — one set of parameters per configuration.  All N configurations are
processed in parallel, one GPU thread per configuration.

---

## Use Case 1: Multi-Particle Batch Tracking

Track N particles through the same lattice, one GPU thread per particle.

### Metal (Apple GPU, Float32)

```julia
using TrackPad, Metal, StaticArrays

beam = Beam(18e9)
lat  = read_madx("ring.madx")   # or build manually

# Build GPULattice and upload to Metal device
gl = gpu_adapt(lat, beam, MetalBackend())

# Allocate N×6 coordinate matrix on the Metal device
N      = 1_000_000
coords = MtlArray(zeros(Float32, N, 6))
# ... fill coords with initial beam distribution ...

# Track one turn (in-place)
batch_linepass!(coords, gl)

# Multi-turn ring tracking
batch_ringpass!(coords, gl, 1000)

# Copy results back to CPU
result = Array(coords)
lost   = vec(any(isnan, result, dims = 2))   # Bool vector: lost particles
println(count(lost), " / $N particles lost")
```

### CUDA (NVIDIA GPU, Float64)

```julia
using TrackPad, CUDA

gl     = gpu_adapt(lat, beam, CUDABackend(); dtype = Float64)
coords = CuArray(zeros(Float64, N, 6))
# ... fill ...
batch_linepass!(coords, gl)
batch_ringpass!(coords, gl, 1000)
```

### CPU Backend

When no GPU is available, `batch_linepass!` runs on the CPU via
KernelAbstractions' built-in CPU backend:

```julia
using TrackPad

gl     = GPULattice(lat, beam; dtype = Float64)
coords = zeros(Float64, N, 6)
batch_linepass!(coords, gl)
```

---

## Use Case 2: Parameter Sweep

Track N configurations (different lattice parameters) in parallel, one GPU
thread per configuration.  This is ideal for:

- Tune footprints (scan quadrupole strengths)
- Dynamic aperture vs. sextupole strength
- RF voltage / phase acceptance scans

### Building a `ParamSweepLattice`

```julia
using TrackPad, Metal

beam = Beam(18e9)
lat  = Lattice([
    Quadrupole(0.5; k1 =  1.2, name = :QF, num_int_steps = 4),
    Drift(1.0),
    Quadrupole(0.5; k1 = -1.2, name = :QD, num_int_steps = 4),
    Drift(1.0),
])

# Build Metal GPULattice
gl = gpu_adapt(lat, beam, MetalBackend())   # Float32 on Metal

# Define 1000 configurations: scan QF k1 from 0.8 to 1.6
N     = 1000
qf_idx  = findfirst(e -> isa(e, Quadrupole) && string(e.name) == "QF",
                    lat.elements)
k1_vals = MtlArray(Float32.(range(0.8, 1.6; length = N)))

# Slot 2 in fparams stores k1 for a Quadrupole
ps = ParamSweepLattice(gl, [(qf_idx, 2, k1_vals)])
```

### Float parameter slot reference

| Element | Slot | Meaning |
|---------|------|---------|
| All | 1 | `L` (length, m) |
| `Quadrupole` | 2 | `k1` |
| `Sextupole` | 2 | `k2` |
| `Octupole` | 2 | `k3` |
| `SBend` | 2 | `angle` (rad) |
| `SBend` | 8 | `irho` = `angle/L` (precomputed) |
| `RFCavity` | 2 | `volt` (eV) |
| `RFCavity` | 3 | `freq` (Hz) |
| `RFCavity` | 4 | `lag` (m) |
| `Corrector` | 2 | `xkick` (rad) |
| `Corrector` | 3 | `ykick` (rad) |
| `Solenoid` | 2 | `ks` |

### Running the sweep

```julia
# Initial coordinates — all particles start at x = 1 mm
r0  = MtlArray(repeat(Float32[1e-3, 0, 0, 0, 0, 0]', N, 1))
out = MtlArray(zeros(Float32, N, 6))

param_sweep_linepass!(out, r0, ps)

results = Array(out)
println("x range after sweep: ", extrema(results[:, 1]))
```

### Multiple simultaneous variations

Pass multiple tuples to vary several elements at once:

```julia
qd_idx  = findfirst(e -> isa(e, Quadrupole) && string(e.name) == "QD",
                    lat.elements)
k1d_vals = MtlArray(Float32.(range(-1.6, -0.8; length = N)))

ps = ParamSweepLattice(gl, [
    (qf_idx, 2, k1f_vals),   # QF k1
    (qd_idx, 2, k1d_vals),   # QD k1
])
```

Each configuration `j` uses `k1f_vals[j]` for QF and `k1d_vals[j]` for QD.

---

## Use Case 3: CPU Multi-Threaded Fallback

[`cpu_batch_linepass!`](@ref) is a pure-CPU alternative that uses
`Threads.@threads` for parallelism.  No GPU or `GPULattice` is required —
it operates directly on the `Lattice`:

```julia
using TrackPad

beam   = Beam(3.0e9)
lat    = Lattice([...])
N      = 10_000
coords = zeros(Float64, N, 6)
# ... fill ...

# Requires Julia to be started with multiple threads:
#   julia --project=. -t 8 myscript.jl
cpu_batch_linepass!(coords, lat, beam)          # 1 turn
cpu_batch_linepass!(coords, lat, beam, 100)     # 100 turns
```

---

## Precision Notes

### Metal (Float32)

TrackPad's Metal path uses 32-bit floats. Compare Metal results against the
Float32 CPU backend with scale-appropriate absolute and relative tolerances.

If higher precision is needed on Apple hardware, use the CPU backend
(`GPULattice(lat, beam; dtype = Float64)`) with `cpu_batch_linepass!`.

### CUDA

Tests on an NVIDIA A100 measured maximum CPU/CUDA differences below `2e-16`
for Float64 and below `2.3e-7` for Float32 over the verification lattice.
Exact bitwise equality is not required because GPU fused operations can differ
from CPU evaluation.

### Lost Particles

A particle is considered lost when its longitudinal momentum `pz² ≤ 0`.
On GPU, lost particles accumulate `Inf` or `NaN` coordinates that propagate
naturally through subsequent elements.  Check for lost particles with:

```julia
lost = vec(any(isnan, Array(coords), dims = 2))
```

---

## Performance Tips

### A100 Benchmark

Run `scripts/benchmark_cpu_gpu.jl` with Julia threads enabled to compare the
threaded element path, the flattened KernelAbstractions CPU path, one CUDA GPU,
and a host-partitioned multi-GPU run. Compilation and input reset are excluded
from device-resident timings.

The 2026-07-22 benchmark used 1,000,000 Float64 particles, 100 FODO elements,
five samples, 64 Julia threads on two AMD EPYC 7313 sockets, and four NVIDIA
A100-SXM4-40GB GPUs:

| Path | Median | Throughput | Speedup vs KA CPU |
|------|--------|------------|-------------------|
| Threaded CPU element tracking | `1.60204 s` | `0.06242 Gparticle-elements/s` | `0.27x` |
| KernelAbstractions CPU | `0.43127 s` | `0.23187 Gparticle-elements/s` | `1.0x` |
| One A100, device resident | `0.007977 s` | `12.536 Gparticle-elements/s` | `54.06x` |
| One A100, including H2D and D2H | `0.015541 s` | `6.4346 Gparticle-elements/s` | `27.75x` |
| Four A100s, device resident | `0.003461 s` | `28.897 Gparticle-elements/s` | `124.62x` |

The four-GPU path manually partitions particles across devices and achieved
`2.31x` strong scaling over one A100. These measurements came from a shared
server with existing GPU memory allocations, so they are representative rather
than peak dedicated-node results.

- **Workgroup size**: the default workgroup size is 256 threads.  For small
  particle counts (< 512) you may want to call the kernel directly with a
  smaller workgroup size.
- **Multiple turns**: call `batch_ringpass!(coords, gl, nturns)` rather than
  looping in Julia; this avoids Julia overhead between turns.
- **Memory layout**: coordinates are stored in Julia column-major `N×6`
  matrices, so adjacent particle threads access contiguous values for each
  coordinate component.
- **Parameter sweep memory**: the 3D `fparams` array for a sweep of N
  configurations over a lattice of M elements uses
  `18 × M × N × sizeof(T)` bytes.  For 1000 configurations and 500 elements
  in Float32, that is ~36 MB — well within Metal device memory.

---

## Known Limitations

| Feature | Status |
|---------|--------|
| Supported elements | `Marker`, `Drift`, `Quadrupole`, `Sextupole`, `Octupole`, `SBend`, `RFCavity`, `Corrector`, `Solenoid`, `ThinMultipole` |
| `ExactSBend` / `LBend` on GPU | Rejected with `ArgumentError` |
| TPSA power-series tracking on GPU | Not yet implemented |
| Space charge / `BeamBeam` / `Wake` / `Wiggler` | Rejected with `ArgumentError` |
| Misalignment, aperture, radiation, unsupported fringe settings | Rejected when nonzero |
| `Float64` on Metal | Not supported (Apple GPU constraint) |
