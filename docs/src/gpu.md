```@meta
CurrentModule = TrackPad
```

# [GPU Acceleration](@id gpu_guide)

TrackPad.jl supports GPU-accelerated particle tracking via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
which provides a single backend-agnostic kernel language that compiles to:

| Backend | Package | Float width |
|---------|---------|-------------|
| Apple Metal | `Metal.jl` | **Float32 only** (Apple GPU constraint) |
| NVIDIA CUDA | `CUDA.jl` | Float32 or Float64 |
| CPU (multi-core) | *(built-in)* | Float32 or Float64 |

Four use cases are supported:

1. **Multi-particle batch tracking** — N particles tracked in parallel through the same lattice.
2. **Parameter sweep** — one (or N) particle(s) tracked through N different lattice configurations simultaneously.
3. **CPU multi-threaded fallback** — no GPU required; uses `Threads.@threads` over particles.
4. **Batched derivatives** — Jacobians, Hessian-vector products, and Hessians for N initial conditions using Enzyme.

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

Install the backend package in the active environment, then load it with
TrackPad. The corresponding extension activates automatically.

---

## Core Types

### `GPULattice`

[`GPULattice`](@ref) is a flat-array encoding of a [`Lattice`](@ref) that can
be executed by a GPU kernel.  All element parameters are packed into two
matrices:

- `fparams :: Matrix{T}` — float parameters, shape `(18, n_elems)`
- `iparams :: Matrix{Int32}` — integer parameters (e.g. integration steps), shape `(2, n_elems)`
- `elem_types :: Vector{Int32}` — element type codes
- `periodic :: Bool` — source lattice boundary for repeated-pass validation

The `T` type parameter controls precision.  Use `Float32` for Metal or
`Float64` for CUDA/CPU.

### `ParamSweepLattice`

[`ParamSweepLattice`](@ref) stores one shared `GPULattice` parameter table plus
only the selected per-configuration values. For `K` varied parameters and `N`
configurations, `variation_values` has shape `N×K`. This order gives
coalesced access when adjacent GPU threads read the same varied parameter. A small shared
`variation_index` table maps each `(parameter slot, element)` to a variation
row. Unvaried elements bypass this lookup entirely.

The kernel applies an override lazily when the corresponding element parameter
is loaded. It does not materialize a separate `18×M` lattice for every
configuration. All N configurations are processed in parallel, one GPU thread
per configuration.

---

## Use Case 1: Multi-Particle Batch Tracking

Track N particles through the same lattice, one GPU thread per particle.

### Metal (Apple GPU, Float32)

```julia
using TrackPad, Metal, StaticArrays

beam = Beam(18e9)
lat, beam = read_madx("ring.madx"; periodic=true)   # or build manually

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
lost   = vec(any(x -> !isfinite(x), result; dims = 2))
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
    Quadrupole(0.5, 1.2; name=:QF, num_int_steps=4),
    Drift(1.0),
    Quadrupole(0.5, -1.2; name=:QD, num_int_steps=4),
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
| `SBend` | 2 | `angle` (rad) — the curvature `angle/L` is derived in the kernel, so slot 8 must not be swept |
| `StrongThinGaussianBeam` | 2 | `amplitude` = `N r0 q_w q_s / γ_w` |
| `StrongThinGaussianBeam` | 3, 4 | `rmssizex`, `rmssizey` (m) |
| `StrongThinGaussianBeam` | 5, 6 | `xoffset`, `yoffset` (m) |
| `RFCavity` | 2 | `volt` (eV) |
| `RFCavity` | 3 | `freq` (Hz) |
| `RFCavity` | 4 | `lag` (m) |
| `RFCavity` | 8 | reference-particle `charge` (e) |
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

Pass multiple equal-length vectors to vary several parameters together. This
aligned mode is appropriate for Monte Carlo trials and correlated errors:

```julia
qd_idx  = findfirst(e -> isa(e, Quadrupole) && string(e.name) == "QD",
                    lat.elements)
k1f_trials = Float32.(1.2 .+ 0.01 .* randn(N))
k1d_trials = Float32.(-1.2 .+ 0.01 .* randn(N))

ps = ParamSweepLattice(gl, [
    (qf_idx, 2, k1f_trials),   # trial j QF k1
    (qd_idx, 2, k1d_trials),   # trial j QD k1
])
```

Each configuration `j` uses `k1f_trials[j]` and `k1d_trials[j]`.

### Cartesian parameter grids

Use `mode=:cartesian` for a systematic scan over every combination:

```julia
qf_values = Float32.(range(1.0, 1.4; length = 21))
qd_values = Float32.(range(-1.4, -1.0; length = 17))
rf_values = Float32.(range(1.8e5, 2.2e5; length = 9))

ps = ParamSweepLattice(
    gl,
    [
        (qf_idx, 2, qf_values),
        (qd_idx, 2, qd_values),
        (rf_idx, 2, rf_values),
    ];
    mode = :cartesian,
)

N = 21 * 17 * 9
```

The first variation changes fastest. Parameter vectors are expanded directly
into the sparse `N×K` value table; a dense `18×M×N` lattice is never created.

---

## Use Case 3: CPU Multi-Threaded Fallback

[`cpu_batch_linepass!`](@ref) is a pure-CPU alternative that uses
`Threads.@threads` for parallelism.  No GPU or `GPULattice` is required —
it operates directly on the `Lattice`:

```julia
using TrackPad

beam   = Beam(3.0e9)
lat    = Lattice([...])
ring   = Lattice([...]; periodic=true)
N      = 10_000
coords = zeros(Float64, N, 6)
# ... fill ...

# Requires Julia to be started with multiple threads:
#   julia --project=. -t 8 myscript.jl
cpu_batch_linepass!(coords, lat, beam)          # one pass
cpu_batch_linepass!(coords, ring, beam, 100)    # repeated periodic passes
```

---

## Use Case 4: Batched Derivatives

Load Enzyme together with TrackPad and the GPU backend:

```julia
using TrackPad, CUDA, Enzyme

gl = gpu_adapt(lat, beam, CUDABackend(); dtype = Float64)
coordinates = CuArray(initial_coordinates)  # N×6
N = size(coordinates, 1)
```

### Jacobians

`batch_jacobian!` computes six forward-mode derivatives for every particle:

```julia
jacobian = CUDA.zeros(Float64, N, 6, 6)
batch_jacobian!(jacobian, coordinates, gl)
```

The layout is
`jacobian[particle, output_coordinate, input_coordinate]`. The input
coordinates are not modified. A `Float64` Jacobian requires 288 bytes per
particle, or approximately 275 MiB for one million particles.

### Hessian-vector products

For a direction vector `v[p, :]`, `batch_hessian_vector_product!` returns
the Hessian contracted with `v[p, :]` along its final input axis:

```julia
vectors = CUDA.randn(Float64, N, 6)
product = CUDA.zeros(Float64, N, 6, 6)
batch_hessian_vector_product!(
    product,
    coordinates,
    vectors,
    gl;
    step = 1e-5,
)
```

The current `:enzyme_finite_difference` method uses a centered directional
difference of exact Enzyme Jacobians. Set `step` according to the coordinate
scale when the default `cbrt(eps(T))` is unsuitable.

### Full Hessians

```julia
hessian = CUDA.zeros(Float64, N, 6, 6, 6)
batch_hessian!(hessian, coordinates, gl; step = 1e-5)
```

The layout is
`hessian[particle, output_coordinate, input_coordinate_1, input_coordinate_2]`.
The final two axes are symmetrized by default. A full `Float64` Hessian
requires 1,728 bytes per particle, or approximately 1.61 GiB for one million
particles.

Full Hessians evaluate six Hessian-vector products and are considerably more
expensive than Jacobians. Prefer Hessian-vector products when an optimizer or
analysis can consume directional curvature directly.

Run the CUDA parity suite on an NVIDIA host with:

```bash
julia --project=test/cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=test/cuda test/cuda/runtests.jl
```

---

## Precision Notes

### Metal (Float32)

TrackPad's Metal path uses 32-bit floats. Compare Metal results against the
Float32 CPU backend with scale-appropriate absolute and relative tolerances.

If higher precision is needed on Apple hardware, use the CPU backend
(`GPULattice(lat, beam; dtype = Float64)`) with `cpu_batch_linepass!`.

### CUDA

Tests on an NVIDIA A100 measured maximum CPU/CUDA differences below `2.5e-16`
for Float64 and below `2.4e-7` for Float32 over the verification lattice.
Exact bitwise equality is not required because GPU fused operations can differ
from CPU evaluation.

### Lost Particles

A particle is considered lost when its normalized longitudinal momentum
``\pi_s^2\le0``.
On GPU, lost particles accumulate `Inf` or `NaN` coordinates that propagate
naturally through subsequent elements.  Check for lost particles with:

```julia
lost = vec(any(x -> !isfinite(x), Array(coords); dims = 2))
```

---

## Performance Tips

### A100 Benchmark

Benchmarks should compare the threaded element path, the flattened
KernelAbstractions CPU path, one CUDA GPU, and a host-partitioned multi-GPU run.
Compilation and input reset must be excluded from device-resident timings.

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
- **Multiple turns**: use `batch_ringpass!(coords, gl, nturns)` to make ring
  intent explicit. The current implementation launches one fused-lattice
  kernel per turn.
- **Memory layout**: coordinates are stored in Julia column-major `N×6`
  matrices, so adjacent particle threads access contiguous values for each
  coordinate component.
- **Parameter sweep memory**: for K varied parameters, N configurations, and M
  elements, sweep-specific storage is
  `19 × M × sizeof(Int32) + K × N × sizeof(T)`. The base
  `18 × M × sizeof(T)` lattice table is shared. For 3 varied parameters,
  1,000,000 configurations, and 500 elements in Float64, this is about 24 MB
  for values plus 38 KB for lookup metadata, instead of 72 GB for a dense
  sweep.

---

## [Known Limitations](@id gpu_limitations)

| Feature | Status |
|---------|--------|
| Supported elements | `Marker`, `Patch`, `Drift`, `Quadrupole`, `Sextupole`, `Octupole`, `SBend`, `RFCavity`, `Corrector`, `Solenoid`, `ThinMultipole`, `StrongThinGaussianBeam` |
| `ExactSBend` / `LBend` on GPU | Rejected with `ArgumentError` |
| TPSA power-series tracking on GPU | Not yet implemented |
| Space charge / `BeamBeam` / `Wake` / `Wiggler` | Rejected with `ArgumentError` |
| Misalignment, aperture, radiation, unsupported fringe settings | Rejected when nonzero |
| `Float64` on Metal | Not supported (Apple GPU constraint) |
| Exact nested-Enzyme CUDA Hessians | Blocked by the current Enzyme/LLVM toolchain; centered differences of Enzyme Jacobians are used |
