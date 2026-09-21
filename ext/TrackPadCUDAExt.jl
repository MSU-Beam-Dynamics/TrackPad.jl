"""
    TrackPadCUDAExt

CUDA extension for TrackPad.jl.
Provides `gpu_adapt` helpers that move `GPULattice` / `ParamSweepLattice` to an
NVIDIA GPU via CUDA.jl.

CUDA supports both `Float32` and `Float64`; the adapter defaults to `Float64`.

This module is loaded automatically when `using CUDA` is called alongside
TrackPad.

## Typical workflow
```julia
using TrackPad, CUDA, StaticArrays

beam = Beam(18e9)
lat  = Lattice([Quadrupole(0.5, 1.2; num_int_steps=4),
                Drift(1.0),
                Quadrupole(0.5, -1.2; num_int_steps=4),
                Drift(1.0)]; periodic=true)

N = 100_000
coords = CUDA.randn(Float64, N, 6)
gl_cuda = gpu_adapt(lat, beam, CUDABackend(); dtype=Float64)
track!(coords, gl_cuda)
track!(coords, gl_cuda; nturns=100)
```
"""
module TrackPadCUDAExt

using TrackPad
using CUDA
using Adapt

# ---------------------------------------------------------------------------
# gpu_adapt: Lattice / GPULattice / ParamSweepLattice -> CuArray
# ---------------------------------------------------------------------------

"""
    TrackPad.gpu_adapt(lat, beam, ::CUDABackend; dtype=Float64)

Build a `GPULattice` and upload it to the CUDA device.
"""
function TrackPad.gpu_adapt(lat::TrackPad.Lattice,
                            beam::TrackPad.Beam,
                            ::CUDA.CUDABackend;
                            dtype::Type = Float64)
    dtype in (Float32, Float64) || throw(ArgumentError("CUDA dtype must be Float32 or Float64"))
    gl = TrackPad.GPULattice(lat, beam; dtype = dtype)
    return Adapt.adapt(CUDA.CuArray, gl)
end

"""
    TrackPad.gpu_adapt(gl::GPULattice, ::CUDABackend)
"""
function TrackPad.gpu_adapt(gl::TrackPad.GPULattice, ::CUDA.CUDABackend)
    return Adapt.adapt(CUDA.CuArray, gl)
end

"""
    TrackPad.gpu_adapt(ps::ParamSweepLattice, ::CUDABackend)
"""
function TrackPad.gpu_adapt(ps::TrackPad.ParamSweepLattice, ::CUDA.CUDABackend)
    return Adapt.adapt(CUDA.CuArray, ps)
end

end # module TrackPadCUDAExt
