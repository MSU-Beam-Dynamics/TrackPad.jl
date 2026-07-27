"""
    TrackPadMetalExt

Metal extension for TrackPad.jl.

Provides `gpu_adapt` helpers that move `GPULattice` / `ParamSweepLattice`
to an Apple GPU via Metal.jl.

Metal supports **Float32 only** for GPU arithmetic.  All lattice parameters
are stored as `Float32` on the Metal device.

This module is loaded automatically when `using Metal` is called alongside
TrackPad.

## Typical workflow
```julia
using TrackPad, Metal, StaticArrays

beam = Beam(18e9)
lat  = read_madx("ring.madx")    # or build manually

N      = 1_000_000
coords = MtlArray(zeros(Float32, N, 6))
# … fill coords with initial beam distribution …

gl = gpu_adapt(lat, beam, MetalBackend())   # Float32 on MtlArray
batch_linepass!(coords, gl)                 # in-place on Metal GPU

result = Array(coords)     # copy back to CPU
lost   = any(isnan, result, dims=2)[:, 1]  # Bool vector: which particles lost
```
"""
module TrackPadMetalExt

using TrackPad
using Metal
using Adapt

# -----------------------------------------------------------------------
# gpu_adapt: Lattice → GPULattice on Metal  (Float32)
# -----------------------------------------------------------------------

"""
    TrackPad.gpu_adapt(lat::Lattice, beam::Beam, ::MetalBackend;
                       dtype=Float32) -> GPULattice on MtlArray

Build a `Float32` `GPULattice` and upload it to the Metal device.
`dtype` is accepted but always clamped to `Float32` (Metal limitation).
"""
function TrackPad.gpu_adapt(lat::TrackPad.Lattice,
                              beam::TrackPad.Beam,
                              ::Metal.MetalBackend;
                              dtype::Type = Float32)
    gl = TrackPad.GPULattice(lat, beam; dtype = Float32)
    return Adapt.adapt(Metal.MtlArray, gl)
end

# -----------------------------------------------------------------------
# gpu_adapt: GPULattice (CPU) → GPULattice on Metal
# -----------------------------------------------------------------------

"""
    TrackPad.gpu_adapt(gl::GPULattice, ::MetalBackend) -> GPULattice on MtlArray
"""
function TrackPad.gpu_adapt(gl::TrackPad.GPULattice, ::Metal.MetalBackend)
    return Adapt.adapt(Metal.MtlArray, gl)
end

# -----------------------------------------------------------------------
# gpu_adapt: ParamSweepLattice (CPU) → ParamSweepLattice on Metal
# -----------------------------------------------------------------------

"""
    TrackPad.gpu_adapt(ps::ParamSweepLattice, ::MetalBackend) -> ParamSweepLattice on MtlArray
"""
function TrackPad.gpu_adapt(ps::TrackPad.ParamSweepLattice, ::Metal.MetalBackend)
    return Adapt.adapt(Metal.MtlArray, ps)
end

end # module TrackPadMetalExt
