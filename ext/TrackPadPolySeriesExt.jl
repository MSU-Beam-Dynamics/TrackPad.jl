"""
    TrackPadPolySeriesExt

Extension that activates TPSA map computation when PolySeries.jl is loaded.

To use:
```julia
using TrackPad, PolySeries

# Build a FODO lattice, set up a Beam, then:
M = tpsa_map(lat, beam; order=1)            # SVector{6, CTPS{Float64}}
M_co = tpsa_map(lat, beam; order=2,         # about a given closed orbit
                closed_orbit=[x0,px0,y0,py0,z0,δ0])
```

The returned `SVector{6, CTPS{Float64}}` encodes the transfer map: element `i`
is the Taylor series giving the final coordinate `i` as a function of the six
initial offsets.  To extract the one-turn matrix:

```julia
using LinearAlgebra
M = tpsa_map(lat, beam)
# coefficient(M[i], j) gives ∂final_i / ∂initial_j  (linear part)
```
"""
module TrackPadPolySeriesExt

using TrackPad
using PolySeries
using StaticArrays

# ---------------------------------------------------------------------------
# tpsa_map: compute the one-turn (or one-line) Taylor transfer map
# ---------------------------------------------------------------------------

"""
    tpsa_map(lat::Lattice, beam::Beam;
             order::Int = 1,
             closed_orbit::Union{Nothing, AbstractVector} = nothing) -> SVector{6, CTPS{Float64}}

Compute the Taylor transfer map of `lat` up to the given `order` about
`closed_orbit` (defaults to the zero orbit when `nothing`).

The six-variable PolySeries descriptor is set globally inside the call; if you
need to call other PolySeries code afterwards, make sure to re-set the
descriptor to your requirements.

# Returns
`SVector{6, CTPS{Float64}}` where `M[i]` is the polynomial giving final
coordinate `i` as a function of the six initial offsets `(Δx,Δpₓ,Δy,Δpᵧ,Δz,Δδ)`.

# Example
```julia
using TrackPad, PolySeries

fodo = Lattice([Drift(1.0), Quadrupole(0.5; k1=1.2), Drift(1.0)])
b = Beam(1e9)
M = tpsa_map(fodo, b; order=1)

# Extract linear transfer matrix
R = [get_coefficient(M[i], j) for i in 1:6, j in 1:6]
```
"""
function TrackPad.tpsa_map(lat::Lattice, beam::Beam;
                            order::Int = 1,
                            closed_orbit::Union{Nothing,AbstractVector} = nothing)
    # --- set global PolySeries descriptor (6 phase-space variables) ----------
    set_descriptor!(6, order)

    # --- reference closed-orbit point (Float64) -------------------------------
    co = if closed_orbit !== nothing
        Float64.(closed_orbit)
    else
        zeros(Float64, 6)
    end

    # --- build initial CTPS coordinates: cₒ + 1·eᵢ  (each a Taylor variable) -
    r0 = SVector(
        CTPS(co[1], 1, 6, order),
        CTPS(co[2], 2, 6, order),
        CTPS(co[3], 3, 6, order),
        CTPS(co[4], 4, 6, order),
        CTPS(co[5], 5, 6, order),
        CTPS(co[6], 6, 6, order),
    )

    # --- track through lattice -------------------------------------------------
    return linepass(lat, r0, beam)
end

end # module TrackPadPolySeriesExt
