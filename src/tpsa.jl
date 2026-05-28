# =============================================================================
# TPSA (Truncated Power Series Algebra) API
#
# The actual implementation lives in ext/TrackPadPolySeriesExt.jl and is
# loaded automatically when PolySeries.jl is loaded alongside TrackPad.
# =============================================================================

"""
    tpsa_map(lat::Lattice, beam::Beam;
             order::Int = 1,
             closed_orbit::Union{Nothing, AbstractVector} = nothing)

Compute the Taylor transfer map of `lat` up to polynomial `order` about
`closed_orbit` (zero orbit when `nothing`).

Requires `PolySeries.jl` to be loaded:
```julia
using TrackPad, PolySeries
M = tpsa_map(lat, beam; order = 2)
```

Returns `SVector{6, CTPS{Float64}}` where `M[i]` is the polynomial giving
final phase-space coordinate `i` as a function of the six initial offsets.
"""
function tpsa_map end
