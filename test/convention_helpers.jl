const LONGITUDINAL_COORDINATE_SIGN = (1, 1, 1, 1, -1, 1)

"""Convert coordinate vectors between TrackPad and JuTrack conventions."""
function flip_longitudinal_coordinate(r::AbstractVector)
    out = collect(r)
    out[5] = -out[5]
    return out
end

"""Convert row-major particle matrices between TrackPad and JuTrack conventions."""
function flip_longitudinal_coordinate(r::AbstractMatrix)
    out = copy(r)
    out[:, 5] .*= -1
    return out
end

"""Conjugate a six-dimensional map by the longitudinal sign transformation."""
function canonicalize_jutrack_map(M::AbstractMatrix)
    signs = collect(LONGITUDINAL_COORDINATE_SIGN)
    return signs .* M .* signs'
end
