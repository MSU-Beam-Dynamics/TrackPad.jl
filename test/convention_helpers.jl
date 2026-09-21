const LONGITUDINAL_COORDINATE_SIGN = (1, 1, 1, 1, -1, 1)

"""
    jutrack_beam(energy; mass=M_ELECTRON, charge=-1.0)

TrackPad `Beam` matching a JuTrack call that received `energy=energy`. JuTrack's
`energy` is the kinetic energy while TrackPad's positional energy is the total
energy, so parity tests must build the TrackPad side with the kinetic form.
"""
jutrack_beam(energy; mass=M_ELECTRON, charge=-1.0) =
    Beam(; kinetic=energy, mass=mass, charge=charge)

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
