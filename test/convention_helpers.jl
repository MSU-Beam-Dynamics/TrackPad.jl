# Shared helpers for the JuTrack parity tests. The JuTrack outputs themselves
# are frozen in jutrack_reference.jl (generated; see jutrack_reference/README.md),
# so the test suite has no JuTrack dependency. The generator includes this file
# before that fixture exists, hence the guard.
let f = joinpath(@__DIR__, "jutrack_reference.jl")
    isfile(f) && Base.include(@__MODULE__, f)
end

const LONGITUDINAL_COORDINATE_SIGN = (1, 1, 1, 1, -1, 1)

"""
Ten deterministic macroparticles shared by every JuTrack parity test and by the
generator of `test/jutrack_reference.jl`. The frozen reference data is keyed to
these exact coordinates, so changing them means regenerating the fixture.
"""
const PARITY_PARTICLES = [
    1.0e-4   2.0e-4   3.0e-4  -1.0e-4   5.0e-5   2.0e-4
   -2.2e-4  1.7e-4  -1.1e-4   2.3e-4  -7.0e-5   1.0e-4
    3.5e-4  -2.1e-4  8.0e-5  -1.8e-4   1.4e-4  -2.6e-4
   -4.0e-4  2.9e-4   1.6e-4   9.0e-5  -1.2e-4   3.1e-4
    5.2e-4  -3.3e-4 -2.4e-4   1.1e-4   2.6e-4  -3.7e-4
   -6.1e-4  4.4e-4   2.7e-4  -2.5e-4  -3.0e-4   4.5e-4
    7.0e-4  -5.2e-4 -3.1e-4   3.4e-4   3.3e-4  -5.0e-4
   -8.0e-4  6.0e-4   3.9e-4  -4.2e-4  -3.8e-4   5.8e-4
    9.1e-4  -6.7e-4 -4.6e-4   5.1e-4   4.4e-4  -6.3e-4
   -9.8e-4  7.3e-4   5.3e-4  -5.9e-4  -4.9e-4   7.1e-4
]

"""
    jutrack_reference(key)

Frozen JuTrack output for a parity case, from `test/jutrack_reference.jl`.
Throws a directed error when a case has no recorded reference, which is what a
newly added parity case looks like before the fixture is regenerated.
"""
function jutrack_reference(key::AbstractString)
    haskey(JUTRACK_REFERENCE, key) || error(
        "no frozen JuTrack reference for \"$key\"; regenerate it with " *
        "`julia --project=test/jutrack_reference test/jutrack_reference/generate.jl`")
    return JUTRACK_REFERENCE[key]
end

"""
    jutrack_beam(energy; mass=M_ELECTRON, charge=-1.0)

TrackPad `Beam` matching a JuTrack call that received `energy=energy`. JuTrack's
`energy` is the kinetic energy while TrackPad's positional energy is the total
energy, so parity tests must build the TrackPad side with the kinetic form.
"""
jutrack_beam(energy; mass=M_ELECTRON, charge=-1.0) =
    Beam(; kinetic=energy, mass=mass, charge=charge)

"""Convert coordinate vectors between the TrackPad and JuTrack longitudinal
sign conventions. Only the generator still needs this; it is kept here so the
fixture and the tests share one definition."""
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
