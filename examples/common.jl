module TrackPadExamples

using LinearAlgebra
using TrackPad

export default_beam, fodo_ring, madx_fodo, set_fodo_strengths
export instrumented_fodo, boundary_orbit, bpm_values

default_beam() = Beam(3.0e9)

"""Load the MAD-X FODO cell used by notebooks 01 onward."""
function madx_fodo()
    return read_madx(
        joinpath(@__DIR__, "data", "fodo.madx");
        sequence=:FODO, strict=true,
    )
end

function _set_fodo_strength(element::Quadrupole, qf, qd, sf, sd)
    k1 = element.name == :qfh ? qf : element.name == :qd ? qd : element.k1
    return Quadrupole(
        element.L, k1; name=element.name,
        num_int_steps=element.num_int_steps,
    )
end

function _set_fodo_strength(element::Sextupole, qf, qd, sf, sd)
    k2 = element.name == :sf ? sf : element.name == :sd ? sd : element.k2
    return Sextupole(
        element.L, k2; name=element.name,
        num_int_steps=element.num_int_steps,
    )
end

_set_fodo_strength(element::AbstractElement, qf, qd, sf, sd) = element

"""Return a copy of the MAD-X cell with new family strengths."""
function set_fodo_strengths(lat::Lattice;
                            qf=1.414213562373095,
                            qd=-1.414213562373095,
                            sf=8.0, sd=-8.0)
    elements = AbstractElement[
        _set_fodo_strength(element, qf, qd, sf, sd) for element in lat
    ]
    return Lattice(elements; name=lat.name, periodic=lat.periodic)
end

"""One periodic FODO cell. Strengths are normalized `k1` values in m^-2."""
function fodo_ring(; kf=1.2, kd=-1.2)
    elements = AbstractElement[
        Drift(0.5; name=:D1),
        Quadrupole(0.2, kf; name=:QF, num_int_steps=4),
        Drift(1.0; name=:D2),
        Quadrupole(0.2, kd; name=:QD, num_int_steps=4),
        Drift(0.5; name=:D3),
    ]
    return Lattice(elements; name=:FODO, periodic=true)
end

"""Instrument the MAD-X cell with four BPM/error/corrector stations."""
function instrumented_fodo(;
        qf=1.414213562373095, qd=-1.414213562373095,
        sf=8.0, sd=-8.0, qf_error=0.0, qd_error=0.0,
        error_kicks=[20e-6, -15e-6, 12e-6, -8e-6],
        corrector_kicks=zeros(4))
    all(length(v) == 4 for v in (error_kicks, corrector_kicks)) ||
        throw(ArgumentError("instrumented_fodo expects four orbit stations"))
    base, _ = madx_fodo()
    base = set_fodo_strengths(
        base; qf=qf + qf_error, qd=qd + qd_error, sf, sd,
    )
    elements = AbstractElement[]
    station = 0
    for (index, element) in enumerate(base)
        if index in (1, 5, 9, 13)
            station += 1
            append!(elements, AbstractElement[
                Marker(name=Symbol("BPM$station")),
                Corrector(0.0, error_kicks[station], 0.0;
                          name=Symbol("ERR$station")),
                Corrector(0.0, corrector_kicks[station], 0.0;
                          name=Symbol("HC$station")),
            ])
        end
        push!(elements, element)
    end
    return Lattice(elements; name=:INSTRUMENTED_FODO, periodic=true)
end

"""Track a reference particle and return its coordinates at every boundary."""
function boundary_orbit(lat::Lattice, beam::Beam,
                        initial::TrackPad.SVector{6,T}) where T
    orbit = Matrix{T}(undef, length(lat) + 1, 6)
    orbit[1, :] .= initial
    r = initial
    for (index, element) in enumerate(lat)
        r = pass!(element, r, inv(beam.beta))
        orbit[index + 1, :] .= r
    end
    return orbit
end

"""Select values defined at the entrances of named BPM markers."""
function bpm_values(lat::Lattice, values::AbstractVector)
    indices = findall(element -> element isa Marker &&
        startswith(String(element.name), "BPM"), lat.elements)
    return values[indices]
end

end
