"""
    RFCavity{T, N}

An RF cavity.
"""
struct RFCavity{T, N} <: AbstractCavity
    name::N
    L::T
    volt::T
    freq::T
    h::T
    lag::T
    philag::T
    energy::T
end

function RFCavity(L, volt, freq, lag = 0.0;
                  name::Union{Symbol, String} = :RFCA, h = 1.0, philag = 0.0, energy = 0.0)
    T = _promote_element_type(L, volt, freq, lag, h, philag, energy)
    RFCavity{T, Symbol}(Symbol(name), T(L), T(volt), T(freq), T(h), T(lag), T(philag), T(energy))
end

function Adapt.adapt_structure(to, x::RFCavity)
    RFCavity(nothing, adapt(to, x.L), adapt(to, x.volt), adapt(to, x.freq), adapt(to, x.h),
             adapt(to, x.lag), adapt(to, x.philag), adapt(to, x.energy))
end

const C_LIGHT = 2.99792458e8

"""
    CrabCavity{T, N}

Canonical crab cavity element.
"""
struct CrabCavity{T, N} <: AbstractCavity
    name::N
    L::T
    volt::T
    freq::T
    k::T
    phi::T
    errors::SVector{2, T}
    energy::T
end

function CrabCavity(L;
                    name::Union{Symbol, String} = :CRABCAVITY,
                    volt = 0.0, freq = 0.0, phi = 0.0, errors = nothing, energy = 1.0e9)
    T = _promote_element_type(L, volt, freq, phi, errors, energy)
    CrabCavity{T, Symbol}(
        Symbol(name), T(L), T(volt), T(freq), T(2pi) * T(freq) / T(C_LIGHT), T(phi),
        SVector{2, T}(_default_vec(errors, T, Val(2))), T(energy),
    )
end

function Adapt.adapt_structure(to, x::CrabCavity)
    CrabCavity(
        adapt(to, x.L);
        name = x.name, volt = adapt(to, x.volt), freq = adapt(to, x.freq),
        phi = adapt(to, x.phi), errors = adapt(to, x.errors), energy = adapt(to, x.energy),
    )
end

"""
    AccelCavity{T, N}

Longitudinal accelerating cavity.
"""
struct AccelCavity{T, N} <: AbstractCavity
    name::N
    L::T
    volt::T
    freq::T
    k::T
    h::T
    phis::T
    energy::T
end

function AccelCavity(L;
                     name::Union{Symbol, String} = :ACCELCAVITY,
                     volt = 0.0, freq = 0.0, h = 1.0, phis = 0.0, energy = 1.0e9)
    T = _promote_element_type(L, volt, freq, h, phis, energy)
    AccelCavity{T, Symbol}(
        Symbol(name), T(L), T(volt), T(freq), T(2pi) * T(freq) / T(C_LIGHT), T(h), T(phis), T(energy),
    )
end

function Adapt.adapt_structure(to, x::AccelCavity)
    AccelCavity(
        adapt(to, x.L);
        name = x.name, volt = adapt(to, x.volt), freq = adapt(to, x.freq),
        h = adapt(to, x.h), phis = adapt(to, x.phis), energy = adapt(to, x.energy),
    )
end

"""
    LongitudinalRFMap{T, E}

Simple longitudinal map linked to an RF element.
"""
struct LongitudinalRFMap{T, E<:AbstractElement} <: AbstractLongitudinalRFMap
    alphac::T
    rf::E
end

function Adapt.adapt_structure(to, x::LongitudinalRFMap)
    LongitudinalRFMap(adapt(to, x.alphac), adapt(to, x.rf))
end
