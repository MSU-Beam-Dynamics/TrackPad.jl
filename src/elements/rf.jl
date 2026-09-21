"""
    RFCavity{T, N}

An RF cavity. `energy` is the reference total energy in eV (pass `beam.energy`) and `charge` is the
reference-particle charge in units of the elementary charge. Direct
construction defaults to `+1` for backward constructor compatibility; PALS
and MAD-X readers set it from the reference beam.
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
    charge::T
end

function RFCavity(L, volt, freq, lag = 0.0;
                  name::Union{Symbol, String} = :RFCA, h = 1.0, philag = 0.0,
                  energy = 0.0, charge = 1.0)
    T = _promote_element_type(L, volt, freq, lag, h, philag, energy, charge)
    RFCavity{T, Symbol}(
        Symbol(name), T(L), T(volt), T(freq), T(h), T(lag), T(philag),
        T(energy), T(charge),
    )
end

function Adapt.adapt_structure(to, x::RFCavity)
    RFCavity(
        adapt(to, x.L), adapt(to, x.volt), adapt(to, x.freq), adapt(to, x.lag);
        name=x.name, h=adapt(to, x.h), philag=adapt(to, x.philag),
        energy=adapt(to, x.energy), charge=adapt(to, x.charge),
    )
end

const C_LIGHT = 2.99792458e8

"""
    CrabCavity{T, N}

Canonical crab cavity element. `energy` is the reference total energy in eV (pass `beam.energy`) and
`charge` is the signed reference charge in elementary-charge units.
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
    charge::T
end

function CrabCavity(L;
                    name::Union{Symbol, String} = :CRABCAVITY,
                    volt = 0.0, freq = 0.0, phi = 0.0, errors = nothing,
                    energy = 1.0e9, charge = 1.0)
    T = _promote_element_type(L, volt, freq, phi, errors, energy, charge)
    CrabCavity{T, Symbol}(
        Symbol(name), T(L), T(volt), T(freq), T(2pi) * T(freq) / T(C_LIGHT), T(phi),
        SVector{2, T}(_default_vec(errors, T, Val(2))), T(energy), T(charge),
    )
end

function Adapt.adapt_structure(to, x::CrabCavity)
    CrabCavity(
        adapt(to, x.L);
        name = x.name, volt = adapt(to, x.volt), freq = adapt(to, x.freq),
        phi = adapt(to, x.phi), errors = adapt(to, x.errors),
        energy = adapt(to, x.energy), charge = adapt(to, x.charge),
    )
end

"""
    AccelCavity{T, N}

Longitudinal accelerating cavity. `energy` is the reference total energy in eV (pass `beam.energy`)
and `charge` is the signed reference charge in elementary-charge units.
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
    charge::T
end

function AccelCavity(L;
                     name::Union{Symbol, String} = :ACCELCAVITY,
                     volt = 0.0, freq = 0.0, h = 1.0, phis = 0.0,
                     energy = 1.0e9, charge = 1.0)
    T = _promote_element_type(L, volt, freq, h, phis, energy, charge)
    AccelCavity{T, Symbol}(
        Symbol(name), T(L), T(volt), T(freq), T(2pi) * T(freq) / T(C_LIGHT),
        T(h), T(phis), T(energy), T(charge),
    )
end

function Adapt.adapt_structure(to, x::AccelCavity)
    AccelCavity(
        adapt(to, x.L);
        name = x.name, volt = adapt(to, x.volt), freq = adapt(to, x.freq),
        h = adapt(to, x.h), phis = adapt(to, x.phis),
        energy = adapt(to, x.energy), charge = adapt(to, x.charge),
    )
end

"""
    LongitudinalRFMap(alphac, rf)

Thin longitudinal drift that applies one turn of phase slip for a ring whose
transverse lattice is not tracked. `alphac` is the momentum compaction
`(1/C) dC/dδP` and `rf` the RF element (`RFCavity`, `AccelCavity`, `CrabCavity`)
whose frequency `f` and harmonic number `h` fix the circumference
`C = h c/f`. The map leaves the transverse coordinates and `δE` unchanged and
shifts `z` by

    Δz = −C η δE / β0²,   η = alphac − 1/γ0²,

which is the path-length change of the off-momentum closed orbit expressed in
the stored coordinates `z = s/β0 − ct`, `δE = (E − E0)/(P0 c)` (the second
`β0` converts `δE` to `δP`). `alphac` is exactly the `alphac` field of
[`periodic_twiss`](@ref) with its default `wrt=:deltap`.
"""
struct LongitudinalRFMap{T, E<:AbstractElement} <: AbstractLongitudinalRFMap
    alphac::T
    rf::E
end

function Adapt.adapt_structure(to, x::LongitudinalRFMap)
    LongitudinalRFMap(adapt(to, x.alphac), adapt(to, x.rf))
end
