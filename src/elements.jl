using StaticArrays
using Adapt

export AbstractElement, AbstractMagnet, AbstractDrift, AbstractCavity
export Marker, Drift, Quadrupole, Sextupole, Octupole, SBend, RFCavity, ThinMultipole, Solenoid, Corrector

abstract type AbstractElement end
abstract type AbstractMagnet <: AbstractElement end
abstract type AbstractDrift <: AbstractElement end
abstract type AbstractCavity <: AbstractElement end

# Helper for default values
# We avoid global constants like ZERO6 because they are strictly Float64.
# Instead, we use utility functions to generate zeros of the correct type T.

function _promote_element_type(L, args...)
    T = typeof(L)
    for arg in args
        if !isnothing(arg)
            T = promote_type(T, eltype(arg))
        end
    end
    return T
end

_default_vec(val, ::Type{T}, ::Val{N}) where {T, N} = isnothing(val) ? zero(SVector{N, T}) : val
_default_mat(val, ::Type{T}, ::Val{N}) where {T, N} = isnothing(val) ? zero(SMatrix{N, N, T, N*N}) : val

"""
    Marker{N}

A marker element with no length or physical effect.
"""
struct Marker{N} <: AbstractElement
    name::N
end
Marker(; name::Union{Symbol,String}=:MARKER) = Marker(Symbol(name))
Adapt.adapt_structure(to, x::Marker) = Marker(nothing)

"""
    Drift{T, N}

A drift space.
"""
struct Drift{T, N} <: AbstractDrift
    name::N
    L::T
    t1::SVector{6, T}
    t2::SVector{6, T}
    r1::SMatrix{6, 6, T, 36}
    r2::SMatrix{6, 6, T, 36}
    r_apertures::SVector{6, T}
    e_apertures::SVector{6, T}
end

function Drift(L; 
               name::Union{Symbol,String}=:DRIFT,
               t1=nothing, t2=nothing,
               r1=nothing, r2=nothing,
               r_apertures=nothing, e_apertures=nothing)
    T = _promote_element_type(L, t1, t2, r1, r2, r_apertures, e_apertures)
    Drift{T, Symbol}(Symbol(name), T(L), 
        SVector{6,T}(_default_vec(t1, T, Val(6))), 
        SVector{6,T}(_default_vec(t2, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r1, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r2, T, Val(6))), 
        SVector{6,T}(_default_vec(r_apertures, T, Val(6))), 
        SVector{6,T}(_default_vec(e_apertures, T, Val(6))))
end

function Adapt.adapt_structure(to, x::Drift)
    Drift(nothing, adapt(to, x.L), adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2), adapt(to, x.r_apertures), adapt(to, x.e_apertures))
end

"""
    Quadrupole{T, N}

A quadrupole magnet.
"""
struct Quadrupole{T, N} <: AbstractMagnet
    name::N
    L::T
    k1::T
    polynom_a::SVector{4, T}
    polynom_b::SVector{4, T}
    max_order::Int
    num_int_steps::Int
    rad::Int
    fringe_entrance::Int
    fringe_exit::Int
    t1::SVector{6, T}
    t2::SVector{6, T}
    r1::SMatrix{6, 6, T, 36}
    r2::SMatrix{6, 6, T, 36}
    r_apertures::SVector{6, T}
    e_apertures::SVector{6, T}
    kick_angle::SVector{2, T}
end

function Quadrupole(L, k1; 
                    name::Union{Symbol,String}=:QUAD,
                    polynom_a=nothing, polynom_b=nothing,
                    max_order::Int=1, num_int_steps::Int=10, rad::Int=0,
                    fringe_entrance::Int=0, fringe_exit::Int=0,
                    t1=nothing, t2=nothing,
                    r1=nothing, r2=nothing,
                    r_apertures=nothing, e_apertures=nothing,
                    kick_angle=nothing)
    T = _promote_element_type(L, k1, polynom_a, polynom_b, t1, t2, r1, r2, r_apertures, e_apertures, kick_angle)
    Quadrupole{T, Symbol}(Symbol(name), T(L), T(k1), 
        SVector{4,T}(_default_vec(polynom_a, T, Val(4))), 
        SVector{4,T}(_default_vec(polynom_b, T, Val(4))), 
        max_order, num_int_steps, rad, fringe_entrance, fringe_exit,
        SVector{6,T}(_default_vec(t1, T, Val(6))), 
        SVector{6,T}(_default_vec(t2, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r1, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r2, T, Val(6))), 
        SVector{6,T}(_default_vec(r_apertures, T, Val(6))), 
        SVector{6,T}(_default_vec(e_apertures, T, Val(6))), 
        SVector{2,T}(_default_vec(kick_angle, T, Val(2))))
end

function Adapt.adapt_structure(to, x::Quadrupole)
    Quadrupole(nothing, adapt(to, x.L), adapt(to, x.k1), adapt(to, x.polynom_a), adapt(to, x.polynom_b),
               x.max_order, x.num_int_steps, x.rad, x.fringe_entrance, x.fringe_exit,
               adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2),
               adapt(to, x.r_apertures), adapt(to, x.e_apertures), adapt(to, x.kick_angle))
end

"""
    Sextupole{T, N}

A sextupole magnet.
"""
struct Sextupole{T, N} <: AbstractMagnet
    name::N
    L::T
    k2::T
    polynom_a::SVector{4, T}
    polynom_b::SVector{4, T}
    max_order::Int
    num_int_steps::Int
    rad::Int
    fringe_entrance::Int
    fringe_exit::Int
    t1::SVector{6, T}
    t2::SVector{6, T}
    r1::SMatrix{6, 6, T, 36}
    r2::SMatrix{6, 6, T, 36}
    r_apertures::SVector{6, T}
    e_apertures::SVector{6, T}
    kick_angle::SVector{2, T}
end

function Sextupole(L, k2; 
                   name::Union{Symbol,String}=:SEXT,
                   polynom_a=nothing, polynom_b=nothing,
                   max_order::Int=2, num_int_steps::Int=10, rad::Int=0,
                   fringe_entrance::Int=0, fringe_exit::Int=0,
                   t1=nothing, t2=nothing,
                   r1=nothing, r2=nothing,
                   r_apertures=nothing, e_apertures=nothing,
                   kick_angle=nothing)
    T = _promote_element_type(L, k2, polynom_a, polynom_b, t1, t2, r1, r2, r_apertures, e_apertures, kick_angle)
    Sextupole{T, Symbol}(Symbol(name), T(L), T(k2), 
        SVector{4,T}(_default_vec(polynom_a, T, Val(4))), 
        SVector{4,T}(_default_vec(polynom_b, T, Val(4))), 
        max_order, num_int_steps, rad, fringe_entrance, fringe_exit,
        SVector{6,T}(_default_vec(t1, T, Val(6))), 
        SVector{6,T}(_default_vec(t2, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r1, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r2, T, Val(6))), 
        SVector{6,T}(_default_vec(r_apertures, T, Val(6))), 
        SVector{6,T}(_default_vec(e_apertures, T, Val(6))), 
        SVector{2,T}(_default_vec(kick_angle, T, Val(2))))
end

function Adapt.adapt_structure(to, x::Sextupole)
    Sextupole(nothing, adapt(to, x.L), adapt(to, x.k2), adapt(to, x.polynom_a), adapt(to, x.polynom_b),
              x.max_order, x.num_int_steps, x.rad, x.fringe_entrance, x.fringe_exit,
              adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2),
              adapt(to, x.r_apertures), adapt(to, x.e_apertures), adapt(to, x.kick_angle))
end

"""
    Octupole{T, N}

An octupole magnet.
"""
struct Octupole{T, N} <: AbstractMagnet
    name::N
    L::T
    k3::T
    polynom_a::SVector{4, T}
    polynom_b::SVector{4, T}
    max_order::Int
    num_int_steps::Int
    rad::Int
    fringe_entrance::Int
    fringe_exit::Int
    t1::SVector{6, T}
    t2::SVector{6, T}
    r1::SMatrix{6, 6, T, 36}
    r2::SMatrix{6, 6, T, 36}
    r_apertures::SVector{6, T}
    e_apertures::SVector{6, T}
    kick_angle::SVector{2, T}
end

function Octupole(L, k3; 
                  name::Union{Symbol,String}=:OCT,
                  polynom_a=nothing, polynom_b=nothing,
                  max_order::Int=3, num_int_steps::Int=10, rad::Int=0,
                  fringe_entrance::Int=0, fringe_exit::Int=0,
                  t1=nothing, t2=nothing,
                  r1=nothing, r2=nothing,
                  r_apertures=nothing, e_apertures=nothing,
                  kick_angle=nothing)
    T = _promote_element_type(L, k3, polynom_a, polynom_b, t1, t2, r1, r2, r_apertures, e_apertures, kick_angle)
    Octupole{T, Symbol}(Symbol(name), T(L), T(k3), 
        SVector{4,T}(_default_vec(polynom_a, T, Val(4))), 
        SVector{4,T}(_default_vec(polynom_b, T, Val(4))), 
        max_order, num_int_steps, rad, fringe_entrance, fringe_exit,
        SVector{6,T}(_default_vec(t1, T, Val(6))), 
        SVector{6,T}(_default_vec(t2, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r1, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r2, T, Val(6))), 
        SVector{6,T}(_default_vec(r_apertures, T, Val(6))), 
        SVector{6,T}(_default_vec(e_apertures, T, Val(6))), 
        SVector{2,T}(_default_vec(kick_angle, T, Val(2))))
end

function Adapt.adapt_structure(to, x::Octupole)
    Octupole(nothing, adapt(to, x.L), adapt(to, x.k3), adapt(to, x.polynom_a), adapt(to, x.polynom_b),
             x.max_order, x.num_int_steps, x.rad, x.fringe_entrance, x.fringe_exit,
             adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2),
             adapt(to, x.r_apertures), adapt(to, x.e_apertures), adapt(to, x.kick_angle))
end

"""
    SBend{T, N}

A sector bending magnet.
"""
struct SBend{T, N} <: AbstractMagnet
    name::N
    L::T
    angle::T
    e1::T
    e2::T
    polynom_a::SVector{4, T}
    polynom_b::SVector{4, T}
    max_order::Int
    num_int_steps::Int
    rad::Int
    fint1::T
    fint2::T
    gap::T
    fringe_bend_entrance::Int
    fringe_bend_exit::Int
    fringe_quad_entrance::Int
    fringe_quad_exit::Int
    fringe_int_m0::SVector{5, T}
    fringe_int_p0::SVector{5, T}
    t1::SVector{6, T}
    t2::SVector{6, T}
    r1::SMatrix{6, 6, T, 36}
    r2::SMatrix{6, 6, T, 36}
    r_apertures::SVector{6, T}
    e_apertures::SVector{6, T}
    kick_angle::SVector{2, T}
end

function SBend(L, angle, e1=0.0, e2=0.0; 
               name::Union{Symbol,String}=:SBEND,
               polynom_a=nothing, polynom_b=nothing,
               max_order::Int=0, num_int_steps::Int=10, rad::Int=0,
               fint1=0.0, fint2=0.0, gap=0.0,
               fringe_bend_entrance::Int=1, fringe_bend_exit::Int=1,
               fringe_quad_entrance::Int=0, fringe_quad_exit::Int=0,
               fringe_int_m0=nothing, fringe_int_p0=nothing,
               t1=nothing, t2=nothing,
               r1=nothing, r2=nothing,
               r_apertures=nothing, e_apertures=nothing,
               kick_angle=nothing)
    T = _promote_element_type(L, angle, e1, e2, fint1, fint2, gap,
                     polynom_a, polynom_b, fringe_int_m0, fringe_int_p0,
                     t1, t2, r1, r2, r_apertures, e_apertures, kick_angle)
    SBend{T, Symbol}(Symbol(name), T(L), T(angle), T(e1), T(e2), 
        SVector{4,T}(_default_vec(polynom_a, T, Val(4))), 
        SVector{4,T}(_default_vec(polynom_b, T, Val(4))), 
        max_order, num_int_steps, rad,
        T(fint1), T(fint2), T(gap), fringe_bend_entrance, fringe_bend_exit, fringe_quad_entrance, fringe_quad_exit,
        SVector{5,T}(_default_vec(fringe_int_m0, T, Val(5))), 
        SVector{5,T}(_default_vec(fringe_int_p0, T, Val(5))),
        SVector{6,T}(_default_vec(t1, T, Val(6))), 
        SVector{6,T}(_default_vec(t2, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r1, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r2, T, Val(6))), 
        SVector{6,T}(_default_vec(r_apertures, T, Val(6))), 
        SVector{6,T}(_default_vec(e_apertures, T, Val(6))), 
        SVector{2,T}(_default_vec(kick_angle, T, Val(2))))
end

function Adapt.adapt_structure(to, x::SBend)
    SBend(nothing, adapt(to, x.L), adapt(to, x.angle), adapt(to, x.e1), adapt(to, x.e2),
          adapt(to, x.polynom_a), adapt(to, x.polynom_b), x.max_order, x.num_int_steps, x.rad,
          adapt(to, x.fint1), adapt(to, x.fint2), adapt(to, x.gap),
          x.fringe_bend_entrance, x.fringe_bend_exit, x.fringe_quad_entrance, x.fringe_quad_exit,
          adapt(to, x.fringe_int_m0), adapt(to, x.fringe_int_p0),
          adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2),
          adapt(to, x.r_apertures), adapt(to, x.e_apertures), adapt(to, x.kick_angle))
end

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

function RFCavity(L, volt, freq, lag=0.0; 
                  name::Union{Symbol,String}=:RFCA, h=1.0, philag=0.0, energy=0.0)
    T = _promote_element_type(L, volt, freq, lag, h, philag, energy)
    RFCavity{T, Symbol}(Symbol(name), T(L), T(volt), T(freq), T(h), T(lag), T(philag), T(energy))
end

function Adapt.adapt_structure(to, x::RFCavity)
    RFCavity(nothing, adapt(to, x.L), adapt(to, x.volt), adapt(to, x.freq), adapt(to, x.h),
             adapt(to, x.lag), adapt(to, x.philag), adapt(to, x.energy))
end

"""
    ThinMultipole{T, N}

A thin multipole element.
"""
struct ThinMultipole{T, N} <: AbstractElement
    name::N
    L::T
    polynom_a::SVector{4, T}
    polynom_b::SVector{4, T}
    max_order::Int
    num_int_steps::Int
    rad::Int
    fringe_entrance::Int
    fringe_exit::Int
    t1::SVector{6, T}
    t2::SVector{6, T}
    r1::SMatrix{6, 6, T, 36}
    r2::SMatrix{6, 6, T, 36}
    r_apertures::SVector{6, T}
    e_apertures::SVector{6, T}
    kick_angle::SVector{2, T}
end

function ThinMultipole(L, polynom_a, polynom_b; 
                       name::Union{Symbol,String}=:MULTIPOLE,
                       max_order::Int=1, num_int_steps::Int=1, rad::Int=0,
                       fringe_entrance::Int=0, fringe_exit::Int=0,
                       t1=nothing, t2=nothing,
                       r1=nothing, r2=nothing,
                       r_apertures=nothing, e_apertures=nothing,
                       kick_angle=nothing)
    T = _promote_element_type(L, polynom_a, polynom_b, t1, t2, r1, r2, r_apertures, e_apertures, kick_angle)
    ThinMultipole{T, Symbol}(Symbol(name), T(L), 
        SVector{4,T}(_default_vec(polynom_a, T, Val(4))), 
        SVector{4,T}(_default_vec(polynom_b, T, Val(4))), 
        max_order, num_int_steps, rad, fringe_entrance, fringe_exit,
        SVector{6,T}(_default_vec(t1, T, Val(6))), 
        SVector{6,T}(_default_vec(t2, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r1, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r2, T, Val(6))), 
        SVector{6,T}(_default_vec(r_apertures, T, Val(6))), 
        SVector{6,T}(_default_vec(e_apertures, T, Val(6))), 
        SVector{2,T}(_default_vec(kick_angle, T, Val(2))))
end

function Adapt.adapt_structure(to, x::ThinMultipole)
    ThinMultipole(nothing, adapt(to, x.L), adapt(to, x.polynom_a), adapt(to, x.polynom_b),
                  x.max_order, x.num_int_steps, x.rad, x.fringe_entrance, x.fringe_exit,
                  adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2),
                  adapt(to, x.r_apertures), adapt(to, x.e_apertures), adapt(to, x.kick_angle))
end

"""
    Solenoid{T, N}

A solenoid element.
"""
struct Solenoid{T, N} <: AbstractMagnet
    name::N
    L::T
    ks::T
    t1::SVector{6, T}
    t2::SVector{6, T}
    r1::SMatrix{6, 6, T, 36}
    r2::SMatrix{6, 6, T, 36}
end

function Solenoid(L, ks; 
                  name::Union{Symbol,String}=:SOLENOID,
                  t1=nothing, t2=nothing,
                  r1=nothing, r2=nothing)
    T = _promote_element_type(L, ks, t1, t2, r1, r2)
    Solenoid{T, Symbol}(Symbol(name), T(L), T(ks), 
        SVector{6,T}(_default_vec(t1, T, Val(6))), 
        SVector{6,T}(_default_vec(t2, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r1, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r2, T, Val(6))))
end

function Adapt.adapt_structure(to, x::Solenoid)
    Solenoid(nothing, adapt(to, x.L), adapt(to, x.ks), adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2))
end

"""
    Corrector{T, N}

A corrector element.
"""
struct Corrector{T, N} <: AbstractMagnet
    name::N
    L::T
    xkick::T
    ykick::T
    t1::SVector{6, T}
    t2::SVector{6, T}
    r1::SMatrix{6, 6, T, 36}
    r2::SMatrix{6, 6, T, 36}
end

function Corrector(L, hk, vk; 
                   name::Union{Symbol,String}=:CORRECTOR,
                   t1=nothing, t2=nothing,
                   r1=nothing, r2=nothing)
    T = _promote_element_type(L, hk, vk, t1, t2, r1, r2)
    Corrector{T, Symbol}(Symbol(name), T(L), T(hk), T(vk), 
        SVector{6,T}(_default_vec(t1, T, Val(6))), 
        SVector{6,T}(_default_vec(t2, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r1, T, Val(6))), 
        SMatrix{6,6,T}(_default_mat(r2, T, Val(6))))
end

function Adapt.adapt_structure(to, x::Corrector)
    Corrector(nothing, adapt(to, x.L), adapt(to, x.xkick), adapt(to, x.ykick),
              adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2))
end


