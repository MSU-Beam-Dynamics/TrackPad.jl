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
                    name::Union{Symbol, String} = :QUAD,
                    polynom_a = nothing, polynom_b = nothing,
                    max_order::Int = 1, num_int_steps::Int = 10, rad::Int = 0,
                    fringe_entrance::Int = 0, fringe_exit::Int = 0,
                    t1 = nothing, t2 = nothing,
                    r1 = nothing, r2 = nothing,
                    r_apertures = nothing, e_apertures = nothing,
                    kick_angle = nothing)
    T = _promote_element_type(L, k1, polynom_a, polynom_b, t1, t2, r1, r2, r_apertures, e_apertures, kick_angle)
    Quadrupole{T, Symbol}(Symbol(name), T(L), T(k1),
        SVector{4, T}(_default_vec(polynom_a, T, Val(4))),
        SVector{4, T}(_default_vec(polynom_b, T, Val(4))),
        max_order, num_int_steps, rad, fringe_entrance, fringe_exit,
        SVector{6, T}(_default_vec(t1, T, Val(6))),
        SVector{6, T}(_default_vec(t2, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r1, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r2, T, Val(6))),
        SVector{6, T}(_default_vec(r_apertures, T, Val(6))),
        SVector{6, T}(_default_vec(e_apertures, T, Val(6))),
        SVector{2, T}(_default_vec(kick_angle, T, Val(2))))
end

function Adapt.adapt_structure(to, x::Quadrupole)
    Quadrupole(nothing, adapt(to, x.L), adapt(to, x.k1), adapt(to, x.polynom_a), adapt(to, x.polynom_b),
               x.max_order, x.num_int_steps, x.rad, x.fringe_entrance, x.fringe_exit,
               adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2),
               adapt(to, x.r_apertures), adapt(to, x.e_apertures), adapt(to, x.kick_angle))
end
