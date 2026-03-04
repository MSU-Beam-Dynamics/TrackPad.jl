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

function SBend(L, angle, e1 = 0.0, e2 = 0.0;
               name::Union{Symbol, String} = :SBEND,
               polynom_a = nothing, polynom_b = nothing,
               max_order::Int = 0, num_int_steps::Int = 10, rad::Int = 0,
               fint1 = 0.0, fint2 = 0.0, gap = 0.0,
               fringe_bend_entrance::Int = 1, fringe_bend_exit::Int = 1,
               fringe_quad_entrance::Int = 0, fringe_quad_exit::Int = 0,
               fringe_int_m0 = nothing, fringe_int_p0 = nothing,
               t1 = nothing, t2 = nothing,
               r1 = nothing, r2 = nothing,
               r_apertures = nothing, e_apertures = nothing,
               kick_angle = nothing)
    T = _promote_element_type(L, angle, e1, e2, fint1, fint2, gap,
                     polynom_a, polynom_b, fringe_int_m0, fringe_int_p0,
                     t1, t2, r1, r2, r_apertures, e_apertures, kick_angle)
    SBend{T, Symbol}(Symbol(name), T(L), T(angle), T(e1), T(e2),
        SVector{4, T}(_default_vec(polynom_a, T, Val(4))),
        SVector{4, T}(_default_vec(polynom_b, T, Val(4))),
        max_order, num_int_steps, rad,
        T(fint1), T(fint2), T(gap), fringe_bend_entrance, fringe_bend_exit, fringe_quad_entrance, fringe_quad_exit,
        SVector{5, T}(_default_vec(fringe_int_m0, T, Val(5))),
        SVector{5, T}(_default_vec(fringe_int_p0, T, Val(5))),
        SVector{6, T}(_default_vec(t1, T, Val(6))),
        SVector{6, T}(_default_vec(t2, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r1, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r2, T, Val(6))),
        SVector{6, T}(_default_vec(r_apertures, T, Val(6))),
        SVector{6, T}(_default_vec(e_apertures, T, Val(6))),
        SVector{2, T}(_default_vec(kick_angle, T, Val(2))))
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
    RBend(L, angle; kwargs...)

Rectangular bend convenience constructor (sets `e1=e2=angle/2`).
"""
RBend(L, angle; kwargs...) = SBend(L, angle, angle / 2, angle / 2; kwargs...)

"""
    ExactSBend{T, N}

Exact Sector Bend element. Uses exact geometric map instead of drift-kick-drift.
"""
struct ExactSBend{T, N} <: AbstractMagnet
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
    gk::T
end

function ExactSBend(L, angle, e1 = 0.0, e2 = 0.0;
                    name::Union{Symbol, String} = :ESBEND,
                    polynom_a = nothing, polynom_b = nothing,
                    max_order::Int = 0, num_int_steps::Int = 10, rad::Int = 0,
                    fint1 = 0.0, fint2 = 0.0, gap = 0.0,
                    fringe_bend_entrance::Int = 1, fringe_bend_exit::Int = 1,
                    fringe_quad_entrance::Int = 0, fringe_quad_exit::Int = 0,
                    fringe_int_m0 = nothing, fringe_int_p0 = nothing,
                    t1 = nothing, t2 = nothing,
                    r1 = nothing, r2 = nothing,
                    r_apertures = nothing, e_apertures = nothing,
                    kick_angle = nothing,
                    gk = 0.0)
    T = _promote_element_type(L, angle, e1, e2, fint1, fint2, gap,
                     polynom_a, polynom_b, fringe_int_m0, fringe_int_p0,
                     t1, t2, r1, r2, r_apertures, e_apertures, kick_angle, gk)
    ExactSBend{T, Symbol}(Symbol(name), T(L), T(angle), T(e1), T(e2),
        SVector{4, T}(_default_vec(polynom_a, T, Val(4))),
        SVector{4, T}(_default_vec(polynom_b, T, Val(4))),
        max_order, num_int_steps, rad,
        T(fint1), T(fint2), T(gap), fringe_bend_entrance, fringe_bend_exit, fringe_quad_entrance, fringe_quad_exit,
        SVector{5, T}(_default_vec(fringe_int_m0, T, Val(5))),
        SVector{5, T}(_default_vec(fringe_int_p0, T, Val(5))),
        SVector{6, T}(_default_vec(t1, T, Val(6))),
        SVector{6, T}(_default_vec(t2, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r1, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r2, T, Val(6))),
        SVector{6, T}(_default_vec(r_apertures, T, Val(6))),
        SVector{6, T}(_default_vec(e_apertures, T, Val(6))),
        SVector{2, T}(_default_vec(kick_angle, T, Val(2))),
        T(gk))
end

function Adapt.adapt_structure(to, x::ExactSBend)
    ExactSBend(nothing, adapt(to, x.L), adapt(to, x.angle), adapt(to, x.e1), adapt(to, x.e2),
               adapt(to, x.polynom_a), adapt(to, x.polynom_b), x.max_order, x.num_int_steps, x.rad,
               adapt(to, x.fint1), adapt(to, x.fint2), adapt(to, x.gap),
               x.fringe_bend_entrance, x.fringe_bend_exit, x.fringe_quad_entrance, x.fringe_quad_exit,
               adapt(to, x.fringe_int_m0), adapt(to, x.fringe_int_p0),
               adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2),
               adapt(to, x.r_apertures), adapt(to, x.e_apertures), adapt(to, x.kick_angle), adapt(to, x.gk))
end
