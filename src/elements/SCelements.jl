"""
    DriftSC{T, N}

Canonical drift element with space-charge metadata.
"""
struct DriftSC{T, N} <: AbstractDrift
    name::N
    L::T
    t1::SVector{6, T}
    t2::SVector{6, T}
    r1::SMatrix{6, 6, T, 36}
    r2::SMatrix{6, 6, T, 36}
    r_apertures::SVector{6, T}
    e_apertures::SVector{6, T}
    a::T
    b::T
    Nl::Int
    Nm::Int
    Nsteps::Int
end

function DriftSC(L;
                 name::Union{Symbol, String} = :DRIFT_SC,
                 t1 = nothing, t2 = nothing,
                 r1 = nothing, r2 = nothing,
                 r_apertures = nothing, e_apertures = nothing,
                 a = 1.0, b = 1.0,
                 Nl::Int = 10, Nm::Int = 10, Nsteps::Int = 1)
    T = _promote_element_type(L, t1, t2, r1, r2, r_apertures, e_apertures, a, b)
    DriftSC{T, Symbol}(Symbol(name), T(L),
        SVector{6, T}(_default_vec(t1, T, Val(6))),
        SVector{6, T}(_default_vec(t2, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r1, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r2, T, Val(6))),
        SVector{6, T}(_default_vec(r_apertures, T, Val(6))),
        SVector{6, T}(_default_vec(e_apertures, T, Val(6))),
        T(a), T(b), Nl, Nm, Nsteps)
end

function Adapt.adapt_structure(to, x::DriftSC)
    DriftSC(adapt(to, x.L); name=x.name, t1=adapt(to, x.t1), t2=adapt(to, x.t2),
            r1=adapt(to, x.r1), r2=adapt(to, x.r2), r_apertures=adapt(to, x.r_apertures), e_apertures=adapt(to, x.e_apertures),
            a=adapt(to, x.a), b=adapt(to, x.b), Nl=x.Nl, Nm=x.Nm, Nsteps=x.Nsteps)
end

"""
    QuadrupoleSC{T, N}

Canonical quadrupole element with space-charge metadata.
"""
struct QuadrupoleSC{T, N} <: AbstractMagnet
    name::N
    L::T
    k0::T
    k1::T
    k2::T
    k3::T
    polynom_a::SVector{4, T}
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
    a::T
    b::T
    Nl::Int
    Nm::Int
    Nsteps::Int
end

function QuadrupoleSC(L;
                      name::Union{Symbol, String} = :KQUAD_SC,
                      k0 = 0.0, k1 = 0.0, k2 = 0.0, k3 = 0.0,
                      polynom_a = nothing,
                      max_order::Int = 1, num_int_steps::Int = 10, rad::Int = 0,
                      fringe_entrance::Int = 0, fringe_exit::Int = 0,
                      t1 = nothing, t2 = nothing,
                      r1 = nothing, r2 = nothing,
                      r_apertures = nothing, e_apertures = nothing,
                      kick_angle = nothing,
                      a = 1.0, b = 1.0, Nl::Int = 10, Nm::Int = 10, Nsteps::Int = 1)
    T = _promote_element_type(L, k0, k1, k2, k3, polynom_a, t1, t2, r1, r2, r_apertures, e_apertures, kick_angle, a, b)
    QuadrupoleSC{T, Symbol}(Symbol(name), T(L), T(k0), T(k1), T(k2), T(k3),
        SVector{4, T}(_default_vec(polynom_a, T, Val(4))),
        max_order, num_int_steps, rad, fringe_entrance, fringe_exit,
        SVector{6, T}(_default_vec(t1, T, Val(6))),
        SVector{6, T}(_default_vec(t2, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r1, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r2, T, Val(6))),
        SVector{6, T}(_default_vec(r_apertures, T, Val(6))),
        SVector{6, T}(_default_vec(e_apertures, T, Val(6))),
        SVector{2, T}(_default_vec(kick_angle, T, Val(2))),
        T(a), T(b), Nl, Nm, Nsteps)
end

function Adapt.adapt_structure(to, x::QuadrupoleSC)
    QuadrupoleSC(adapt(to, x.L); name=x.name, k0=adapt(to, x.k0), k1=adapt(to, x.k1), k2=adapt(to, x.k2), k3=adapt(to, x.k3),
                 polynom_a=adapt(to, x.polynom_a), max_order=x.max_order, num_int_steps=x.num_int_steps, rad=x.rad,
                 fringe_entrance=x.fringe_entrance, fringe_exit=x.fringe_exit, t1=adapt(to, x.t1), t2=adapt(to, x.t2),
                 r1=adapt(to, x.r1), r2=adapt(to, x.r2), r_apertures=adapt(to, x.r_apertures), e_apertures=adapt(to, x.e_apertures),
                 kick_angle=adapt(to, x.kick_angle), a=adapt(to, x.a), b=adapt(to, x.b), Nl=x.Nl, Nm=x.Nm, Nsteps=x.Nsteps)
end

"""
    SextupoleSC{T, N}

Canonical sextupole element with space-charge metadata.
"""
struct SextupoleSC{T, N} <: AbstractMagnet
    name::N
    L::T
    k0::T
    k1::T
    k2::T
    k3::T
    polynom_a::SVector{4, T}
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
    a::T
    b::T
    Nl::Int
    Nm::Int
    Nsteps::Int
end

function SextupoleSC(L;
                     name::Union{Symbol, String} = :KSEXT_SC,
                     k0 = 0.0, k1 = 0.0, k2 = 0.0, k3 = 0.0,
                     polynom_a = nothing,
                     max_order::Int = 2, num_int_steps::Int = 10, rad::Int = 0,
                     fringe_entrance::Int = 0, fringe_exit::Int = 0,
                     t1 = nothing, t2 = nothing,
                     r1 = nothing, r2 = nothing,
                     r_apertures = nothing, e_apertures = nothing,
                     kick_angle = nothing,
                     a = 1.0, b = 1.0, Nl::Int = 10, Nm::Int = 10, Nsteps::Int = 1)
    T = _promote_element_type(L, k0, k1, k2, k3, polynom_a, t1, t2, r1, r2, r_apertures, e_apertures, kick_angle, a, b)
    SextupoleSC{T, Symbol}(Symbol(name), T(L), T(k0), T(k1), T(k2), T(k3),
        SVector{4, T}(_default_vec(polynom_a, T, Val(4))),
        max_order, num_int_steps, rad, fringe_entrance, fringe_exit,
        SVector{6, T}(_default_vec(t1, T, Val(6))),
        SVector{6, T}(_default_vec(t2, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r1, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r2, T, Val(6))),
        SVector{6, T}(_default_vec(r_apertures, T, Val(6))),
        SVector{6, T}(_default_vec(e_apertures, T, Val(6))),
        SVector{2, T}(_default_vec(kick_angle, T, Val(2))),
        T(a), T(b), Nl, Nm, Nsteps)
end

function Adapt.adapt_structure(to, x::SextupoleSC)
    SextupoleSC(adapt(to, x.L); name=x.name, k0=adapt(to, x.k0), k1=adapt(to, x.k1), k2=adapt(to, x.k2), k3=adapt(to, x.k3),
                polynom_a=adapt(to, x.polynom_a), max_order=x.max_order, num_int_steps=x.num_int_steps, rad=x.rad,
                fringe_entrance=x.fringe_entrance, fringe_exit=x.fringe_exit, t1=adapt(to, x.t1), t2=adapt(to, x.t2),
                r1=adapt(to, x.r1), r2=adapt(to, x.r2), r_apertures=adapt(to, x.r_apertures), e_apertures=adapt(to, x.e_apertures),
                kick_angle=adapt(to, x.kick_angle), a=adapt(to, x.a), b=adapt(to, x.b), Nl=x.Nl, Nm=x.Nm, Nsteps=x.Nsteps)
end

"""
    OctupoleSC{T, N}

Canonical octupole element with space-charge metadata.
"""
struct OctupoleSC{T, N} <: AbstractMagnet
    name::N
    L::T
    k0::T
    k1::T
    k2::T
    k3::T
    polynom_a::SVector{4, T}
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
    a::T
    b::T
    Nl::Int
    Nm::Int
    Nsteps::Int
end

function OctupoleSC(L;
                    name::Union{Symbol, String} = :KOCT_SC,
                    k0 = 0.0, k1 = 0.0, k2 = 0.0, k3 = 0.0,
                    polynom_a = nothing,
                    max_order::Int = 3, num_int_steps::Int = 10, rad::Int = 0,
                    fringe_entrance::Int = 0, fringe_exit::Int = 0,
                    t1 = nothing, t2 = nothing,
                    r1 = nothing, r2 = nothing,
                    r_apertures = nothing, e_apertures = nothing,
                    kick_angle = nothing,
                    a = 1.0, b = 1.0, Nl::Int = 10, Nm::Int = 10, Nsteps::Int = 1)
    T = _promote_element_type(L, k0, k1, k2, k3, polynom_a, t1, t2, r1, r2, r_apertures, e_apertures, kick_angle, a, b)
    OctupoleSC{T, Symbol}(Symbol(name), T(L), T(k0), T(k1), T(k2), T(k3),
        SVector{4, T}(_default_vec(polynom_a, T, Val(4))),
        max_order, num_int_steps, rad, fringe_entrance, fringe_exit,
        SVector{6, T}(_default_vec(t1, T, Val(6))),
        SVector{6, T}(_default_vec(t2, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r1, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r2, T, Val(6))),
        SVector{6, T}(_default_vec(r_apertures, T, Val(6))),
        SVector{6, T}(_default_vec(e_apertures, T, Val(6))),
        SVector{2, T}(_default_vec(kick_angle, T, Val(2))),
        T(a), T(b), Nl, Nm, Nsteps)
end

function Adapt.adapt_structure(to, x::OctupoleSC)
    OctupoleSC(adapt(to, x.L); name=x.name, k0=adapt(to, x.k0), k1=adapt(to, x.k1), k2=adapt(to, x.k2), k3=adapt(to, x.k3),
               polynom_a=adapt(to, x.polynom_a), max_order=x.max_order, num_int_steps=x.num_int_steps, rad=x.rad,
               fringe_entrance=x.fringe_entrance, fringe_exit=x.fringe_exit, t1=adapt(to, x.t1), t2=adapt(to, x.t2),
               r1=adapt(to, x.r1), r2=adapt(to, x.r2), r_apertures=adapt(to, x.r_apertures), e_apertures=adapt(to, x.e_apertures),
               kick_angle=adapt(to, x.kick_angle), a=adapt(to, x.a), b=adapt(to, x.b), Nl=x.Nl, Nm=x.Nm, Nsteps=x.Nsteps)
end

"""
    SBendSC{T, N}

Canonical sector bend with space-charge metadata.
"""
struct SBendSC{T, N} <: AbstractMagnet
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
    a::T
    b::T
    Nl::Int
    Nm::Int
    Nsteps::Int
end

function SBendSC(L, angle, e1 = 0.0, e2 = 0.0;
                 name::Union{Symbol, String} = :SBEND_SC,
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
                 a = 1.0, b = 1.0, Nl::Int = 10, Nm::Int = 10, Nsteps::Int = 1)
    T = _promote_element_type(L, angle, e1, e2, fint1, fint2, gap, polynom_a, polynom_b,
                              fringe_int_m0, fringe_int_p0, t1, t2, r1, r2, r_apertures, e_apertures,
                              kick_angle, a, b)
    SBendSC{T, Symbol}(Symbol(name), T(L), T(angle), T(e1), T(e2),
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
        T(a), T(b), Nl, Nm, Nsteps)
end

function Adapt.adapt_structure(to, x::SBendSC)
    SBendSC(adapt(to, x.L), adapt(to, x.angle), adapt(to, x.e1), adapt(to, x.e2);
            name=x.name, polynom_a=adapt(to, x.polynom_a), polynom_b=adapt(to, x.polynom_b),
            max_order=x.max_order, num_int_steps=x.num_int_steps, rad=x.rad,
            fint1=adapt(to, x.fint1), fint2=adapt(to, x.fint2), gap=adapt(to, x.gap),
            fringe_bend_entrance=x.fringe_bend_entrance, fringe_bend_exit=x.fringe_bend_exit,
            fringe_quad_entrance=x.fringe_quad_entrance, fringe_quad_exit=x.fringe_quad_exit,
            fringe_int_m0=adapt(to, x.fringe_int_m0), fringe_int_p0=adapt(to, x.fringe_int_p0),
            t1=adapt(to, x.t1), t2=adapt(to, x.t2), r1=adapt(to, x.r1), r2=adapt(to, x.r2),
            r_apertures=adapt(to, x.r_apertures), e_apertures=adapt(to, x.e_apertures),
            kick_angle=adapt(to, x.kick_angle), a=adapt(to, x.a), b=adapt(to, x.b),
            Nl=x.Nl, Nm=x.Nm, Nsteps=x.Nsteps)
end

"""
    RBendSC(L, angle; kwargs...)

Rectangular bend with space-charge metadata (sets `e1=e2=angle/2`).
"""
RBendSC(L, angle; kwargs...) = SBendSC(L, angle, angle / 2, angle / 2; kwargs...)

"""
    ERBend(L, angle; kwargs...)

Exact rectangular bend convenience constructor (sets `e1=e2=angle/2`).
"""
ERBend(L, angle; kwargs...) = ExactSBend(L, angle, angle / 2, angle / 2; kwargs...)

"""
    LBend{T, N}

Linear-map bend element.
"""
struct LBend{T, N} <: AbstractMagnet
    name::N
    L::T
    angle::T
    e1::T
    e2::T
    K::T
    by_error::T
    fint1::T
    fint2::T
    full_gap::T
    t1::SVector{6, T}
    t2::SVector{6, T}
    r1::SMatrix{6, 6, T, 36}
    r2::SMatrix{6, 6, T, 36}
    r_apertures::SVector{6, T}
    e_apertures::SVector{6, T}
end

function LBend(L, angle, e1 = 0.0, e2 = 0.0;
               name::Union{Symbol, String} = :LBEND,
               K = 0.0, by_error = 0.0, fint1 = 0.0, fint2 = 0.0, full_gap = 0.0,
               t1 = nothing, t2 = nothing,
               r1 = nothing, r2 = nothing,
               r_apertures = nothing, e_apertures = nothing)
    T = _promote_element_type(L, angle, e1, e2, K, by_error, fint1, fint2, full_gap, t1, t2, r1, r2, r_apertures, e_apertures)
    LBend{T, Symbol}(Symbol(name), T(L), T(angle), T(e1), T(e2), T(K), T(by_error), T(fint1), T(fint2), T(full_gap),
        SVector{6, T}(_default_vec(t1, T, Val(6))),
        SVector{6, T}(_default_vec(t2, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r1, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r2, T, Val(6))),
        SVector{6, T}(_default_vec(r_apertures, T, Val(6))),
        SVector{6, T}(_default_vec(e_apertures, T, Val(6))))
end

function Adapt.adapt_structure(to, x::LBend)
    LBend(adapt(to, x.L), adapt(to, x.angle), adapt(to, x.e1), adapt(to, x.e2);
          name=x.name, K=adapt(to, x.K), by_error=adapt(to, x.by_error),
          fint1=adapt(to, x.fint1), fint2=adapt(to, x.fint2), full_gap=adapt(to, x.full_gap),
          t1=adapt(to, x.t1), t2=adapt(to, x.t2), r1=adapt(to, x.r1), r2=adapt(to, x.r2),
          r_apertures=adapt(to, x.r_apertures), e_apertures=adapt(to, x.e_apertures))
end

"""
    SpaceCharge{T, N}

Integrated spectral space-charge element metadata.
"""
struct SpaceCharge{T, N} <: AbstractElement
    name::N
    L::T
    effective_len::T
    Nl::Int
    Nm::Int
    a::T
    b::T
end

function SpaceCharge(L;
                     name::Union{Symbol, String} = :SPACECHARGE,
                     effective_len = 0.0, Nl::Int = 10, Nm::Int = 10, a = 1.0, b = 1.0)
    T = _promote_element_type(L, effective_len, a, b)
    SpaceCharge{T, Symbol}(Symbol(name), T(L), T(effective_len), Nl, Nm, T(a), T(b))
end

function Adapt.adapt_structure(to, x::SpaceCharge)
    SpaceCharge(adapt(to, x.L); name=x.name, effective_len=adapt(to, x.effective_len), Nl=x.Nl, Nm=x.Nm, a=adapt(to, x.a), b=adapt(to, x.b))
end

"""
    Translation{T, N}

Coordinate translation helper element.
"""
struct Translation{T, N} <: AbstractElement
    name::N
    L::T
    dx::T
    dy::T
    ds::T
end

function Translation(L;
                     name::Union{Symbol, String} = :TRANSLATION,
                     dx = 0.0, dy = 0.0, ds = 0.0)
    T = _promote_element_type(L, dx, dy, ds)
    Translation{T, Symbol}(Symbol(name), T(L), T(dx), T(dy), T(ds))
end

function Adapt.adapt_structure(to, x::Translation)
    Translation(adapt(to, x.L); name=x.name, dx=adapt(to, x.dx), dy=adapt(to, x.dy), ds=adapt(to, x.ds))
end

"""
    YRotation{T, N}

Rotation-about-y helper element.
"""
struct YRotation{T, N} <: AbstractElement
    name::N
    L::T
    angle::T
end

function YRotation(L;
                   name::Union{Symbol, String} = :YROTATION,
                   angle = 0.0)
    T = _promote_element_type(L, angle)
    YRotation{T, Symbol}(Symbol(name), T(L), T(angle))
end

function Adapt.adapt_structure(to, x::YRotation)
    YRotation(adapt(to, x.L); name=x.name, angle=adapt(to, x.angle))
end
