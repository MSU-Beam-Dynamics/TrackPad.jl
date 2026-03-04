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
               name::Union{Symbol, String} = :DRIFT,
               t1 = nothing, t2 = nothing,
               r1 = nothing, r2 = nothing,
               r_apertures = nothing, e_apertures = nothing)
    T = _promote_element_type(L, t1, t2, r1, r2, r_apertures, e_apertures)
    Drift{T, Symbol}(Symbol(name), T(L),
        SVector{6, T}(_default_vec(t1, T, Val(6))),
        SVector{6, T}(_default_vec(t2, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r1, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r2, T, Val(6))),
        SVector{6, T}(_default_vec(r_apertures, T, Val(6))),
        SVector{6, T}(_default_vec(e_apertures, T, Val(6))))
end

function Adapt.adapt_structure(to, x::Drift)
    Drift(nothing, adapt(to, x.L), adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2), adapt(to, x.r_apertures), adapt(to, x.e_apertures))
end
