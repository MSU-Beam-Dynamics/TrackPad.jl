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
                  name::Union{Symbol, String} = :SOLENOID,
                  t1 = nothing, t2 = nothing,
                  r1 = nothing, r2 = nothing)
    T = _promote_element_type(L, ks, t1, t2, r1, r2)
    Solenoid{T, Symbol}(Symbol(name), T(L), T(ks),
        SVector{6, T}(_default_vec(t1, T, Val(6))),
        SVector{6, T}(_default_vec(t2, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r1, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r2, T, Val(6))))
end

function Adapt.adapt_structure(to, x::Solenoid)
    Solenoid(nothing, adapt(to, x.L), adapt(to, x.ks), adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2))
end
